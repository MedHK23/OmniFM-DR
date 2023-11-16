# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from io import BytesIO

import os
import math
import logging
import random
import warnings
from nick_utils.RLE import test

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
from utils.vision_helper import RandomAugment
import utils.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:

        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    code_masks = None
    if samples[0].get("code_mask", None) is not None:
        code_masks = torch.cat([sample['code_mask'] for sample in samples])

    conf = torch.cat([s['conf'] for s in samples], dim=0)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    multi_hot_label = None
    # multi_hot_label = torch.cat([sample['multi_hot_label'] for sample in samples])
    # print(f"[-----------] collect, multi_hot_label: {multi_hot_label.shape}")
    
    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "code_masks": code_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
        "conf": conf,
        "multi_hot_label": multi_hot_label,
    }

    return batch


class DRJointDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        mimic_t2i=None,
        mimic_i2t=None,
        max_src_length=128,
        max_tgt_length=30,
        seed=7,
        code_dict_size=8192,
        num_bins=1000,
        patch_image_size=384,
        code_image_size=128,
        # pure_text_dataset=None,
        # pure_image_dataset=None,
        # detection_dataset=None,
        vqa_dataset=None,
        vqa_multiple_dataset=None,
        visual_ground_dataset=None,
        visual_ground_multiple_dataset=None,
        caption_dataset=None,
        seg_dataset=None,
        MIMIC_path=None,
        private_data_path=None,
        negative_dataset=None,
        vqa_attritube_dataset_positive=None,
        vqa_attritube_dataset_negative=None,
        all_object_list=None,
        all_caption_list=None,
        type2ans_dict=None,
        ans2type_dict=None,
        max_image_size=512,
        mask_ratio=0.3,
        random_ratio=0.0,
        keep_ratio=0.0,
        mask_length="span-poisson",
        poisson_lambda=3.0,
        replace_length=1
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict, mimic_t2i=mimic_t2i, mimic_i2t=mimic_i2t)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.seed = seed
        self.code_dict_size = code_dict_size
        self.num_bins = num_bins
        self.patch_image_size = patch_image_size
        self.code_image_size = code_image_size

        # self.pure_text_dataset = pure_text_dataset
        # self.pure_image_dataset = pure_image_dataset
        # self.detection_dataset = detection_dataset
        
        self.caption_dataset = caption_dataset
        self.vqa_dataset = vqa_dataset
        self.vqa_multiple_dataset = vqa_multiple_dataset
        self.visual_ground_dataset = visual_ground_dataset
        self.visual_ground_multiple_dataset = visual_ground_multiple_dataset
        self.seg_dataset = seg_dataset
        self.MIMIC_path = MIMIC_path
        self.private_data_path = private_data_path
        self.negative_dataset = negative_dataset
        self.vqa_attritube_dataset_positive = vqa_attritube_dataset_positive
        self.vqa_attritube_dataset_negative = vqa_attritube_dataset_negative
        
        self.epoch = 0

        self.all_object_list = all_object_list
        self.all_caption_list = all_caption_list
        self.type2ans_dict = type2ans_dict
        self.ans2type_dict = ans2type_dict

        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.keep_ratio = keep_ratio
        self.mask_length = mask_length
        self.poisson_lambda = poisson_lambda
        self.replace_length = replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_idx = src_dict.index("<mask>")
        self.mask_whole_word = (
            get_whole_word_mask(self.bpe, self.src_dict)
            if self.mask_length != "subword"
            else None
        )
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda
            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.pos_tgt_item = self.encode_text(" yes")
        self.neg_tgt_item = self.encode_text(" no")

        self.mask_left = self.mask_top = int(0.5 * self.code_image_size)
        self.mask_right = self.mask_bottom = int(1.5 * self.code_image_size)
        self.mask_ids = [
            i*self.code_image_size*2+j
            for i in range(self.code_image_size*2) for j in range(self.code_image_size*2)
            if not (self.mask_left <= i < self.mask_right and self.mask_top <= j < self.mask_bottom)
        ]

        scales = np.arange(patch_image_size-32, patch_image_size+32).tolist()
        # for image-text pair
        self.patch_resize_transform = transforms.Compose([
            T.RandomResize(scales, max_size=patch_image_size+32),
            transforms.CenterCrop(patch_image_size),
            RandomAugment(2, 2, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for pure image
        self.patch_crop_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for detection
        self.detection_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.LargeScaleJitter(output_size=self.code_image_size*2, aug_scale_min=1.0, aug_scale_max=1.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])
        # for visual grounding
        self.visual_grounding_transform = T.Compose([
            T.RandomResize(scales, max_size=672),
            T.ObjectCenterCrop((patch_image_size, patch_image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])
        # for positioning
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])
        
        # disease list
        self.diseases_vg = ['Infiltration', 'Consolidation', 'Pleural Effusion', 'Cardiomegaly', 'Edema', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Lung Opacity', 'Nodule', 'Mass']
        self.diseases_mimic = ['Atelectasis', 'Lung Lesion', 'Pneumonia', 'Fracture', 'Cardiomegaly', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Consolidation', 'Lung Opacity', 'Edema', 'No Finding']
        # vg possible location list, e.g., in the lower left
        self.locations = ['on the right upper side', 'on the left lower side', 'on the left upper side', 'on the middle right side', 'on the middle left side', 'on the right lower side']
        self.mimic_long_tail = ['Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation', 'Edema', \
            'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion', \
            'Lung Opacity', 'Mass', 'No Finding', 'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', \
            'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta']

        self.debug = False

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def process_detection(self, index):
        detection_prompt_pool = [
            ' Could you locating and classifying all lesions in the DR chest image?',
            ' Could you identifying all abnormal locations and their categories in the DR chest image?',
            ' Could you show all lesions in the image?',
            ' what are the lesions in the image?'
        ][:1]
        prompt = detection_prompt_pool[random.randint(0, len(detection_prompt_pool)-1)]
        # 
        # # print(f"self.dataset[index]: {len(self.dataset[index])}")
        # # print(self.dataset[index])
        image_id, image_str, text, label = self.dataset[index]
        with open(os.path.join(self.private_data_path, image_str)) as f:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(f.read()))).convert("RGB")
            
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        label_list = label.strip().split('&&')
        ## box order: random (pix2pix: random; ofa: by size)
        random.shuffle(label_list)
        for label in label_list:
            x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
            boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
            boxes_target["labels"].append(cat)
            boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
        boxes_target["labels"] = np.array(boxes_target["labels"])
        boxes_target["area"] = torch.tensor(boxes_target["area"])

        patch_image, boxes_target = self.detection_transform(image, boxes_target)
        patch_mask = torch.tensor([True])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        quant_boxes = []
        for i, box in enumerate(boxes_target["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
            quant_boxes.append(self.bpe.encode(' {}'.format(boxes_target["labels"][i])))
        src_item = self.encode_text(prompt)
        tgt_item = self.encode_text(' '.join(quant_boxes), use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        # print("Detection: ")
        # print(prompt)
        # print(example)
        return [example]

    def process_vqa(self, index, the_dataset, single=True):
        
        ## export as configuration?
        vqa_single_prompt_pool = [
            'Is {} present in this image?',
            'Does the image show any signs or symptoms of {}?',
            'Is the presence of {} possible in the image?',
        ][:1]
        vqa_multi_prompt_pool = [
            'What diseases are included in this image?',
            'Which illnesses are depicted in this image?',
            'Can you tell me what diseases are visible in this image?',
        ][:1]
        prompt_pool = vqa_single_prompt_pool if single else vqa_multi_prompt_pool
        # 0, 5, 2, 3
        uniq_id, image_str, base_question, ref = the_dataset[index]
        question = prompt_pool[random.randint(0, len(prompt_pool) - 1)]
        # print(f"Base question: {base_question}")
        if single:
            disease = base_question.split(' ')[-1]
            if disease[-1] == '?':
                disease = disease[:-1]
            question = question.format(disease)
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str))).convert("RGB")
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        # print(f"Vqa prompt: {question}")
        src_item = self.encode_text(' {}'.format(question))

        ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref.split('&&')}
        answer = max(ref_dict, key=ref_dict.get)
        conf = torch.tensor([ref_dict[answer]])
        tgt_item = self.encode_text(" {}".format(answer))

        ## INACTIVATE
        predict_objects = None
        self.add_object = False
        if self.add_object and predict_objects is not None:
            predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
            predict_object_item = self.encode_text(" object: {}".format(predict_object_seq))
            src_item = torch.cat([src_item, predict_object_item])

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        
        ## INACTIVATE
        self.prompt_type = 'none'
        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1]
        else:
            raise NotImplementedError
        target_item[:-len(tgt_item)-1] = self.tgt_dict.pad()

        code_mask = torch.tensor([False])
        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
            "conf": conf,
            "code_mask": code_mask,
        }
        
        ## INACTIVATE
        self.constraint_trie = None
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item)-len(tgt_item)-1, len(target_item)):
                constraint_prefix_token = [self.tgt_dict.bos()] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
            
        # print(f"VQA: {single}")
        # print(example)
        return [example]
    
    def process_vg(self, index, the_dataset, single=True, with_location=False):
        """
        @ with_location: add location prompt at instruction, e.g., where is xxx in the top left?
        """
        vg_multi_prompt_pool = [
            ' Where is {} {}? please answer with coordinates',
            ' What are the coordinates of {} located {}?',
            ' Can you give me the coordinates of {} {}',
            ' Please provide the coordinates for {} {} region of the image',
            ' Give the accurate bbox of {}.',   # OmniFM-DR paper insturction 
        ][-1:]

        ### Dataset:
        # 0                                              194_111
        # 1                                                 7181
        # 2                                               Nodule
        # 3                                    515,1168,692,1330
        # 4                                      Vindr/images...
        # 1, 3, 2, 0, 4
        uniq_id, coors, labels, locations, image_str = the_dataset[index]
        coors_pool = coors.split('&&')
        labels_pool = labels.split(',')
        labels_pool = [l.lower() for l in labels_pool]

        if with_location:
            locations_pool = locations.split('&&')
            assert (len(coors_pool) == len(labels_pool) == len(locations_pool))

        random_ind = random.randint(0, len(coors_pool) - 1)
        region_coord = coors_pool[random_ind]
        label = labels_pool[random_ind]
        
        if with_location:
            location = locations_pool[random_ind]
        
        image = Image.open(image_str).convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        coors = region_coord.strip().split(',')
        coors = np.array(coors, dtype=np.float32).reshape(-1, 4)
        ## box order: random (pix2pix: random; ofa: by size)
        num_boxes = coors.shape[0]
        areas = (coors[:,2] - coors[:,0]) * (coors[:,3] - coors[:,1])
        
        boxes_target["boxes"] = torch.tensor(coors)
        boxes_target["labels"] = np.array([0]*num_boxes)
        boxes_target["area"] = torch.tensor(areas)

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        
        region_coord = ""
        for i in range(num_boxes):
            quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][i][0] * (self.num_bins - 1)).round()))
            quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][i][1] * (self.num_bins - 1)).round()))
            quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][i][2] * (self.num_bins - 1)).round()))
            quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][i][3] * (self.num_bins - 1)).round()))
            region_coord += "{} {} {} {} ".format(quant_x0, quant_y0, quant_x1, quant_y1)
        region_coord = region_coord[:-1]
        src_item_pool = []
        for i, text in enumerate(vg_multi_prompt_pool):
            if with_location:
                text = text.format(label, location).lower()
            else:
                text = text.format(label).lower()
            src_caption = self.pre_caption(text, self.max_src_length)

            if self.debug and i == 0:
                print(f"[Visual Grounding]")
                print(f"Instruction: {src_caption}")
                print(f"Answer: {region_coord}")

            src_item = self.encode_text(src_caption)
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            src_item_pool.append(src_item)

        tgt_item = self.encode_text(region_coord, use_bpe=False)
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        conf = torch.tensor([2.0])
        code_mask = torch.tensor([False])
        
        ## add classification label
        multi_hot_labels = self._get_multi_hot_label(labels, split_tag=',')
        
        examples = []
        for ind, src_item in enumerate(src_item_pool):
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "w_resize_ratio": resize_w / w,
                "h_resize_ratio": resize_h / h,
                "region_coord": None, # region
                "conf": conf,
                "code_mask": code_mask,
                "multi_hot_label": multi_hot_labels,
            }
            examples.append(example)

        ## get vqa example
        vqa_example = []
        vqa_example_yesno_location = []
        if with_location:
            vqa_example = self.vg_append_vqa((uniq_id, patch_image, labels_pool, locations_pool, random_ind, resize_w / w, resize_h / h, multi_hot_labels))
            vqa_example_yesno_location = self.vg_append_vqa_yesno_location((uniq_id, patch_image, labels_pool, locations_pool, random_ind, resize_w / w, resize_h / h, multi_hot_labels))
        vqa_example_yesno = self.vg_append_vqa_yesno((uniq_id, patch_image, labels_pool, resize_w / w, resize_h / h, multi_hot_labels))
        vqas = vqa_example + vqa_example_yesno + vqa_example_yesno_location
        examples = self._random_choose(examples)
        vqas = self._random_choose(vqas)
        return examples, vqas

    def vg_append_vqa(self, infos):
        question_pool = [
            'Can you describe the relative location of {} in the chest image?',
            'Where is the relative location of {} in this chest image?',
            'In this chest image, where is the relative position of {}?'
        ][:1]
        # id, image, caption, tags
        uniq_id, patch_image, labels_pool, locations_pool, ind, w_ratio, h_ratio, multi_hot_labels = infos
        label = labels_pool[ind]
        location = locations_pool[ind]
        
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])

        src_item_pool = []
        for base_question in question_pool:
            question = self.pre_caption(base_question.format(label), self.max_src_length)
            # print(f"VG vqa: {question, location}")
            src_item = self.encode_text(" {}".format(question))
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            src_item_pool.append(src_item)
        
        tgt_item = self.encode_text(" {}".format(location))
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])
        conf = torch.tensor([2.0])
        code_mask = torch.tensor([False])
        examples = []
        for src_item in src_item_pool:
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "w_resize_ratio": w_ratio,
                "h_resize_ratio": h_ratio,
                "region_coord": None, # region
                "conf": conf,
                "code_mask": code_mask,
                "multi_hot_label": multi_hot_labels,
            }
            examples.append(example)
        return examples
    
    def vg_append_vqa_yesno(self, infos):
        question_pool_yesno = [
            'Does this DR image have {} ?',
            "is there {} in this DR image ?",
            "Is {} in this image?"
        ][-1:]
        
        uniq_id, patch_image, labels, w_ratio, h_ratio, multi_hot_labels = infos
        answer = labels
        negative_answer = []
        for l in self.diseases_vg:
            if l not in answer:
                negative_answer.append(l)
        positive = answer[random.randint(0, len(answer)-1)].lower()
        negative = negative_answer[random.randint(0, len(negative_answer) - 1)].lower()
      
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])
        examples = []
        for cur_example, ans in zip([positive, negative], ['yes, there is {}.', 'no {}.']):
            ## build source item pool
            src_item_pool = []
            for i, base_question in enumerate(question_pool_yesno):
                base_question = base_question.lower()
                base_question = base_question.format(cur_example).lower()
                # print(f"VG Yes/No quest: {base_question}")
                question = self.pre_caption(base_question, self.max_src_length)
                
                if self.debug and i == 0:
                    print("[Visual Grounding, VQA-yes/no]")
                    print(f"Instruction: {question}")

                src_item = self.encode_text(" {}".format(question))
                src_item = torch.cat([self.bos_item, src_item, self.eos_item])
                src_item_pool.append(src_item)

            if self.debug:
                print(f"Answer: {ans.format(cur_example)}")

            tgt_item = self.encode_text(" {}".format(ans.format(cur_example).lower()))
            target_item = torch.cat([tgt_item, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            conf = torch.tensor([2.0])
            code_mask = torch.tensor([False])
            for src_item in src_item_pool:
                example = {
                    "id": uniq_id,
                    "source": src_item,
                    "patch_image": patch_image,
                    "patch_mask": patch_mask,
                    "target": target_item,
                    "prev_output_tokens": prev_output_item,
                    "w_resize_ratio": w_ratio,
                    "h_resize_ratio": h_ratio,
                    "region_coord": None, # region
                    "conf": conf,
                    "code_mask": code_mask,
                    "multi_hot_label": multi_hot_labels,
                }
                examples.append(example)
        return examples
    
    def vg_append_vqa_yesno_location(self, infos):
        question_pool_yesno = [
            'Is there {} {}? ',
            "Does {} {} exist?",
            'Can {} {} be found?',
            "Is there any evidence of {} {}"
        ][:1]
        # id, image, caption, tags
        uniq_id, patch_image, labels, locations, ind, w_ratio, h_ratio, multi_hot_labels = infos
        label = labels[ind]
        location = locations[ind]
        positive_location = []
        for lab, loc in zip(labels,  locations):
            if lab == label:
                positive_location.append(loc)
        negative_location = []
        for loc in self.locations:
            if loc not in positive_location:
                negative_location.append(loc)
        
        answer = labels
        negative_answer = []
        for l in self.diseases_vg:
            if l not in answer:
                negative_answer.append(l)
        positive = location
        if len(negative_location):
            negative = negative_location[random.randint(0, len(negative_location) - 1)]
        else:
            negative = None
        # print(f"***VQA: {answer}")
      
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        # patch_image = image
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])

        examples = []
        for cur_example, ans in zip([(label, positive), (label, negative)], ['yes, there is {} {}.', 'no, there is no {} {}.']):
            if cur_example[1] is None:
                continue
            ## build source item pool
            src_item_pool = []
            for base_question in question_pool_yesno:
                base_question = base_question.format(*cur_example).lower()
                # print(f"VG Yes/No quest with location: {base_question}")
                question = self.pre_question(base_question, self.max_src_length)
                src_item = self.encode_text(" {}".format(question))
                src_item = torch.cat([self.bos_item, src_item, self.eos_item])
                src_item_pool.append(src_item)
            # print(f"VG yesno with location ans: {ans.format(*cur_example)}")
            tgt_item = self.encode_text(" {}".format(ans.format(*cur_example).lower()))
            target_item = torch.cat([tgt_item, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            conf = torch.tensor([2.0])
            code_mask = torch.tensor([False])
            for src_item in src_item_pool:
                example = {
                    "id": uniq_id,
                    "source": src_item,
                    "patch_image": patch_image,
                    "patch_mask": patch_mask,
                    "target": target_item,
                    "prev_output_tokens": prev_output_item,
                    "w_resize_ratio": w_ratio,
                    "h_resize_ratio": h_ratio,
                    "region_coord": None, # region
                    "conf": conf,
                    "code_mask": code_mask,
                    "multi_hot_label": multi_hot_labels,
                }
                examples.append(example)
        return examples
    
    def process_seg(self, index, fix_coor_nums=None):
        """
        @ fix_coor_nums: whether to use fixed number of points during training
        """
        seg_prompt_pool = [
            ' what region of the chest represents the contour of the {}?',
            ' Which area of the chest outlines the boundaries of the {}?',
            ' Can you identify the chest area that reflects the shape of the {} contour?',
            ' Please show the {} region.',
            ' Please segment the {} from the given image.' # OmniFM-DR format
        ][-1:]

        ### Dataset
        # 0                                                    0
        # 1                         left lung&&right lung&&heart
        # 2    349,51,371,56,393,68,414,85,430,106,441,126,44...
        # 3                                  dr_preprocessed_...
        # 4                                             18028180
        # 5                                             54705304
        
        # process index: 0, 3, 1, 2
        uniq_id, image_str, text, region_coord = self.seg_dataset[index]

        text_list = text.split('&&')
        coord_list = region_coord.split('&&')
        random_ind = random.randint(0, len(text_list)-1)
        
        ### For chestXmask data, we just use cardiac data
        random_ind = -1

        label = text_list[random_ind]
        if '&&' not in text:
            label = label + ' lung'

        region_coord = coord_list[random_ind]
        text = seg_prompt_pool[random.randint(0, len(seg_prompt_pool)-1)].format(label)
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str))).convert("RGB")
        image = Image.open(image_str).convert("RGB")

        w, h = image.size
        boxes_target = {"polygons": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        region_coord = region_coord.replace(';', ',')
        
        nodes = region_coord.strip().split(',')
        region_list = [float(x.strip()) for x in nodes]

        if fix_coor_nums is not None and isinstance(fix_coor_nums, int):
            if len(region_list) < fix_coor_nums * 2:
                n = (fix_coor_nums * 2) // len(region_list)
                for _ in range(n+1):
                    region_list += region_list
            region_list = region_list[:fix_coor_nums * 2]

        region = torch.tensor(region_list)
        boxes_target["polygons"] = torch.tensor([region_list])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = self.polygon_area(region_list[::2], region_list[1::2])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        
        region_coord = ""
        for i in range(len(region_list)):
            quant_coor = "<bin_{}>".format(int((patch_boxes["polygons"][0][i] * (self.num_bins - 1)).round()))
            region_coord += quant_coor
            if i != len(region_list) - 1:
                region_coord += " "
        
        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.encode_text(src_caption)
        tgt_item = self.encode_text(region_coord, use_bpe=False)

        if self.debug:
            print("[Segmentation]")
            print(f"Instruction: {src_caption}")
            print(f"Answer: {region_coord}")

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        conf = torch.tensor([2.0])
        code_mask = torch.tensor([False])
        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region,
            "conf": conf,
            "code_mask": code_mask,
        }
        return [example]
    
    def polygon_area(self, x, y):
        """
        计算多边形的面积
        x: list, 多边形各个顶点的 x 坐标
        y: list, 多边形各个顶点的 y 坐标
        """
        n = len(x)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= y[i] * x[j]
        area = abs(area) / 2.0
        return area

    def process_caption(self, index):
        report_generation_prompt_pool = [
            ' Please generate a report for this DR image.',
            ' Please describe this DR chest image in details.',
            ' what can we get from this chest medical image?',
            ' What information can be obtained from this DR chest image?',
            ("You are a helpful radiology assistant. Describe what lines, "
                "tubes, and devices are present and each of their locations. "
                "Describe if pneumothorax is present; if present, describe size "
                "on each side. Describe if pleural eŦusion is present; if present, "
                "describe amount on each side. Describe if lung opacity (atelectasis, "
                "fibrosis, consolidation, infiltrate, lung mass, pneumonia, pulmonary "
                "edema) is present; if present, describe kinds and locations. Describe "
                "the cardiac silhouette size. Describe the width and contours of the "
                "mediastinum. Describe if hilar enlargement is present; if enlarged, "
                "describe side. Describe what fractures or other skeletal abnormalities are present."), # Med-Palm format
            ' describe the image',  # OmniFM-DR paper insturction 
        ][-1:]

        ### Dataset
        # 0                                                    0
        # 1                                                    0
        # 2    the cardiac, mediastinal and hilar contours ar...
        # 3                                           No Finding
        # 4                                  dr_preprocessed_...
        # 5                                             10000032
        # 6                                             53189527
        # 7                                                    0
        ## positive : negative = 9:1
        random_choose = random.randint(0, 9)
        split_ind = 8 if self.negative_dataset is not None else 10
        if self.split != 'train':
            split_ind = 10
        if random_choose <= split_ind:
            # 0, 4, 2, 3, 5, 6
            uniq_id, image_str, caption, tags, info1, info2 = self.dataset[index]
        else:
            uniq_id_n, image_str, caption, tags, info1, info2 = self.negative_dataset[index]
            uniq_id = str(int(uniq_id_n) + 200000)

        images = image_str.split(',')
        random_image_ind = random.randint(0, len(images)-1)
        image = images[random_image_ind]
        if image_str == '/mnt/lustre/niziyu/data/MIMIC/dr_preprocessed_jpgs/17486231_53979270.jpg':
            return self.__getitem__(index)
        image = Image.open(image) 
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if self.split == 'train':
            # caption = caption.translate(self.transtab).strip()
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        
        ## build source item pool
        src_item_pool = []
        for i, prompt in enumerate(report_generation_prompt_pool):
            prompt = prompt.lower()
            
            if self.debug:
                print("[Report Generation]")
                print(f"Instruction: {prompt}")

            src_item = self.encode_text(prompt)
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            src_item_pool.append(src_item)
        
        if self.debug:
            print(f"Answer: {tgt_caption}")

        tgt_item = self.encode_text(" {}".format(tgt_caption), use_mimic=True)

        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        conf = torch.tensor([2.0])
        code_mask = torch.tensor([False])
        examples = []
        
        ## add classification label
        multi_hot_labels = self._get_multi_hot_label(tags)
        
        for ind, src_item in enumerate(src_item_pool):
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "code_mask": code_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "conf": conf,
                "multi_hot_label": multi_hot_labels
            }
            examples.append(example)
        
        ## add other instructions for the same image
        vqa_example = self.caption_append_vqa((uniq_id, patch_image, caption, tags, multi_hot_labels))
        vqa_yesno_example = self.caption_append_vqa_yesno((uniq_id, patch_image, caption, tags, multi_hot_labels))
        vqas = vqa_example + vqa_yesno_example
        examples = self._random_choose(examples)
        vqas = self._random_choose(vqas)
        return examples, vqas

    def caption_append_vqa(self, infos):
        disease_list = ', '.join(self.diseases_mimic)
        question_pool_multi_diseases = [
            'From the DR chest image, what illness can be inferred?',
            " what disease does the DR chest image have?",
            "Can you identify the disease in the DR image?",
            "Please tell me the disease in this DR image",
            f' given these following diseases: {disease_list}, which diseases may be diagnoised in the given DR image?', # Med Palm-M format
            'What disease does this image have?',   # OmniFM-DR format
        ][-1:]

        uniq_id, patch_image, _, tags, multi_hot_labels = infos
        answer = tags.replace("&&", ", ").lower()
      
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])

        ## build source item pool
        src_item_pool = []
        for i, base_question in enumerate(question_pool_multi_diseases):
            base_question = base_question.lower()
            question = self.pre_question(base_question, self.max_src_length)

            if self.debug and i == 0:
                print(f"[Report Generation, VQA-all]")
                print(f"Instruction: {question}")

            src_item = self.encode_text(" {}".format(question))
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            src_item_pool.append(src_item)
        
        if self.debug:
            print(f"Answer: {answer}")

        tgt_item = self.encode_text(" {}".format(answer))
        pos_src_item = self.encode_text(' what is the answer to question " {} ". is " {} "?'.format(question, answer))

        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])
        pos_src_item = torch.cat([self.bos_item, pos_src_item, self.eos_item]) if type != 'visual_grounding' else None
        conf = torch.tensor([2.0])
        code_mask = torch.tensor([False])
        examples = []
        for src_item in src_item_pool:
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "code_mask": code_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "conf": conf,
                "multi_hot_label": multi_hot_labels
            }
            examples.append(example)
        return examples

    def caption_append_vqa_yesno(self, infos):
        question_pool_yesno = [
            'Does this DR image have {} ?',
            "is there {} in this DR image ?",
            'Is {} in this image?',
        ][-1:]

        uniq_id, patch_image, _, tags, multi_hot_labels = infos
        answer = tags.split('&&')
        answer = [a.lower() for a in answer]
        negative_answer = []
        for l in self.mimic_long_tail:
            if l not in answer:
                negative_answer.append(l)
        positive = answer[random.randint(0, len(answer) - 1)]
        negative = negative_answer[random.randint(0, len(negative_answer) - 1)]
      
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])

        examples = []
        for cur_example, ans in zip([positive, negative], ['yes, there is {}.', 'no {}.']):
            ## build source item pool
            src_item_pool = []
            for i, base_question in enumerate(question_pool_yesno):
                base_question = base_question.lower()
                base_question = base_question.format(cur_example)
                question = self.pre_question(base_question, self.max_src_length)

                if self.debug and i == 0:
                    print("[Report Generation, VQA-yes/no]")
                    print(f"Instruction: {question}")

                src_item = self.encode_text(" {}".format(question))
                src_item = torch.cat([self.bos_item, src_item, self.eos_item])
                src_item_pool.append(src_item)
            
            if self.debug:
                print(f"Answer: {ans.format(cur_example)}")

            tgt_item = self.encode_text(" {}".format(ans.format(cur_example)))
            target_item = torch.cat([tgt_item, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            conf = torch.tensor([2.0])
            code_mask = torch.tensor([False])
            for src_item in src_item_pool:
                example = {
                    "id": uniq_id,
                    "source": src_item,
                    "patch_image": patch_image,  # use caption-task patch_image in forward
                    "patch_mask": patch_mask,
                    "code_mask": code_mask,
                    "target": target_item,
                    "prev_output_tokens": prev_output_item,
                    "conf": conf,
                    "multi_hot_label": multi_hot_labels
                }
                examples.append(example)
        return examples
    
    def vqa_attribute(self, index):
        """
        Attribute VQA. 
        NOTE: This interface is redundant. Please avoid designing it this way in future development.
        """
        instruction_pool = [
            'What is the level of {}?', # severity
            'Where is {}?',     # location
        ]

        ### Dataset
        # /mnt/lustre/niziyu/projects/OFA/dataset/report_generation/mimic/all_11_0910/train_xxx_balanced_filtered_50_addFullAdj.tsv
        # 0                                                94473
        # 1                                                94473
        # 2    an endotracheal tube terminates . cm above the...
        # 3                        Atelectasis&&Pleural Effusion
        # 4    /mnt/lustre/niziyu/data/MIMIC/dr_preprocessed_...
        # 5                                                  NaN
        # 6                                                  NaN
        # 7    atelectasis,right&small&base,effusion,right&sm...
        random_choose = random.randint(0, 9)
        split_ind = 8 if self.vqa_attritube_dataset_negative is not None else 10
        if self.split != 'train':
            split_ind = 10
        if random_choose <= split_ind:
            # process index: 0, 4, 7
            uniq_id, image_str, label_info = self.vqa_attritube_dataset_positive[index]
        else:
            uniq_id, image_str, label_info = self.vqa_attritube_dataset_negative[index]
            uniq_id = str(int(uniq_id) + 200000)
        
        # get label and attributes
        location_tuple, severity_tuple = self._process_adjs(label_info)
        if len(location_tuple) and not len(severity_tuple):
            info = location_tuple
            instruction_pool = instruction_pool[-1:]
        elif not len(location_tuple) and len(severity_tuple):
            info = severity_tuple
            instruction_pool = instruction_pool[:1]
        elif len(location_tuple) and len(severity_tuple):
            if random.randint(0, 1):
                info = location_tuple
                instruction_pool = instruction_pool[-1:]
            else:
                info = severity_tuple
                instruction_pool = instruction_pool[:1]
        else:
            return self.vqa_attribute(index + 1)
        label, attributes = info[random.randint(0, len(info)-1)]
        caption = attributes[0]
        
        images = image_str.split(',')
        random_image_ind = random.randint(0, len(images)-1)
        image = images[random_image_ind]
        if image_str == '/mnt/lustre/niziyu/data/MIMIC/dr_preprocessed_jpgs/17486231_53979270.jpg':
            return self.__getitem__(index)
        image = Image.open(image) 
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if self.split == 'train':
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        
        ## build source item pool
        src_item_pool = []
        for i, prompt in enumerate(instruction_pool):
            prompt = prompt.format(label).lower()
            
            if self.debug:
                print("[VQA]")
                print(f"Instruction: {prompt}")

            src_item = self.encode_text(prompt)
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            src_item_pool.append(src_item)
        
        if self.debug:
            print(f"Answer: {tgt_caption}")

        tgt_item = self.encode_text(" {}".format(tgt_caption), use_mimic=True)

        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        conf = torch.tensor([2.0])
        code_mask = torch.tensor([False])
        examples = []
        
        for ind, src_item in enumerate(src_item_pool):
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "code_mask": code_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "conf": conf,
            }
            examples.append(example)
        return examples

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch):
            report_example, vqa_example = self.process_caption(index)
            pair_samples = report_example
            extra_samples = []
            if self.split == 'train':
                vg, vg_vqas = self.process_vg(0, self.visual_ground_dataset) if self.visual_ground_dataset else []  # DONE
                seg_list = self.process_seg(0) if self.seg_dataset else []
                vqa_attribute = self.vqa_attribute(0) if self.vqa_attritube_dataset_positive else []

                ## random choose
                random_flag = random.randint(1, 256)
                if self.debug:
                    print(f"Random flag: {random_flag}")

                if random_flag <= 116:
                    pair_samples = report_example
                    if self.debug:
                        print("[choose report generation]")
                elif random_flag > 116 and random_flag <= 187:
                    pair_samples = seg_list
                    if self.debug:
                        print("[choose segmentation]")
                elif random_flag > 187 and random_flag <= 215:
                    pair_samples = vg
                    if self.debug:
                        print("[choose visual grounding]")
                elif random_flag > 215 and random_flag <= 230:
                    pair_samples = vqa_example
                    if self.debug:
                        print("[choose vqa]")
                elif random_flag > 230 and random_flag <= 245:
                    pair_samples = vqa_attribute
                    if self.debug:
                        print("[choose vqa attribute]")
                elif random_flag > 245 and random_flag <= 256:
                    pair_samples = vg_vqas
                    if self.debug:
                        print("[choose vqa visual grounding]")
        return pair_samples, extra_samples

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=4, high=len(self.tgt_dict)-self.code_dict_size-self.num_bins, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []   # containing image-text pairs
        samples_v2 = []   # containing detection data, text data and image data
        for sample_tuple in samples:
            samples_v1 += sample_tuple[0]
            samples_v2 += sample_tuple[1]
        if samples_v2 != []:
            res_v1 = collate(samples_v1, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
            res_v2 = collate(samples_v2, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
            return res_v1, res_v2
        else:
            res_v1 = collate(samples_v1, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
            return res_v1

    def _random_choose(self, canditates):
        if isinstance(canditates, list):
            num = len(canditates)
            if num:
                random_ind = random.randint(0, num-1)
                return [canditates[random_ind]]
        return canditates

    def _get_multi_hot_label(self, labels, split_tag='&&'):
        # print(f"[---------] data prepare: labels: {labels}")
        disease_list = self.mimic_long_tail
        labels = labels.split(split_tag)
        labels = [l.strip() for l in labels]
        multi_hot = torch.zeros(len(disease_list), dtype=torch.int16)
        for l in labels:
            if l in disease_list:
                ind = disease_list.index(l)
                multi_hot[ind] = 1
        multi_hot = multi_hot.unsqueeze(0)
        # print(f"[---------] data prepare: multi-hot: {multi_hot}")
        return multi_hot

    def _process_adjs(self, adjs):
        locations = ['right', 'left', 'mid', 'base', 'lower lobe', 'bibasilar', 'basilar']
        locations = {l:1 for l in locations}
        adj_list = adjs.split(',')
        location_tuple = []
        severity_tuple = []
        for i in range(0, len(adj_list), 2):
            if i + 1 >= len(adj_list):
                continue
            k = adj_list[i].strip().lower()
            vs = adj_list[i+1].split('&')
            vs = sorted([v.strip() for v in vs if v.strip() in locations], reverse=True)
            if len(vs):
                location_tuple.append([k, vs])
            vs = sorted([v.strip() for v in vs if v.strip() not in locations], reverse=True)
            if len(vs):
                severity_tuple.append([k, vs])
        return location_tuple, severity_tuple
