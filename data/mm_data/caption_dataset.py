# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import os
from io import BytesIO

import logging
import warnings
import string

import numpy as np
import torch
import base64
import random
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

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }

    return batch


class CaptionDataset(OFADataset):
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
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False,
        negative_dataset=None
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict, mimic_t2i=mimic_t2i, mimic_i2t=mimic_i2t)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        
        
        scales = np.arange(patch_image_size, patch_image_size+10).tolist()
        """
        # data augmentation
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            T.RandomResize(scales, max_size=512),
            transforms.CenterCrop(patch_image_size),
            RandomAugment(2, 2, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        """
        # self.patch_resize_transform = transforms.Compose([
        #     lambda image: image.convert("RGB"),
        #     transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std),
        # ])
        # """

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.RandomCrop(int(patch_image_size//8*7)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        

        if type(bpe).__name__ == 'GPT2BPE':
            # self.prompt = " what does the image describe?"
            # self.prompt = " what report can a doctor give for this 2D medical image? "
            # self.prompt = " what can we get from this chest medical image? "
            self.prompt = " what disease does this chest image have? "
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片描述了什么内容?"
        # print(f"TTTTTTTTTTTTTTTTTTTTTTTTTTTT: ")
        # print(self.pad, self.eos)
        # exit()
        self.negative_dataset = negative_dataset
        self.diseases_mimic = ['Atelectasis', 'Lung Lesion', 'Pneumonia', 'Fracture', 'Cardiomegaly', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Consolidation', 'Lung Opacity', 'Edema']
        
    def __getitem__(self, index):
        # """
        ## random choose from positive or negative dataset (9:1)
        high = 5
        # random_choose = np.random.randint(low=0, high=high)
        random_choose = random.randint(0, high)
        # print(f"self.negative_dataset: {self.negative_dataset}")
        split_ind = 2 if self.negative_dataset is not None else high
        split_ind = high
        # print(f"Random choose: {random_choose, split_ind}")
        if self.split != 'train':
            split_ind = high
        if random_choose <= split_ind:
            uniq_id, image, caption, labels, adjs = self.dataset[index]
            # uniq_id, image, caption, labels, info1, info2 = self.dataset[index]
            # print(f"Use positive data: {uniq_id, image, caption}")
        else:
            uniq_id, image, caption, labels, adjs = self.negative_dataset[index]
            # uniq_id, image, caption, labels, info1, info2 = self.negative_dataset[index]
            # print("Use negative data: {uniq_id, image, captioin}")

        """
        unique_id, image, caption = self.dataset[index]
        # print(unique_id, image, caption)
        """

        caption = caption.lower()
        caption = caption.split('&&')
        caption = [c.strip() for c in caption if len(c.strip())]
        # random.shuffle(caption)
        caption = sorted(caption)
        caption = ", ".join(caption)
        # print(caption)
       
        images = image.split(',')
        random_image_ind = np.random.randint(low=0, high=len(images))
        # print(f"Random Image: {random_image_ind}")
        image = images[random_image_ind]
        

        ## add label info in prompt
        # """
        label_list = labels.split('&&')
        label_list = [l.lower() for l in label_list]
        
        """
        ### add extra info based on classifications
        if 'No Finding'.lower() in label_list:
            extra_info = ' it seems that there are no diseases in this image.'
        else:
            # extra_info = ' there can be {} diseases in this image.'.format(', '.join(label_list))
            extra_info = ' what disease does this image have? there are {} in this image. '.format(', '.join(label_list))
            for d in self.diseases_mimic:
                if d.lower() in label_list:
                    vqa_info = ' does this image have {}? yes, there is {}. '.format(d.lower(), d.lower())
                else:
                    vqa_info = ' does this image have {}? no, no {} find in this image. '.format(d.lower(), d.lower())
                extra_info += vqa_info 
        # extra_info = ' test '
        self.prompt = " based on the following info: ' {} ', what can we get from this chest medical image? ".format(extra_info)
        # exit()
        """   

        ## Cardiomegaly prompt  
        # self.prompt = " based on the following cardiomegaly info: ' {} ', what can we get from this chest medical image? ".format(card_prompt)  
        # self.prompt = " what can we get from this chest medical image? " 
        # self.prompt = " based on the following pneumothorax info: ' {} ', what can we get from this chest medical image? ".format(card_prompt) 
        
        # self.prompt = ' given the following info: {}, please generate an report for this DR image. '.format(', '.join([info1, info2]))

        # ### 1. just 'what disease does this image have'
        # if 'No Finding'.lower() in label_list:
        #     extra_info = ' it seems that there are no diseases in this image.'
        # else:
        #     extra_info = ' what disease does this image have? there are {} in this image. '.format(', '.join(label_list))
        # self.prompt = " based on the following info: ' {} ', what can we get from this chest medical image? ".format(extra_info)

        # ### 2. 'what disease does this image have?' + 'does this image have xxx ?'
        # if 'No Finding'.lower() in label_list:
        #     extra_info = ' it seems that there are no diseases in this image.'
        # else:
        #     # extra_info = ' there can be {} diseases in this image.'.format(', '.join(label_list))
        #     extra_info = ' what disease does this image have? there are {} in this image. '.format(', '.join(label_list))
        #     for d in self.diseases_mimic:
        #         if d.lower() in label_list:
        #             vqa_info = ' does this image have {}? yes, there is {}. '.format(d.lower(), d.lower())
        #         else:
        #             vqa_info = ' does this image have {}? no, no {} find in this image. '.format(d.lower(), d.lower())
        #         extra_info += vqa_info 
        # self.prompt = " based on the following info: ' {} ', what can we get from this chest medical image? ".format(extra_info)

        # ### 3. based on 2, add all prompt
        # prompt_list = prompt.split(',')
        # prompt_dict = {}
        # for i in range(0, len(prompt_list), 2):
        #     if i + 1 >= len(prompt_list):
        #         break
        #     k = prompt_list[i].lower()
        #     v = prompt_list[i+1]
        #     v = ' '.join(v.split('&')).lower()
        #     prompt_dict[k] = v
        # if 'No Finding'.lower() in label_list:
        #     extra_info = ' it seems that there are no diseases in this image.'
        # else:
        #     # extra_info = ' there can be {} diseases in this image.'.format(', '.join(label_list))
        #     extra_info = ' what disease does this image have? there are {} in this image. '.format(', '.join(label_list))
        #     for d in self.diseases_mimic:
        #         cur_disease = d.lower()
        #         if cur_disease in label_list:
        #             adjs = prompt_dict.get(cur_disease, '')
        #             vqa_info = ' does this image have {}? yes, there is {} {}. '.format(cur_disease, adjs, cur_disease)
        #         else:
        #             if cur_disease == 'cardiomegaly':
        #                 vqa_info = ' does this image have {}? no, find normal cardiomegaly in this image. '.format(cur_disease)
        #             else:
        #                 vqa_info = ' does this image have {}? no, no {} find in this image. '.format(cur_disease, cur_disease)
        #         extra_info += vqa_info 
        # self.prompt = " based on the following info: ' {} ', what can we get from this chest medical image? ".format(extra_info)

        # ### 4. just 'what disease does this image have' + descriptions
        # if 'No Finding'.lower() in label_list:
        #     extra_info = ' it seems that there are no diseases in this image.'
        # else:
        #     label_with_description = []
        #     for l in label_list:
        #         adjs = prompt_dict.get(l, '')
        #         label_with_description.append(adjs + ' ' + l)
        #     extra_info = ' what disease does this image have? there are {} in this image. '.format(', '.join(label_with_description))
        # self.prompt = " based on the following info: ' {} ', what can we get from this chest medical image? ".format(extra_info)

        # ### For disease level test
        # prompt_list = prompt.split(',')
        # prompt_dict = {}
        # for i in range(0, len(prompt_list), 2):
        #     if i + 1 >= len(prompt_list):
        #         break
        #     k = prompt_list[i].lower()
        #     v = prompt_list[i+1]
        #     v = ' '.join(v.split('&&')).lower()
        #     prompt_dict[k] = v
        # nums = len(prompt_dict)
        # rand_ind = random.randint(0, nums-1)
        # test_k = list(prompt_dict.keys())[rand_ind]
        # test_v = prompt_dict[test_k]
        # self.prompt = " what is the level of {} ?".format(test_k)
        # caption = test_v

        # print(f'************** {self.prompt}')
        # print(f"++++++++++++++++++++ {caption}")

        # self.prompt = " what can we get from this chest medical image? "

        # ### [Classsification]-[location]
        # adj_process = self._process_adjs(adjs)
        # if adj_process is None:
        #     return self.__getitem__(index+1)
        # k, v = adj_process
        # caption = v
        # self.prompt = " where is {} in this DR image ?".format(k)

        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        if image == '/mnt/lustre/niziyu/data/MIMIC/dr_preprocessed_jpgs/17486231_53979270.jpg':
            return self.__getitem__(index)
        
        image = Image.open(image)
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if self.split == 'train' and not self.scst:
            # caption = caption.translate(self.transtab).strip()
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        src_item = self.encode_text(self.prompt)
        tgt_item = self.encode_text(" {}".format(tgt_caption), use_mimic=True)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)

    def _process_adjs(self, adjs):
        locations = ['right', 'left', 'mid', 'base', 'lower lobe', 'bibasilar', 'basilar']
        locations = {l:1 for l in locations}
        adj_list = adjs.split(',')
        adj_tuple = []
        for i in range(0, len(adj_list), 2):
            if i + 1 >= len(adj_list):
                continue
            k = adj_list[i].strip().lower()
            vs = adj_list[i+1].split('&')
            vs = sorted([v.strip() for v in vs if v.strip() in locations], reverse=True)
            if not len(vs):
                continue
            adj_tuple.append((k, vs[0]))
        if len(adj_tuple) == 0:
            return None
        return adj_tuple[random.randint(0, len(adj_tuple)-1)]

