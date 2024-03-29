# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string

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


class ReportVqaDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
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

        scales = np.arange(patch_image_size, patch_image_size+20).tolist()
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            T.RandomResize(scales, max_size=672),
            transforms.CenterCrop(patch_image_size),
            RandomAugment(2, 2, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.test_cls = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what can we get from this chest medical image?"
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片描述了什么内容?"

    def process_qa(self, infos):
        base_question = " what disease does the image show?"
        # id, image, caption, tags
        uniq_id, image, _, tags = infos
        # ans = tags.split('&&')
        answer = tags.replace("&&", ", ")

        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        # patch_image = image
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])
        
        question = self.pre_question(base_question, self.max_src_length)
        # ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in refs.split('&&')}
        # answer = max(ref_dict, key=ref_dict.get)
        # conf = ref_dict[answer]
        src_item = self.encode_text(" {}".format(question))
        tgt_item = self.encode_text(" {}".format(answer))
        # conf = torch.tensor([conf])
        pos_src_item = self.encode_text(' what is the answer to question " {} ". is " {} "?'.format(question, answer))
        
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])
        pos_src_item = torch.cat([self.bos_item, pos_src_item, self.eos_item]) if type != 'visual_grounding' else None
        
        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
        }
        return [example]
        
    def __getitem__(self, index):
        uniq_id, image, caption, tags = self.dataset[index]

        if image == '/mnt/lustre/niziyu/data/MIMIC/dr_preprocessed_jpgs/17486231_53979270.jpg':
            return self.__getitem__(index)
        image = Image.open(image)
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
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
        tgt_item = self.encode_text(" {}".format(tgt_caption))

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
        report_example = [example]
        vqa_example = self.process_qa((uniq_id, patch_image, caption, tags))
        return report_example, vqa_example

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

