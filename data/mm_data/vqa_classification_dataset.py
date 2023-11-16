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

    multi_hot_label = torch.cat([sample['multi_hot_label'] for sample in samples])

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
        "multi_hot_label": multi_hot_label,
    }

    return batch


class VqaClassificationDataset(OFADataset):
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
        # """
        # data augmentation
        # self.patch_resize_transform = transforms.Compose([
        #     lambda image: image.convert("RGB"),
        #     T.RandomResize(scales, max_size=384),
        #     transforms.CenterCrop(patch_image_size),
        #     RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness',
        #                                           'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std),
        # ])

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.RandomCrop(int(patch_image_size//8*7)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what disease does this chest image have? "
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片描述了什么内容?"
        self.negative_dataset = negative_dataset
        self.chestxray14_labels = ['Consolidation', 'Cardiomegaly', 'Mass', 'Edema', 'Nodule', 'Pneumonia', 'Atelectasis', 'Hernia', 'Effusion', 'Fibrosis', 'Infiltration', 'Pneumothorax', 'Pleural Thickening', 'Emphysema']
        self.chestxray14_labels = [l.lower() for l in self.chestxray14_labels]

        self.rsna_labels = ['Penumonia', 'No Finding']  # wrong spell
        self.rsna_labels = [l.lower() for l in self.rsna_labels]

        self.diseases_mimic = ['Atelectasis', 'Lung Lesion', 'Pneumonia', 'Fracture', 'Cardiomegaly', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Consolidation', 'Lung Opacity', 'Edema', 'No Finding']
        self.diseases_mimic = [l.lower() for l in self.diseases_mimic]

        # Cardiomegaly&Atelectasis&Edema&Effusion&Consolidation&Pneumonia&Pneumothorax
        # intersection of MIMIC and chestXray14
        self.mmic_chestxray14 = [l for l in self.diseases_mimic if l in self.chestxray14_labels] + ['Pleural Effusion']
        self.mmic_chestxray14 = [l.lower() for l in self.mmic_chestxray14]

        # Codalab mimic 26 labels
        self.codalab_mimic_26 = ['Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass', 'No Finding', 'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta']
        self.codalab_mimic_26 = [l.lower() for l in self.codalab_mimic_26]

    def __getitem__(self, index):
        # """
        ## random choose from positive or negative dataset (4:1)
        high = 5
        # random_choose = np.random.randint(low=0, high=high)
        random_choose = random.randint(0, high)
        split_ind = 2 if self.negative_dataset is not None else high
        if self.split != 'train':
            split_ind = high
        if random_choose <= split_ind:
            uniq_id, image, labels = self.dataset[index]
            # print(f"Use positive data: {uniq_id, image, labels, cardic}")
        else:
            uniq_id, image, labels = self.negative_dataset[index]
            # print(f"Use negative data: {uniq_id, image, labels, cardic}")

        label_list = labels.split('&&')
        label_list = [l.lower().strip() for l in label_list]
        label_list = [l for l in label_list if len(l)]

        # # use corrected Cardiomegaly label (proved to be useful)
        # if int(cardic) != 1:
        #     label_list = [l for l in label_list if l != 'cardiomegaly']
        # elif int(cardic) == 1:
        #     if 'cardiomegaly' not in label_list:
        #         label_list.append('cardiomegaly')

        # # use corrected 7 labels (intersection between MIMIC and Chestxray)
        # label_list = [l for l in label_list if l not in self.mmic_chestxray14]
        # label_list_update = cardic.split('&')
        # label_list_update = [l.lower().strip() for l in label_list_update if l.lower().strip() in self.mmic_chestxray14]
        # label_list = label_list + label_list_update

        multi_hot_labels = self._get_multi_hot_label(label_list)
        caption = ", ".join(label_list)
        # print(f"**** : {caption}")

        images = image.split(',')
        # random_image_ind = np.random.randint(low=0, high=len(images))
        random_image_ind = random.randint(0, len(images)-1)
        image = images[random_image_ind]
        if image == '/mnt/lustre/niziyu/data/MIMIC/dr_preprocessed_jpgs/17486231_53979270.jpg':
            return self.__getitem__(index)
        image = Image.open(image)
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if self.split == 'train' and not self.scst:
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
            "prev_output_tokens": prev_output_item,
            "multi_hot_label": multi_hot_labels
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

    def _get_multi_hot_label(self, labels):
        ## TODO: add target label to args
        # target_label = self.codalab_mimic_26
        target_label = self.chestxray14_labels
        # target_label = self.rsna_labels

        labels = [l.strip() for l in labels]
        multi_hot = torch.zeros(len(target_label), dtype=torch.int16)
        for l in labels:
            if l in target_label:
                ind = target_label.index(l)
                multi_hot[ind] = 1
        multi_hot = multi_hot.unsqueeze(0)
        return multi_hot
