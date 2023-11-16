# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings

import random
import numpy as np
import torch
import base64
import copy
import utils.transforms as T

from PIL import Image, ImageFile
from utils.vision_helper import RandomAugment
from data import data_utils
from data.ofa_dataset import OFADataset
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation

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

    w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
    h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
    region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

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
        "w_resize_ratios": w_resize_ratios,
        "h_resize_ratios": h_resize_ratios,
        "region_coords": region_coords
    }

    return batch


class RefcocoDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=512,
        imagenet_default_mean_and_std=False,
        num_bins=1000,
        max_image_size=512
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        
        # aug
        self.augment_brightness = 0.002
        self.augment_contrast = 0.002
        self.augment_hue = 0.0002
        self.augment_saturation = 0.002

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        """
        scales = np.arange(patch_image_size, patch_image_size+10).tolist()
        # for image-text pair
        self.positioning_transform = T.Compose([
            T.RandomResize(scales, max_size=540),
            T.CenterCrop(patch_image_size),
            RandomAugment(2, 2, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        """
        # for positioning
        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])
        # """

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = ' which region does the text " {} " describe?'
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = '这段文字" {} "描述的是哪个区域？'

    def __getitem__(self, index):
        uniq_id, base64_str, text, region_coord = self.dataset[index]

        image = Image.open(base64_str).convert("RGB")
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        
        # random augmentation
        # image, x0, y0, x1, y1 = self.random_augmentation(image, np.array([[float(x0), float(y0), float(x1), float(y1)]]))
        
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        src_caption = self.pre_caption(text, self.max_src_length)
        text2 = src_caption.split(',')
        src_item = self.encode_text(f'What are the coordinates of {text2[0]} located {text2[1][1:-1]}?')  #add loc
        # src_item = self.encode_text(f'What are the coordinates of {text2[0]}?')    #no loc
        #src_item = self.encode_text(self.prompt.format(src_caption))  # ori
        tgt_item = self.encode_text(region_coord, use_bpe=False)
        #print("For checking...")
        #print(f'What are the coordinates of {text2[0]} located {text2[1][1:-1]}?')
        # print(tgt_item)
        
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
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region
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

    def random_augmentation(self, im, boxes):
        random_angle = random.randint(0, 3) * 90
        if random_angle != 0:
            # rotate by random_angle degrees.
            im = im.rotate(random_angle, expand=True)
            x, y, x2, y2 = copy.deepcopy(boxes[:, 0]), copy.deepcopy(boxes[:, 1]), copy.deepcopy(boxes[:, 2]), copy.deepcopy(boxes[:, 3])
            w = x2 - x
            h = y2 - y
            if random_angle == 90:
                boxes[:, 0] = y
                boxes[:, 1] = im.size[1] - x - w
                boxes[:, 2] = h
                boxes[:, 3] = w
            elif random_angle == 180:
                boxes[:, 0] = im.size[0] - x - w
                boxes[:, 1] = im.size[1] - y - h
            elif random_angle == 270:
                boxes[:, 0] = im.size[0] - y - h
                boxes[:, 1] = x
                boxes[:, 2] = h
                boxes[:, 3] = w

        # Random horizontal flip
        if random.randint(0, 1):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            boxes[:, 0] = im.size[0] - boxes[:, 0] - boxes[:, 2]

        # Apply image brightness, contrast etc augmentation
        if self.augment_brightness:
            brightness_factor = random.normalvariate(1, self.augment_brightness)
            brightness_factor = max(0, brightness_factor)
            im = adjust_brightness(im, brightness_factor)
        if self.augment_contrast:
            contrast_factor = random.normalvariate(1, self.augment_contrast)
            contrast_factor = max(0, contrast_factor)
            im = adjust_contrast(im, contrast_factor)
        if self.augment_hue:
            hue_factor = random.normalvariate(0, self.augment_hue)
            hue_factor = max(-0.5, hue_factor)
            hue_factor = min(0.5, hue_factor)
            im = adjust_hue(im, hue_factor)
        if self.augment_saturation:
            saturation_factor = random.normalvariate(1, self.augment_saturation)
            saturation_factor = max(0, saturation_factor)
            im = adjust_saturation(im, saturation_factor)
        x0, y0, w, h = boxes[0]
        x1 = x0 + w
        y1 = y0 + h
        return im, x0, y0, x1, y1
