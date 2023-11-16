# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string
import math
import copy

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


class ReportMLMDataset(OFADataset):
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
            
        self.mask_span_distribution = None
        self.poisson_lambda = 3.0
        self.mask_length = "span-poisson"
        self.code_image_size = 128
        self.mask_ratio = 0.3
        self.random_ratio = 0
        self.num_bins = 1000
        self.code_dict_size = 8192
        self.replace_length = -1
        self.mask_idx = src_dict.index("<mask>")
        
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
    
    def process_pure_text(self, infos):
        patch_image = torch.zeros((3, self.code_image_size*2, self.code_image_size*2))
        patch_mask = torch.tensor([False])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        examples = []
        for _ in range(2):
            # uniq_id, text = self.pure_text_dataset[index]
            uniq_id, text = infos
            text = text.strip().lower()
            # print(f"Text: {text}")
            text_item = self.encode_text(" {}".format(text), length=512)
            # print(f"Text item: {text_item}")
            # print(f"Len :{len(text_item)}")
            text_item = text_item[-256:]
            # print(f"After clip: {text_item}")
            
            text_item = torch.cat([self.bos_item, text_item, self.eos_item])
            mask_text_item = self.add_whole_word_mask(text_item.clone(), self.mask_ratio)
            # print(f"After mask: {mask_text_item}")
            # exit()
            prefix_item = self.encode_text(' what is the complete text of " "?')
            src_item = torch.cat([prefix_item[:-2], mask_text_item[1:-1], prefix_item[-2:]])
            tgt_item = text_item[1:-1]
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            target_item = torch.cat([tgt_item, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
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
    
    def word_starts(self, source):
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
    
    def __getitem__(self, index):
        uniq_id, image, caption, tags = self.dataset[index]
        caption_bak = copy.deepcopy(caption)

        image = Image.open(image)
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if self.split == 'train' and not self.scst:
            caption = caption.translate(self.transtab).strip()
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
        # vqa_example = self.process_qa((uniq_id, patch_image, caption, tags))
        mlm_example = self.process_pure_text((uniq_id, caption_bak))
        return report_example, mlm_example

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

