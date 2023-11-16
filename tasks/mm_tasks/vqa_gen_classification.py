# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
import sacrebleu
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.vqa_classification_dataset import VqaClassificationDataset
from data.file_dataset import FileDataset
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD

from .caption import CaptionConfig, CaptionTask
EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

### this task is based on caption task.

@register_task("vqa_gen_cls", dataclass=CaptionConfig)
class VqaGenClsTask(CaptionTask):
    def __init__(self, cfg: CaptionConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        
        # self.negative_dataset = None
        # if self.cfg.data_negative is not None:
        #     self.negative_dataset = FileDataset(self.cfg.data_negative, self.cfg.selected_cols)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = VqaClassificationDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            mimic_t2i=self.cfg.mimic_t2i,
            mimic_i2t=self.cfg.mimic_i2t,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            scst=getattr(self.cfg, 'scst', False),
            negative_dataset=self.negative_dataset
        )