# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
import math
from typing import Optional
from fairseq.tasks import register_task
from fairseq.data import FairseqDataset, iterators

from tasks.ofa_task import OFATask, OFAConfig
from data.pretrain_data.joint_dataset import DRJointDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class DRJointConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": ""}
    )
    vqa_data: Optional[str] = field(
        default=None,
        metadata={"help": "vqa dataset, Yes/No"},
    )
    vqa_multi_data: Optional[str] = field(
        default=None,
        metadata={"help": "vqa, classification"},
    )
    caption_data: Optional[str] = field(
        default=None,
        metadata={"help": "caption data"},
    )
    vg_data: Optional[str] = field(
        default=None,
        metadata={"help": "visual grounded dataset, single object"},
    )
    vg_multi_data: Optional[str] = field(
        default=None,
        metadata={"help": "visual grounded dataset, multi objects"},
    )
    seg_data: Optional[str] = field(
        default=None,
        metadata={"help": "segmentation dataset"},
    )
    MIMIC_path: Optional[str] = field(
        default=None,
        metadata={"help": "root path for MIMIC dataset"},
    )
    private_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "private data path"},
    )
    caption_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "caption data selected cols"},
    )
    vqa_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "vqa data selected cols"},
    )
    seg_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "segmentation data selected cols"},
    )
    
    text_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure text data selected cols"},
    )
    image_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure image data selected cols"},
    )
    detection_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "detection data selected cols"},
    )
    neg_sample_dir: Optional[str] = field(
        default=None,
        metadata={"help": "negative sample directory, which contains captions (taken from all image-text pairs), "
                          "answers (taken from VQA), "
                          "objects (taken form OpenImages) "},
    )
    code_image_size: int = field(
        default=128, metadata={"help": "the resolution of the generated image in the image infilling task"}
    )

    pretrain_seed: int = field(
        default=7,
        metadata={"help": "pretrain seed"},
    )

    mask_ratio: float = field(
        default=0.3,
        metadata={"help": "fraction of words/subwords that will be masked"},
    )
    random_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], use random token this often"},
    )
    keep_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], keep original token this often"},
    )
    mask_length: str = field(
        default="span-poisson",
        metadata={"help": "mask length to choose ['subword', 'word', 'span-poisson']"},
    )
    poisson_lambda: float = field(
        default=3.0,
        metadata={"help": "randomly shuffle sentences for this proportion of inputs"},
    )
    replace_length: int = field(
        default=1,
        metadata={"help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)"},
    )
    mimic_t2i: Optional[str] = field(
        default=None,
        metadata={"help": "mimic vocabulary, token 2 index"},
    )
    mimic_i2t: Optional[str] = field(
        default=None,
        metadata={"help": "mimic vocabulary, index 2 token"},
    )


@register_task("dr_joint_task", dataclass=DRJointConfig)
class DRJointTask(OFATask):
    def __init__(self, cfg: DRJointConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        # self.type2ans_dict = json.load(open(os.path.join(self.cfg.neg_sample_dir, 'type2ans.json')))
        # self.ans2type_dict = {}
        # for type, answer_list in self.type2ans_dict.items():
        #     if type == 'other':
        #         continue
        #     for answer in answer_list:
        #         self.ans2type_dict[answer] = type

        # self.all_object_list = [
        #     row.strip() for row in open(os.path.join(self.cfg.neg_sample_dir, 'object.txt')) if row.strip() != ''
        # ]
        # self.all_caption_list = [
        #     row.strip() for row in open(os.path.join(self.cfg.neg_sample_dir, 'all_captions.txt')) if row.strip() != ''
        # ]

        # self.pure_text_dataset = None
        # self.pure_image_dataset = None
        # self.detection_dataset = None
        
        # self.report_generation_dataset = None
        self.vqa_dataset = None
        self.vqa_multiple_dataset = None
        self.visual_ground_dataset = None
        self.visual_ground_multiple_dataset = None
        self.caption_dataset = None
        self.seg_dataset = None
        self.MIMIC_path = self.cfg.MIMIC_path
        self.private_data_path = self.cfg.private_data_path
        
        if self.cfg.vqa_data is not None:
            self.vqa_dataset = FileDataset(self.cfg.vqa_data, self.cfg.vqa_selected_cols)
        if self.cfg.vqa_multi_data is not None:
            self.vqa_multiple_dataset = FileDataset(self.cfg.vqa_multi_data, self.cfg.vqa_selected_cols)
        if self.cfg.vg_data is not None:
            self.visual_ground_dataset = FileDataset(self.cfg.vg_data, self.cfg.selected_cols)
        if self.cfg.vg_multi_data is not None:
            self.visual_ground_multiple_dataset = FileDataset(self.cfg.vg_multi_data, self.cfg.selected_cols)
        if self.cfg.caption_data is not None:
            self.caption_dataset = FileDataset(self.cfg.caption_data, self.cfg.caption_selected_cols)
        if self.cfg.seg_data is not None:
            self.seg_dataset = FileDataset(self.cfg.seg_data, self.cfg.seg_selected_cols)
            
        # if self.cfg.text_data is not None:
        #     self.pure_text_dataset = FileDataset(self.cfg.text_data, self.cfg.text_selected_cols)
        # if self.cfg.image_data is not None:
        #     self.pure_image_dataset = FileDataset(self.cfg.image_data, self.cfg.image_selected_cols)
        # if self.cfg.detection_data is not None:
        #     self.detection_dataset = FileDataset(self.cfg.detection_data, self.cfg.detection_selected_cols)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        file_path = paths[(epoch - 1) % (len(paths))]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = DRJointDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            mimic_t2i=self.cfg.mimic_t2i,
            mimic_i2t=self.cfg.mimic_i2t,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            seed=self.cfg.pretrain_seed,
            code_dict_size=self.cfg.code_dict_size,
            num_bins=self.cfg.num_bins,
            patch_image_size=self.cfg.patch_image_size,
            code_image_size=self.cfg.code_image_size,
            vqa_dataset=self.vqa_dataset,
            vqa_multiple_dataset=self.vqa_multiple_dataset,
            visual_ground_dataset=self.visual_ground_dataset,
            visual_ground_multiple_dataset=self.visual_ground_multiple_dataset,
            caption_dataset=self.caption_dataset,
            seg_dataset=self.seg_dataset,
            MIMIC_path=self.MIMIC_path,
            private_data_path=self.private_data_path,
            # all_object_list=self.all_object_list,
            # all_caption_list=self.all_caption_list,
            # type2ans_dict=self.type2ans_dict,
            # ans2type_dict=self.ans2type_dict,
            max_image_size=self.cfg.max_image_size,
            mask_ratio=self.cfg.mask_ratio,
            random_ratio=self.cfg.random_ratio,
            keep_ratio=self.cfg.keep_ratio,
            mask_length=self.cfg.mask_length,
            poisson_lambda=self.cfg.poisson_lambda,
            replace_length=self.cfg.replace_length
        )
   
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # create mini-batches with given size constraints
        batch_sampler = [
            [j for j in range(i, min(i + max_sentences, len(dataset)))]
            for i in range(0, len(dataset), max_sentences)
        ]
        total_row_count = dataset.dataset.get_total_row_count()
        num_batches = math.ceil(math.ceil(total_row_count / num_shards) / max_sentences)
        if len(batch_sampler) < num_batches:
            batch_sampler.append([1])

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=1,
            shard_id=0,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size
        )

        return epoch_iter

