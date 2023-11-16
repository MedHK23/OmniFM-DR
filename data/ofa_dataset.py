# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import logging
import re
import torch.utils.data
from fairseq.data import FairseqDataset
import string
import pickle       

CHINESE_PUNCTUATION = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'      
ENGLISH_PUNCTUATION = string.punctuation


logger = logging.getLogger(__name__)


class OFADataset(FairseqDataset):
    def __init__(self, split, dataset, bpe, src_dict, tgt_dict, mimic_t2i=None, mimic_i2t=None):
        self.split = split
        self.dataset = dataset
        self.bpe = bpe
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.bos = src_dict.bos()
        self.eos = src_dict.eos()
        self.pad = src_dict.pad()
        self.bos_item = torch.LongTensor([self.bos])
        self.eos_item = torch.LongTensor([self.eos])
      
        # load mimic vocabulary
        if mimic_t2i is not None and mimic_i2t is not None:
            with open(mimic_t2i, 'rb') as f:
                self.mimic_t2i_dic = pickle.load(f)
            with open(mimic_i2t, 'rb') as f:
                self.mimic_i2t_dic = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True, use_mimic=False):
        """
        if not use_mimic:
            s = self.tgt_dict.encode_line(
                line=self.bpe.encode(text) if use_bpe else text,
                add_if_not_exist=False,
                append_eos=False
            ).long()
        else:
            s = torch.tensor([self.get_id_by_token(t) for t in text.split()]).long()
        """
        # """
        s = self.tgt_dict.encode_line(
            line=self.bpe.encode(text) if use_bpe else text,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        # """
        """
        ## use mimic vocabulary
        s = self.tgt_dict.encode_line(
                text,
                add_if_not_exist=False,
                append_eos=False
            ).long()
        """

        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    def pre_question(self, question, max_ques_words=None):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question

    def pre_caption(self, caption, max_words=None):
        caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if max_words is not None and len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    def pre_chinese(self, text, max_words):     
        text = text.lower().replace(CHINESE_PUNCTUATION, " ").replace(ENGLISH_PUNCTUATION, " ")     
        text = re.sub(      
            r"\s{2,}",      
            ' ',        
            text,       
        )       
        text = text.rstrip('\n')        
        text = text.strip(' ')[:max_words]      
        return text
  
    def get_id_by_token(self, token):
        if token not in self.mimic_t2i_dic:
            return self.mimic_t2i_dic['<unk>']
        return self.mimic_t2i_dic[token]
