# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from .label_smoothed_cross_entropy import *


@register_criterion(
    "adjust_label_smoothed_cross_entropy_with_cls", dataclass=AdjustLabelSmoothedCrossEntropyCriterionConfig
)
class AdjustLabelSmoothedCrossEntropyCriterionWithClassification(AdjustLabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        ignore_eos=False,
        report_accuracy=False,
        drop_worst_ratio=0,
        drop_worst_after=0,
        use_rdrop=False,
        reg_alpha=1.0,
        sample_patch_num=196,
        constraint_range=None
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            ignore_eos=False,
            report_accuracy=False,
            drop_worst_ratio=0,
            drop_worst_after=0,
            use_rdrop=False,
            reg_alpha=1.0,
            sample_patch_num=196,
            constraint_range=None
        )
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.ignore_eos = ignore_eos
        self.report_accuracy = report_accuracy
        self.drop_worst_ratio = drop_worst_ratio
        self.drop_worst_after = drop_worst_after
        self.use_rdrop = use_rdrop
        self.reg_alpha = reg_alpha
        self.sample_patch_num = sample_patch_num

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if isinstance(sample, list):
            if self.sample_patch_num > 0:
                sample[0]['net_input']['sample_patch_num'] = self.sample_patch_num
            loss_v1, sample_size_v1, logging_output_v1 = self.forward(model, sample[0], update_num, reduce)
            loss_v2, sample_size_v2, logging_output_v2 = self.forward(model, sample[1], update_num, reduce)
            v1_weight = 0.2
            v2_weight = 1.0
            # cls_weight = 1.0
            # cls_loss= cls_weight * (cls_loss_v1 + cls_loss_v2) * 0.5
            # print(f"[---------] cls_loss: {cls_loss}")
            loss = v1_weight * (loss_v1 / sample_size_v1) + v2_weight * (loss_v2 / sample_size_v2)
            # loss = loss_v2 / sample_size_v2
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                # "cls_loss": cls_loss.data,
                "loss_v1": loss_v1.data,
                "loss_v2": loss_v2.data,
                "nll_loss": logging_output_v1["nll_loss"].data / sample_size_v1 + logging_output_v2["nll_loss"].data / sample_size_v2,
                "ntokens": logging_output_v1["ntokens"] + logging_output_v2["ntokens"],
                "nsentences": logging_output_v1["nsentences"] + logging_output_v2["nsentences"],
                "sample_size": 1,
                "sample_size_v1": sample_size_v1,
                "sample_size_v2": sample_size_v2,
            }
            return loss, sample_size, logging_output

        if self.use_rdrop:
            construct_rdrop_sample(sample)

        net_output = model(**sample["net_input"])
        net_output, net_cls_output = net_output[:2], net_output[-1]
        loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)
        cls_loss_weight = 1
        cls_loss = self.classification_loss(net_cls_output, sample["multi_hot_label"]) * cls_loss_weight
        # cls_loss = self.classification_loss_OHEM(net_cls_output, sample["multi_hot_label"], ratio=0.75) * cls_loss_weight
        # loss = loss + cls_loss
        loss = cls_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        logging_output = {
            "loss": loss.data,
            "cls_loss": cls_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def classification_loss(self, pred, target):
        loss = nn.BCEWithLogitsLoss()
        target = target.unsqueeze(1).to(pred.device).to(pred.dtype)
        output = loss(pred, target)
        # print(f'[------------] classification loss: {output}')
        return output

    def classification_loss_OHEM(self, pred, target, ratio=0.5):
        loss = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        target = target.unsqueeze(1).to(pred.device).to(pred.dtype)
        num_class = target.shape[-1]
        losses = []
        for i in range(num_class):
            cur_loss = loss(pred[..., i], target[..., i])
            losses.append(cur_loss.item())
        keep_num = int(num_class * ratio)
        losses = sorted(losses)[-keep_num:]
        output = sum(losses) / keep_num
        output = torch.tensor(output).to(pred.device).to(pred.dtype)
        # print(f'[------------] classification loss with OHEM: {output}')
        return output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum_cls = sum(log.get("cls_loss", 0) for log in logging_outputs) / max(len(logging_outputs), 1)
        loss_sum_v1 = sum(log.get("loss_v1", 0) for log in logging_outputs)
        loss_sum_v2 = sum(log.get("loss_v2", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_cls", loss_sum_cls, max(len(logging_outputs), 1), round=3
        )
        metrics.log_scalar(
            "loss_v1", loss_sum_v1 / max(sample_size_v1, 1), max(sample_size_v1, 1), round=3
        )
        metrics.log_scalar(
            "loss_v2", loss_sum_v2 / max(sample_size_v2, 1), max(sample_size_v2, 1), round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v1", sample_size_v1, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v2", sample_size_v2, 1, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
