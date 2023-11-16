# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

"""
OFA
"""
from typing import Optional

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .ofa import OFAModel, OFAClassificationHead

logger = logging.getLogger(__name__)


@register_model("ofa_classification")
class OFAClassificationModel(OFAModel):
    __jit_unused_properties__ = ["supported_targets"]

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        cls_dim = self.args.encoder_embed_dim
        self.classification_head = OFAClassificationHead(
            input_dim=cls_dim,
            inner_dim=cls_dim*4,
            num_classes=26, # TODO: add as args
            activation_fn='gelu',
            pooler_dropout=0.1,
            pooler_classifier='mlp',
        )
        self.classification_head = weight_init(self.classification_head)
        
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()
        
        self.args.cls_token = True
            


    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        patch_images: Optional[torch.Tensor] = None,
        patch_images_2: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        code_masks: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True
            
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            patch_images=patch_images,
            patch_masks=patch_masks,
            patch_images_2=patch_images_2,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            sample_patch_num=sample_patch_num
        )
        
        cls_token = encoder_out['encoder_out'][0][:1].transpose(0, 1)
        # print(f'[----------] classification cls_tokne: {cls_token.shape}')
        x_cls = self.classification_head(cls_token)
        # print(f'[----------] classification cls_tokne out: {x_cls.shape}')
        
        x, extra = self.decoder(
            prev_output_tokens,
            code_masks=code_masks,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return x, extra, x_cls


def weight_init(m):
    '''
    https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    return m
                    
@register_model_architecture("ofa_classification", "ofa_cls_large")
def ofa_classification_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.0)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_classifier = getattr(args, "pooler_classifier", "mlp")

    args.resnet_drop_path_rate = getattr(args, "resnet_drop_path_rate", 0.0)
    args.encoder_drop_path_rate = getattr(args, "encoder_drop_path_rate", 0.0)
    args.decoder_drop_path_rate = getattr(args, "decoder_drop_path_rate", 0.0)

    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    args.token_bucket_size = getattr(args, "token_bucket_size", 256)
    args.image_bucket_size = getattr(args, "image_bucket_size", 42)

    args.freeze_encoder_embedding = getattr(args, "freeze_encoder_embedding", False)
    args.freeze_decoder_embedding = getattr(args, "freeze_decoder_embedding", False)
    args.add_type_embedding = getattr(args, "add_type_embedding", True)
    args.attn_scale_factor = getattr(args, "attn_scale_factor", 2)

    args.code_image_size = getattr(args, "code_image_size", 128)
    args.patch_layernorm_embedding = getattr(args, "patch_layernorm_embedding", True)
    args.code_layernorm_embedding = getattr(args, "code_layernorm_embedding", True)
    args.entangle_position_embedding = getattr(args, "entangle_position_embedding", False)
    args.disable_entangle = getattr(args, "disable_entangle", False)
    args.sync_bn = getattr(args, "sync_bn", False)

    args.scale_attn = getattr(args, "scale_attn", False)
    args.scale_fc = getattr(args, "scale_fc", False)
    args.scale_heads = getattr(args, "scale_heads", False)
    args.scale_resids = getattr(args, "scale_resids", False)

    args.orig_patch_image_size = getattr(args, "orig_patch_image_size", 256)

    args.classification_num = getattr(args, "classification_num", 14)


@register_model_architecture("ofa_classification", "ofa_cls_base")
def ofa_classification_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.resnet_type = getattr(args, "resnet_type", "resnet101")
    ofa_classification_large_architecture(args)


@register_model_architecture("ofa_classification", "ofa_cls_huge")
def ofa_classification_huge_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1280)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    ofa_classification_large_architecture(args)


@register_model_architecture("ofa_classification", "ofa_cls_medium")
def ofa_classification_medium_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 512)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.resnet_type = getattr(args, "resnet_type", "resnet101")
    ofa_classification_large_architecture(args)


@register_model_architecture("ofa_classification", "ofa_cls_tiny")
def ofa_classification_tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.resnet_type = getattr(args, "resnet_type", "resnet50")
    ofa_classification_large_architecture(args)
