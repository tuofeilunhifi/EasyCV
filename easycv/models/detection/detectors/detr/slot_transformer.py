# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from easycv.models.builder import NECKS
from easycv.models.utils import (TransformerEncoder, TransformerEncoderLayer,
                                 _get_activation_fn, _get_clones)
from .detr_transformer import SinePositionalEncoding


class SlotGrouping(nn.Module):

    def __init__(self,
                 d_model=256,
                 dim_feedforward=2048,
                 dropout=0.1,
                 nhead=8,
                 activation='relu',
                 temp=0.07,
                 eps=1e-6):
        super().__init__()
        self.temp = temp
        self.eps = eps

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                x,
                tgt,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                ori_pos: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(
        #     q,
        #     k,
        #     value=tgt,
        #     attn_mask=tgt_mask,
        #     key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        x_prev = x
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(tgt, dim=2),
                            F.normalize(x, dim=1))
        attn = (dots / self.temp).softmax(dim=1) + self.eps
        tgt2 = torch.einsum('bdhw,bkhw->bkd', x_prev,
                            attn / attn.sum(dim=(2, 3), keepdim=True))

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SlotTransformerEncoderLayer(TransformerEncoderLayer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SlotTransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None,
                 d_model=256,
                 query_scale_type=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale_type = query_scale_type
        self.norm = norm

        self.slot_layers = _get_clones(SlotGrouping(), num_layers)

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                ori_pos_embed: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt: Optional[Tensor] = None,
                spatial_shape: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = src

        intermediate = []
        for layer_id, (layer, slot_layer) in enumerate(
                zip(self.layers, self.slot_layers)):
            # rescale the content and pos sim
            if self.query_scale_type == 'cond_elewise':
                pos_scales = self.query_scale(output)
            else:
                pos_scales = 1
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos * pos_scales)

            # reverse HWxNxC to NxCxHxW
            output = output.permute(1, 2, 0).view(*spatial_shape)

            tgt = slot_layer(
                output,
                tgt,
                memory_key_padding_mask=mask,
                ori_pos=ori_pos_embed,
                pos=pos,
                query_pos=query_pos)
            intermediate.append(tgt)

            # flatten NxCxHxW to HWxNxC
            output = output.flatten(2).permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        return torch.stack(intermediate)


@NECKS.register_module
class SlotTransformer(nn.Module):

    def __init__(self,
                 in_channels=1024,
                 num_queries=100,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.slot_embed = nn.Embedding(num_queries, d_model)
        self.positional_encoding = SinePositionalEncoding(
            num_feats=128, normalize=True)

        encoder_layer = SlotTransformerEncoderLayer(d_model, nhead,
                                                    dim_feedforward, dropout,
                                                    activation,
                                                    normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = SlotTransformerEncoder(encoder_layer,
                                              num_encoder_layers, encoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries

    def init_weights(self):
        for p in self.named_parameters():
            if 'input_proj' in p[0] or 'slot_embed' in p[
                    0] or 'positional_encoding' in p[0]:
                continue
            if p[1].dim() > 1:
                nn.init.xavier_uniform_(p[1])

    def forward(self, src, img_metas):
        src = src[0]

        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        bs = src.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        mask = src.new_ones((bs, input_img_h, input_img_w))
        for img_id in range(bs):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            mask[img_id, :img_h, :img_w] = 0

        src = self.input_proj(src)

        # slots
        tgt = self.slot_embed(
            torch.arange(0, self.num_queries,
                         device=src.device)).unsqueeze(0).repeat(bs, 1, 1)

        # interpolate masks to have the same spatial shape with x
        mask = F.interpolate(
            mask.unsqueeze(1), size=src.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(mask)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]

        # flatten NxCxHxW to HWxNxC
        spatial_shape = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        ori_pos_embed = pos_embed
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        memory = self.encoder(
            src,
            src_key_padding_mask=mask,
            ori_pos_embed=ori_pos_embed,
            pos=pos_embed,
            tgt=tgt,
            spatial_shape=spatial_shape,
            query_pos=None)

        return memory
