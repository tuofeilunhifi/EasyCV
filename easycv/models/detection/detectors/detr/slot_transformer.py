# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn

from easycv.models.builder import NECKS


class SlotGrouping(nn.Module):

    def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.temp = temp
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)

    def forward(self, x):
        x_prev = x
        slots = self.slot_embed(
            torch.arange(0, self.num_slots,
                         device=x.device)).unsqueeze(0).repeat(
                             x.size(0), 1, 1)
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(slots, dim=2),
                            F.normalize(x, dim=1))
        attn = (dots / self.temp).softmax(dim=1) + self.eps
        # attn = (dots / self.temp).sigmoid() + self.eps
        slots = torch.einsum('bdhw,bkhw->bkd', x_prev,
                             attn / attn.sum(dim=(2, 3), keepdim=True))
        return slots, dots


@NECKS.register_module
class SlotTransformer(nn.Module):

    def __init__(self, num_prototypes, in_channels, d_model, temp):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.slot_layer = SlotGrouping(num_prototypes, d_model, temp)

    def init_weights(self):
        pass

    def forward(self, src, img_metas):
        src = src[0]

        src = self.input_proj(src)
        slots, dots = self.slot_layer(src)

        return slots.unsqueeze(0)
