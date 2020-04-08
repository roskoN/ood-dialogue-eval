# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import logging
import math

import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling_gpt2 import (MLP, Attention, Block,
                                                   GPT2Model, LayerNorm)

logger = logging.getLogger(__name__)


class AttentionFP16(Attention):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(AttentionFP16, self).__init__(nx, n_ctx, config, scale)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)  # point out by Yen-Chun, FP16 overflow

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)


class BlockFP16(Block):
    def __init__(self, n_ctx, config, scale=False):
        super(BlockFP16, self).__init__(n_ctx, config, scale)
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = AttentionFP16(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)


class GPT2ModelFP16(GPT2Model):
    def __init__(self, config):
        super(GPT2ModelFP16, self).__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = BlockFP16(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self.init_weights)
