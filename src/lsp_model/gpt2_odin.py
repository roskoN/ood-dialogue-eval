from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling_gpt2 import (GPT2LMHead,
                                                   GPT2PreTrainedModel)
from torch.nn import CrossEntropyLoss

from modeling_base import GPT2ModelFP16


class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2ModelFP16(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        lm_labels=None,
        past=None,
    ):
        hidden_states, presents = self.transformer(
            input_ids, position_ids, token_type_ids, past
        )
        # import pdb; pdb.set_trace()
        g_logits = F.sigmoid(self.linear_g_component(hidden_states))
        h_cosine_sim = F.cosine_similarity(x1=self.weight_h, x2=hidden_states, dim=1)

        lm_logits = g_logits / h_cosine_sim
        if lm_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction="none")
            loss1 = loss_fct1(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)
            )
            loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
            label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
            loss = torch.sum(loss1) / torch.sum(label_size)
            ppl = torch.exp(
                torch.mean(torch.sum(loss1, dim=1).float() / label_size.float())
            )
            # ppl = torch.mean(torch.exp(torch.sum(loss1, dim=1)/label_size))
            return loss, ppl
        return lm_logits, presents

    def forward_pointwise(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        lm_labels=None,
        past=None,
    ):
        hidden_states, presents = self.transformer(
            input_ids, position_ids, token_type_ids, past
        )
        # import pdb; pdb.set_trace()
        g_logits = F.sigmoid(self.linear_g_component(hidden_states))
        h_cosine_sim = F.cosine_similarity(x1=self.weight_h, x2=hidden_states, dim=1)

        lm_logits = g_logits / h_cosine_sim
        if lm_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction="none")
            loss1 = loss_fct1(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)
            )
            loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
            label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
            loss1 = torch.sum(loss1, dim=1) / label_size
            ppl1 = torch.exp(loss1)

            return loss1, ppl1
        return lm_logits, presents
