from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
from pytorch_pretrained_bert.modeling_gpt2 import (GPT2LMHead,
                                                   GPT2PreTrainedModel)
from torch.distributions.bernoulli import Bernoulli
from torch.nn import CrossEntropyLoss

from .modeling_base import GPT2ModelFP16


class GPT2LMHeadModelBackground(GPT2PreTrainedModel):
    def __init__(self, config, mu=0.15):
        super(GPT2LMHeadModelBackground, self).__init__(config)
        self.transformer = GPT2ModelFP16(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.bernoulli_dist = Bernoulli(torch.tensor([mu]))
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
        if self.training:
            input_size = input_ids.size()
            pertrubation_mask = self.bernoulli_dist.sample(sample_shape=input_size)
            random_ints = torch.randint(low=0, high=self.config.vocab_size, size=input_size)

            for batch_dim in range(input_size[0]):
                for sample_dim in range(input_size[1]):
                    if pertrubation_mask[batch_dim, sample_dim] == 0:
                        continue

                    # Very small chance, 1 / self.config.vocab_size,
                    # that the pertrubed input gets the same id,
                    # i.e. it does not change
                    input_ids[batch_dim, sample_dim] = random_ints[batch_dim, sample_dim]

        hidden_states, presents = self.transformer(
            input_ids, position_ids, token_type_ids, past
        )
        # import pdb; pdb.set_trace()
        lm_logits = self.lm_head(hidden_states)
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
            outputs = (loss, ppl)

            if not self.training:
                outputs = outputs + (lm_logits,)

            return outputs
        return lm_logits, presents

    def forward_pointwise(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        lm_labels=None,
        past=None,
    ):
        input_size = torch.Size(input_ids)
        pertrubation_mask = self.bernoulli_dist.sample(sample_shape=input_size)
        random_ints = torch.randint(low=0, high=self.config.vocab_size, size=input_size)

        for batch_dim in range(input_size[0]):
            for sample_dim in range(input_size[1]):
                if pertrubation_mask[batch_dim, sample_dim] == 0:
                    continue

                # Very small chance, 1 / self.config.vocab_size,
                # that the pertrubed input gets the same id,
                # i.e. it does not change
                input_ids[batch_dim, sample_dim] = random_ints[batch_dim, sample_dim]

        hidden_states, presents = self.transformer(
            input_ids, position_ids, token_type_ids, past
        )
        # import pdb; pdb.set_trace()
        lm_logits = self.lm_head(hidden_states)
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
