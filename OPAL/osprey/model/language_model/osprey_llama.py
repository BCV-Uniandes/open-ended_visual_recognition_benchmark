#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..layer import MaskExtractor
from ..loss import DistributedContrastiveLossMultiPositive
from ..osprey_arch import OspreyMetaForCausalLM, OspreyMetaModel


class CausalLMOutputWithPastAndContrastiveLoss(CausalLMOutputWithPast):

    def __init__(self, vision_acc=None, text_acc=None, grad_norm=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_acc: Optional[torch.FloatTensor] = vision_acc
        self.text_acc: Optional[torch.FloatTensor] = text_acc
        self.grad_norm: Optional[torch.FloatTensor] = grad_norm


class OspreyConfig(LlamaConfig):
    model_type = "osprey"


class OspreyLlamaModel(OspreyMetaModel, LlamaModel):
    config_class = OspreyConfig

    def __init__(self, config: LlamaConfig):
        super(OspreyLlamaModel, self).__init__(config)


class OspreyLlamaForCausalLM(LlamaForCausalLM, OspreyMetaForCausalLM):
    config_class = OspreyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = OspreyLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mask_extractor = MaskExtractor()
        self.temperature = nn.Parameter(torch.tensor(0.1))

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        img_metas=None,
        masks=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        gt_labels: Optional[List[torch.LongTensor]] = None,
        step: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_token_len = input_ids.shape[1]
        self.model = self.model.bfloat16()
        # print(step + 1, (step + 1) % 1 == 0)
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = (
            self.prepare_inputs_labels_for_multimodal(
                input_ids,
                masks,
                attention_mask,
                past_key_values,
                labels,
                images,
                contrastive_learning=(
                    ((step + 1) % 2 == 0) if labels is not None else False
                ),
            )
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.bfloat16()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        self.lm_head = self.lm_head.to(hidden_states.dtype)
        logits = self.lm_head(hidden_states)

        loss = None
        vision_acc = None
        text_acc = None
        if labels is not None:
            # take the average of all the hidden states for each valid token in each sequence
            # text_embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(
            #     1
            # ) / attention_mask.sum(1).unsqueeze(-1)

            # select index of the last valid token in each sequence
            last_token_index = torch.sum(attention_mask, dim=1) - 1
            text_embeddings = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                last_token_index,
            ]

            if (step + 1) % 2 == 0:
                loss_fct = DistributedContrastiveLossMultiPositive()
                loss, vision_acc, text_acc = loss_fct(
                    labels.bfloat16(),
                    text_embeddings,
                    None,  # prompts_per_image``
                    apply_hidden_norm=True,
                    temperature=self.temperature,
                    is_symmetric_loss=False,
                    stabilize_loss=False,
                    is_dist=True,
                    key1=gt_labels,
                    key2=gt_labels,
                    get_accuracy=True,
                    random_negatives=False,
                    max_batch_size=1024,
                )
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]

            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPastAndContrastiveLoss(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vision_acc=vision_acc,
            text_acc=text_acc,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("osprey", OspreyConfig)
AutoModelForCausalLM.register(OspreyConfig, OspreyLlamaForCausalLM)
