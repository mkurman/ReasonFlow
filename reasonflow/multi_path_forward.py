import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from einops import rearrange
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.generation.utils import LogitsProcessorList, GenerationMixin, GenerationConfig
from dataclasses import dataclass
from dataclasses import field

import warnings
warnings.filterwarnings("ignore")

from .utils import has_eos, truncate_output_to_eos

def multiple_path_with_noise_inference(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    thinker_ids: list[int] = [0],
    num_of_thoughts: int = 3,
    acceptance_threshold: float = 0.5,
    temperatures: Optional[torch.Tensor] = [0.7, 1.3],
    use_tokens: bool = True,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    Perform multiple path inference through the model, allowing for separate thinker paths.

    Args:
        input_ids (torch.LongTensor): The model input IDs.
        attention_mask (Optional[torch.Tensor]): Attention mask.
        position_ids (Optional[torch.LongTensor]): Position IDs.
        past_key_values (Optional[Union[Cache, List[torch.FloatTensor]]]): Past key values.
        use_cache (Optional[bool]): Whether to use cache.
        output_attentions (Optional[bool]): Whether to output attentions.
        output_hidden_states (Optional[bool]): Whether to output hidden states.
        return_dict (Optional[bool]): Whether to return a dictionary.
        cache_position (Optional[torch.LongTensor]): Cache position.
        thinker_ids (list[int]): List of thinker IDs.
        num_of_thoughts (int): Number of thoughts.
        acceptance_threshold (float): Acceptance threshold. Defines the threshold for swapping the probabilities of the top two tokens.
        temperatures (Optional[torch.Tensor]): Temperatures tensor. Defines the temperature for the top two tokens.
        use_tokens (bool): Whether to use tokens (True) or internal thoughts (hidden states, False) during thought generation. Default is True.

    Returns:
        Union[Tuple, CausalLMOutputWithPast]: Model outputs.
    """
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

    num_of_thoughts = max(1, num_of_thoughts)

    logits = None
    input_embs = []

    first_past_key_values = None

    with torch.inference_mode():
        for i in range(input_ids.shape[0]):
            inputs_embeds_thinker = input_ids.clone()

            if i > 0:
                gen = torch.Generator(device=input_ids.device)
                gen.manual_seed(3407 + int(i * 1e3))  # Increment seed per thinker
                inputs_embeds_thinker = self.model.embed_tokens(input_ids[0].clone())
                rand_tensor = torch.rand(inputs_embeds_thinker.shape, device=inputs_embeds_thinker.device, generator=gen)
                inputs_embeds_thinker = inputs_embeds_thinker + rand_tensor * inputs_embeds_thinker
            else:
                inputs_embeds_thinker = self.model.embed_tokens(input_ids[0].clone())

            # hidden_states.append(inputs_embeds_thinker)
            input_embs.append(inputs_embeds_thinker)
            del inputs_embeds_thinker

        input_embs = torch.stack(input_embs, dim=0)

        hidden_states = None

        for j in range(num_of_thoughts):
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=input_embs if j == 0 else hidden_states,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
            )

            hidden_states = outputs["last_hidden_state"]

            hidden_states_dtype = hidden_states.dtype

            past_key_values = outputs["past_key_values"] if use_cache else None

            if first_past_key_values is None and use_cache:
                first_past_key_values = past_key_values

            hidden_states = self.lm_head(hidden_states)

            hidden_states = hidden_states.softmax(dim=-1).to(hidden_states_dtype)

            if j == 0:
                for thinker_id in thinker_ids:
                    if thinker_id == 0:
                        continue

                    max_token = (
                        hidden_states[thinker_id]
                        .topk(thinker_id + 1, dim=-1, largest=True, sorted=True)
                        .indices
                    ).view(-1)

                    # Swap the probabilities of the top two tokens, where the first token is the max token and the second token is the Jth max token
                    prob = hidden_states[thinker_id, ..., -1:, max_token[-1]]

                    prob_ration = prob / hidden_states[thinker_id, ..., -1:, max_token[0]]

                    if prob_ration > acceptance_threshold:
                        hidden_states[thinker_id, ..., -1:, max_token[-1]] = hidden_states[thinker_id, ..., -1:, max_token[0]]
                        hidden_states[thinker_id, ..., -1:, max_token[0]] = prob
                    else:
                        hidden_states[thinker_id, ..., -1:, max_token[-1]] *= temperatures[1]
                        hidden_states[thinker_id, ..., -1:, max_token[0]] *= temperatures[0]
       
            if logits is None:
                logits = hidden_states.float()
            else:
                logits = torch.cat([logits, hidden_states[:, -1:, :].float()], dim=1)

            if use_tokens:
                hidden_states = self.model.embed_tokens(hidden_states[:, -1:, :].argmax(dim=-1))
            else:
                hidden_states = hidden_states[:, -1:, :] @ self.lm_head.weight
                # hidden_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True)

            if cache_position is not None:
                cache_position = cache_position[..., -1:] + 1

            if position_ids is not None:
                position_ids = position_ids[..., -1:] + 1
                    

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=first_past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

