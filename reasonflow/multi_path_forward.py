import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from einops import rearrange
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.generation.utils import (
    LogitsProcessorList,
    GenerationMixin,
    GenerationConfig,
)
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
    noise_scale: float = 1.0,  # Controls the scale of random noise
    random_seed_base: int = 3407,  # Base seed for randomization
    use_early_stopping: bool = False,
    early_stopping_threshold: float = 0.1,
    diversity_penalty: float = 0.0,
    debug: bool = False,
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
        noise_scale (float): Controls the scale of random noise.
        random_seed_base (int): Base seed for randomization.
        use_early_stopping (bool): Whether to use early stopping.
        early_stopping_threshold (float): Threshold for early stopping.
        diversity_penalty (float): Penalty to encourage diversity among paths.
        debug (bool): Whether to output debugging information.

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
        # Optimized embedding initialization - compute embeddings in a batch
        base_embeds = self.model.embed_tokens(input_ids[0])  # Only embed once
        
        # Preallocate tensor for all embeddings to avoid multiple allocations
        batch_size = input_ids.shape[0]
        embed_dim = base_embeds.shape[-1]
        seq_len = base_embeds.shape[0]
        all_embeds = torch.empty(
            (batch_size, seq_len, embed_dim), 
            device=input_ids.device, 
            dtype=base_embeds.dtype
        )
        
        # First thinker gets the original embeddings
        all_embeds[0] = base_embeds
        
        # Optimized noise application for remaining thinkers
        if batch_size > 1:
            # Create noise for all other thinkers at once (more efficient)
            for i in range(1, batch_size):
                gen = torch.Generator(device=input_ids.device)
                gen.manual_seed(random_seed_base + int(i * 1e3))  # Increment seed per thinker
                
                # Apply scaled noise directly to the embeddings (avoid copies)
                all_embeds[i] = base_embeds * (1.0 + torch.rand(
                    base_embeds.shape, 
                    device=input_ids.device,
                    generator=gen,
                ) * noise_scale)
        
        input_embs = all_embeds

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

                    dtype_ = hidden_states.dtype

                    if thinker_id == 0:
                        continue

                    max_token = (
                        hidden_states[thinker_id]
                        .topk(thinker_id + 1, dim=-1, largest=True, sorted=True)
                        .indices
                    ).view(-1)

                    # Swap the probabilities of the top two tokens, where the first token is the max token and the second token is the Jth max token
                    prob = hidden_states[thinker_id, ..., -1:, max_token[-1]]

                    prob_ration = (
                        prob / hidden_states[thinker_id, ..., -1:, max_token[0]]
                    )

                    if prob_ration > acceptance_threshold:
                        hidden_states[thinker_id, ..., -1:, max_token[-1]] = (
                            hidden_states[thinker_id, ..., -1:, max_token[0]]
                        )
                        hidden_states[thinker_id, ..., -1:, max_token[0]] = prob
                    else:
                        hidden_states[
                            thinker_id, ..., -1:, max_token[-1]
                        ] *= temperatures[1]
                        hidden_states[
                            thinker_id, ..., -1:, max_token[0]
                        ] *= temperatures[0]

            if logits is None:
                logits = hidden_states.float()
            else:
                logits = torch.cat([logits, hidden_states[:, -1:, :].float()], dim=1)

            # Token selection and output processing - optimized for performance
            if use_tokens:
                # Faster token determination and embedding in one step
                with torch.no_grad():  # Explicitly mark no grad for inference
                    token_indices = hidden_states[:, -1:, :].argmax(dim=-1)
                    hidden_states = self.model.embed_tokens(token_indices)
            else:
                with torch.no_grad():  # Ensure no gradients for maximum speed
                    hidden_states = torch.matmul(hidden_states[:, -1:, :], self.lm_head.weight.data)
                    # Optimized projection and normalization
                    if hasattr(self, "non_normalized_hidden_states") and self.non_normalized_hidden_states["tensor"].size(0) > 0:
                        try:
                            # Direct non-normalized hidden states access with prefetch
                            device = hidden_states.device
                            dtype = hidden_states.dtype
                            
                            non_norm_hs = self.non_normalized_hidden_states["tensor"][:, -1:, :]
                            non_norm_hs = non_norm_hs.to(device=device, dtype=dtype, non_blocking=True)
                            
                            # Vectorized normalization - faster variance calculation
                            epsilon = self.model.norm.variance_epsilon
                            norm_weight = self.model.norm.weight
                            
                            # Fused operation for better performance
                            variance = torch.var(non_norm_hs, dim=-1, keepdim=True, unbiased=False) + epsilon
                            scale = norm_weight * torch.rsqrt(variance)
                            hidden_states = non_norm_hs * scale
                        except Exception:
                            pass

            if cache_position is not None:
                cache_position = cache_position[..., -1:] + 1

            if position_ids is not None:
                position_ids = position_ids[..., -1:] + 1

            if use_early_stopping and j > 1:
                path_variance = torch.var(hidden_states, dim=0).mean()
                if path_variance < early_stopping_threshold:
                    break

        if diversity_penalty > 0:
            similarity_matrix = torch.matmul(hidden_states, hidden_states.transpose(-1, -2))
            diversity_loss = similarity_matrix.sum() * diversity_penalty
            logits -= diversity_loss

        if debug:
            path_stats = {
                "token_diversity": torch.var(hidden_states.argmax(dim=-1), dim=0).mean().item(),
                "probability_confidence": hidden_states.max(dim=-1)[0].mean().item(),
            }
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=first_past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                metadata=path_stats,
            )

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=first_past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
