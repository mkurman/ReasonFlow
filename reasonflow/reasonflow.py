import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import (
    GenerationMixin,
    GenerationConfig,
    LogitsProcessorList,
)
from typing import List, Tuple, Optional, Union
from logging import getLogger

from .generation.config import ReasonFlowConfig
from .generation.temperature import dynamic_temperature_scheduling
from .generation.uncertainty import measure_uncertainty, adapt_acceptance_threshold
from .generation.thoughts import determine_thought_length
from .generation.selection import select_best_thinkers
from .generation.utils import calculate_loss, truncate_output_to_eos, has_eos
from .multi_path_forward import multiple_path_with_noise_inference
from .hooks.last_hs_hook import register_non_norm_hook

logger = getLogger(__name__)


class ReasonFlow(GenerationMixin):
    """
    ReasonFlow implements multi-path reasoning capabilities in language models.
    
    This class extends the HuggingFace GenerationMixin to provide multi-path
    generation with noise, allowing models to explore multiple reasoning paths
    and fuse the best results.
    """
    
    def __init__(
        self,
        config: ReasonFlowConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize ReasonFlow.
        
        Args:
            config: Configuration object for ReasonFlow
            model: The pretrained language model
            tokenizer: The tokenizer for the model
        """
        self.config = config
        self.model = model

        # Register the hook to get the non-normalized hidden states
        register_non_norm_hook(self.model)

        self.tokenizer = tokenizer
        self.generation_config = model.generation_config

        # Generation Mixin backward compatibility
        self.config.is_encoder_decoder = model.config.is_encoder_decoder
        self.config._get_non_default_generation_parameters = (
            model.config._get_non_default_generation_parameters
        )

        self.initialize_parameters()
        self.model.forward = multiple_path_with_noise_inference.__get__(self.model)

    def initialize_parameters(self) -> None:
        """Initialize parameters based on the config."""
        self.num_of_thinkers = max(1, self.config.num_of_thinkers)
        self.num_of_thoughts = max(1, self.config.num_of_thoughts)

        if self.num_of_thinkers == 1:
            self.num_of_thoughts = 1

        if isinstance(self.model.config.eos_token_id, int):
            self.model.config.eos_token_id = [self.model.config.eos_token_id]

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_thoughts(
        self,
        text: Union[str, list[str]],
        max_new_tokens: int = 4096,
        stream: bool = True,
        apply_chat_template: bool = True,
        device: str = "cuda",
        logits_processor: Optional[LogitsProcessorList] = None,
        torch_dtype: str = "float32",
        generation_config: Optional[GenerationConfig] = None,
        use_tokens: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple thoughts with dynamic adaptation.
        
        Args:
            text: Input text or list of texts
            max_new_tokens: Maximum number of tokens to generate
            stream: Whether to stream output
            apply_chat_template: Whether to apply chat template
            device: Device to use
            logits_processor: Logits processor
            torch_dtype: Torch data type
            generation_config: Generation config
            use_tokens: Whether to use tokens directly or hidden states
            **kwargs: Additional arguments for generation
            
        Returns:
            Tuple of (output_tokens, best_thinkers_summary)
        """
        self.model.eval()
        self.model.to(device)

        num_of_generated_tokens = 0
        output = None
        iteration = 0
        prev_acceptance_rate = None

        # Initialize input processing
        if isinstance(text, str):
            text = [text]

        if not apply_chat_template:
            input_ids = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
        else:
            input_ids = self.tokenizer(
                self.tokenizer.apply_chat_template(
                    text, tokenize=False, add_generation_prompt=True
                ),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        input_ids = input_ids.to(self.model.device)

        best_thinkers_summary = []

        if stream:
            print(self.tokenizer.decode(input_ids["input_ids"][0]), end="")

        past_key_values = None

        accumulated_chunks = []
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )
        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.min_tokens_to_keep = self.num_of_thoughts

        input_ids_length = input_ids["input_ids"].shape[-1]

        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and generation_config.min_length is not None
        )

        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name="input_ids",
            inputs_tensor=input_ids,
            input_ids_length=input_ids_length,
        )

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids["input_ids"].shape[1],
            encoder_input_ids=input_ids["input_ids"],
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=input_ids["input_ids"].device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )

        # Initialize current_uncertainty with a default value for the first iteration
        current_uncertainty = 0.5  # Moderate uncertainty at the beginning
        
        while num_of_generated_tokens < max_new_tokens:
            with torch.autocast(dtype=getattr(torch, torch_dtype), device_type=self.model.device.type):
                with torch.no_grad():
                    # Measure input complexity for thoughts
                    input_complexity = input_ids["input_ids"].shape[-1]
                    
                    # Update temperatures dynamically
                    current_temperatures = dynamic_temperature_scheduling(
                        iteration,
                        self.config,
                        prev_acceptance_rate.unsqueeze(0) if prev_acceptance_rate is not None else None
                    )
                    
                    # Use acceptance threshold based on current uncertainty
                    current_acceptance_threshold = (
                        self.config.acceptance_threshold if iteration == 0
                        else adapt_acceptance_threshold(
                            self.config,
                            iteration,
                            current_uncertainty,
                            prev_acceptance_rate
                        )
                    )
                    
                    # Build batch dimension for thinkers
                    if num_of_generated_tokens == 0:
                        batch_input_ids = input_ids["input_ids"].repeat_interleave(
                            self.num_of_thinkers, dim=0
                        )
                        batch_attention_mask = input_ids[
                            "attention_mask"
                        ].repeat_interleave(self.num_of_thinkers, dim=0)
                        cache_position = (
                            torch.arange(
                                input_ids["input_ids"].shape[-1],
                                device=input_ids["input_ids"].device,
                            )
                            + 1
                        )
                        position_ids = (
                            torch.arange(
                                input_ids["input_ids"].shape[-1],
                                device=input_ids["input_ids"].device,
                            ).unsqueeze(0)
                            + 1
                        )
                    else:
                        batch_input_ids = input_ids["input_ids"][
                            ..., -1:
                        ].repeat_interleave(self.num_of_thinkers, dim=0)
                        batch_attention_mask = input_ids["attention_mask"][
                            ..., -1:
                        ].repeat_interleave(self.num_of_thinkers, dim=0)

                        cache_position = (
                            torch.arange(
                                input_ids["input_ids"].shape[-1],
                                device=input_ids["input_ids"].device,
                            )
                            + 1
                        )
                        position_ids = (
                            torch.arange(
                                input_ids["input_ids"].shape[-1],
                                device=input_ids["input_ids"].device,
                            ).unsqueeze(0)
                            + 1
                        )

                        cache_position = cache_position[..., -1:].contiguous()
                        position_ids = position_ids[..., -1:].contiguous()

                    thinker_ids = torch.arange(
                        self.num_of_thinkers, device=batch_input_ids.device
                    ).repeat(input_ids["input_ids"].shape[0])

                    # Determine optimal thought length
                    num_thoughts = determine_thought_length(
                        self.config,
                        input_complexity,
                        iteration,
                        current_uncertainty
                    )

                    # Pass all thinkers in one forward call
                    results = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        use_cache=True,
                        output_attentions=False,
                        thinker_ids=thinker_ids,
                        acceptance_threshold=current_acceptance_threshold,
                        temperatures=current_temperatures,
                        past_key_values=past_key_values,
                        num_of_thoughts=num_thoughts,
                        cache_position=cache_position,
                        position_ids=position_ids,
                        return_dict=True,
                        use_tokens=use_tokens,
                        **model_kwargs,
                    )

                    # Process results
                    final_logits = results.logits[
                        :, -self.num_of_thoughts : -self.num_of_thoughts + 1, :
                    ].contiguous()

                    # Now we can measure uncertainty for next iteration
                    current_uncertainty = measure_uncertainty(final_logits)

                    # Calculate losses and select best thinkers
                    hidden_states = results.logits[:, :-1, :].contiguous()
                    loss = calculate_loss(
                        hidden_states, 
                        results.logits, 
                        self.model.config.vocab_size, 
                        self.num_of_thinkers
                    )
                    
                    # Select best thinkers using quality and diversity
                    best_thinkers, best_scores = select_best_thinkers(
                        self.config,
                        results.logits,
                        -loss,  # Convert loss to score (higher is better)
                        hidden_states
                    )
                    
                    best_thinkers_summary.append(best_thinkers)
                    prev_acceptance_rate = (best_scores > self.config.acceptance_threshold).float().mean()

                    # Process best thinkers' outputs - improved for performance
                    best_logits = final_logits[best_thinkers]
                    
                    # Faster weighted average when we have multiple best thinkers
                    if len(best_thinkers) > 1 and best_scores is not None:
                        # Normalize scores for weighted average
                        weights = torch.softmax(best_scores, dim=0).unsqueeze(-1).unsqueeze(-1)
                        # Weighted sum is more accurate than simple sum
                        result = (best_logits * weights).sum(dim=0).squeeze(1)
                    else:
                        # Simple sum for single thinker case
                        result = best_logits.sum(dim=0).squeeze(1)
                        
                    result = prepared_logits_processor(input_ids["input_ids"], result)

                    # Token selection - optimized
                    if generation_config.do_sample:
                        # Pre-compute softmax with better numerical stability
                        probs = torch.nn.functional.softmax(result * (1.0 / generation_config.temperature), dim=-1)
                        result = torch.multinomial(probs, num_samples=1).view(-1, 1)
                    else:
                        # Faster argmax for greedy decoding
                        result = result.argmax(dim=-1).view(-1, 1)

                    past_key_values = results.past_key_values

                    # Optimize KV-cache handling - avoid unnecessary copies when possible
                    if len(best_thinkers) == 1:
                        # Fast path for single best thinker
                        best_thinker_idx = best_thinkers.item()
                        past_key_values = [
                            [
                                past_key_values[i][j][best_thinker_idx:best_thinker_idx+1]
                                .repeat_interleave(self.num_of_thinkers, dim=0)
                                for j in range(len(past_key_values[0]))
                            ]
                            for i in range(len(past_key_values))
                        ]
                    else:
                        # More efficient averaging with explicit device and dtype
                        device = past_key_values[0][0].device
                        dtype = past_key_values[0][0].dtype
                        past_key_values = [
                            [
                                past_key_values[i][j][best_thinkers]
                                .mean(dim=0, keepdim=True)
                                .to(device=device, dtype=dtype)
                                .repeat_interleave(self.num_of_thinkers, dim=0)
                                for j in range(len(past_key_values[0]))
                            ]
                            for i in range(len(past_key_values))
                        ]
                    
                    # Explicit garbage collection for better memory management
                    if iteration % 5 == 0:  # Less frequent cleanup
                        torch.cuda.empty_cache()

                    accumulated_chunks.clear()
                    
            # Update iteration counter
            iteration += 1
            
            if output is None:
                output = result
            else:
                output = torch.cat([output, result], dim=1)

            input_ids["input_ids"] = torch.cat([input_ids["input_ids"], result], dim=1)

            if input_ids["attention_mask"] is not None:
                input_ids["attention_mask"] = torch.cat(
                    [input_ids["attention_mask"], torch.ones_like(result)], dim=1
                )

            result = result.cpu().numpy()

            num_of_generated_tokens += len(result)

            if has_eos(result, self.model.config.eos_token_id):
                result = truncate_output_to_eos(result, self.model.config.eos_token_id)

                if stream:
                    print(self.tokenizer.decode(result[0].tolist()), end="")
                break

            if stream:
                print(self.tokenizer.decode(result[0].tolist()), end="")

        return output, best_thinkers_summary