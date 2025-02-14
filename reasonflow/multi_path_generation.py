import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerationMixin, GenerationConfig, LogitsProcessorList
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from .utils import has_eos, truncate_output_to_eos
from .multi_path_forward import multiple_path_with_noise_inference
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class ReasonFlowConfig:
    num_of_thinkers: int = 1
    num_of_thoughts: int = 3
    topk_thinkers: int = 1
    acceptance_threshold: float = 0.5
    temperatures: Optional[List[float]] = field(default_factory=lambda: [0.7, 1.3])


class ReasonFlow(GenerationMixin):
    def __init__(
        self,
        config: ReasonFlowConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = model.generation_config

        # Generation Mixin backward compatibility
        self.config.is_encoder_decoder = model.config.is_encoder_decoder
        self.config._get_non_default_generation_parameters = model.config._get_non_default_generation_parameters

        self.initialize_parameters()
        self.model.forward = multiple_path_with_noise_inference.__get__(self.model)

    def initialize_parameters(self) -> None:
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
        Generate multiple thoughts from a single input text.

        Args:
            text (Union[str, list[str]]): The input text.
            max_new_tokens (int): The maximum number of tokens to generate.
            stream (bool): Whether to stream the output.
            apply_chat_template (bool): Whether to apply the chat template.
            device (str): The device to use.
            logits_processor (Optional[LogitsProcessorList]): The logits processor.
            torch_dtype (str): The torch data type.
            generation_config (Optional[GenerationConfig]): The generation configuration.
            use_tokens (bool): Whether to use tokens (True) or internal thoughts (hidden states, False) during thought generation.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The generated tokens and the best thinkers summary.

        """

        self.model.eval()
        self.model.to(device)

        num_of_generated_tokens = 0
        output = None

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
            print(self.tokenizer.decode(input_ids['input_ids'][0]), end="")

        past_key_values = None

        accumulated_chunks = []
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.min_tokens_to_keep = self.num_of_thoughts
        
        input_ids_length = input_ids['input_ids'].shape[-1]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None

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
            input_ids_seq_length=input_ids['input_ids'].shape[1],
            encoder_input_ids=input_ids['input_ids'],
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=input_ids['input_ids'].device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )

        while num_of_generated_tokens < max_new_tokens:
            with torch.autocast(dtype=getattr(torch, torch_dtype), device_type=device):
                with torch.no_grad():
                    # Build a single batch dimension for every thinker
                    if num_of_generated_tokens == 0:
                        batch_input_ids = input_ids["input_ids"].repeat_interleave(self.num_of_thinkers, dim=0)
                        batch_attention_mask = input_ids["attention_mask"].repeat_interleave(self.num_of_thinkers, dim=0)
                        cache_position = torch.arange(input_ids["input_ids"].shape[-1], device=input_ids["input_ids"].device) + 1
                        position_ids = torch.arange(input_ids["input_ids"].shape[-1], device=input_ids["input_ids"].device).unsqueeze(0) + 1
                    else:
                        batch_input_ids = input_ids["input_ids"][..., -1:].repeat_interleave(self.num_of_thinkers, dim=0)
                        batch_attention_mask = input_ids["attention_mask"][..., -1:].repeat_interleave(self.num_of_thinkers, dim=0)
                        
                        cache_position = torch.arange(input_ids["input_ids"].shape[-1], device=input_ids["input_ids"].device) + 1
                        position_ids = torch.arange(input_ids["input_ids"].shape[-1], device=input_ids["input_ids"].device).unsqueeze(0) + 1

                        cache_position = cache_position[..., -1:].contiguous()
                        position_ids = position_ids[..., -1:].contiguous()
                    
                    thinker_ids = torch.arange(self.num_of_thinkers, device=batch_input_ids.device).repeat(input_ids["input_ids"].shape[0])

                    # Pass all thinkers in one forward call
                    results = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        use_cache=True,
                        output_attentions=False,
                        thinker_ids=thinker_ids,
                        acceptance_threshold=self.config.acceptance_threshold,
                        temperatures=self.config.temperatures,
                        past_key_values=past_key_values,
                        num_of_thoughts=self.num_of_thoughts,  # only produce max_depth tokens each pass
                        cache_position=cache_position,
                        position_ids=position_ids,
                        return_dict=True,
                        use_tokens=use_tokens,
                        **model_kwargs,
                    )

                    # Take only the first new tokens from each thinker
                    final_logits = results.logits[:, -self.num_of_thoughts:-self.num_of_thoughts+1, :].contiguous()

                    past_key_values = results.past_key_values

                    loss_fct = nn.CrossEntropyLoss(reduction="none")

                    hidden_states = results.logits[:, :-1, :].contiguous().view(-1, self.model.config.vocab_size)
                    input_ids_cpy = results.logits[:, 1:, :].argmax(dim=-1).reshape(-1)

                    loss = (
                        loss_fct(hidden_states, input_ids_cpy).view(self.num_of_thinkers, -1).sum(dim=-1)
                    )

                    # Select the best thinkers
                    best_thinkers = loss.topk(
                        min(self.config.topk_thinkers, self.num_of_thinkers if self.num_of_thinkers > 2 else 1),
                        dim=0,
                        sorted=True,
                        largest=False,
                    ).indices.view(-1)

                    best_thinkers_summary.append(best_thinkers)

                    best_logits = final_logits[best_thinkers]

                    result = best_logits.sum(dim=0).squeeze(1)

                    result = prepared_logits_processor(input_ids['input_ids'], result)

                    # token selection
                    if generation_config.do_sample:
                        result = torch.clamp(result, 0.0, 1.0)
                        result = torch.multinomial(result.squeeze(), num_samples=1).unsqueeze(0).squeeze(-1)
                    else:
                        result = torch.argmax(result, dim=-1)

                    if len(result.size()) == 1:
                        result = result.unsqueeze(1)

                    result = result.view(-1, 1)

                    past_key_values = results.past_key_values

                    # Set the past_key_values to be equale the best thinkers past_key_values for the next iteration
                    past_key_values = [
                        [past_key_values[i][j][best_thinkers].mean(dim=0, keepdim=True).repeat_interleave(self.num_of_thinkers, dim=0) for j in range(len(past_key_values[0]))]
                    for i in range(len(past_key_values))]

                    torch.cuda.empty_cache()

                    accumulated_chunks.clear()
             
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