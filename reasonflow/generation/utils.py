import torch
import torch.nn as nn
from typing import Tuple, List

def calculate_loss(hidden_states: torch.Tensor, logits: torch.Tensor, vocab_size: int, num_thinkers: int) -> torch.Tensor:
    """Helper method to calculate loss for thinker selection.
    
    Args:
        hidden_states: Hidden states tensor
        logits: Logits tensor
        vocab_size: Size of vocabulary
        num_thinkers: Number of thinkers
        
    Returns:
        Loss tensor for each thinker
    """
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    hidden_states = hidden_states.view(-1, vocab_size)
    input_ids_cpy = logits[:, 1:, :].argmax(dim=-1).reshape(-1)
    return loss_fct(hidden_states, input_ids_cpy).view(num_thinkers, -1).sum(dim=-1)


def truncate_output_to_eos(output, eos_token_ids: List[int]):
    """Truncate output at the EOS token.
    
    Args:
        output: Output tensor or array
        eos_token_ids: List of EOS token IDs
        
    Returns:
        Truncated output
    """
    for i, output_token in enumerate(output.flatten()):
        if output_token in eos_token_ids:
            return output[:, :i+1]
    return output


def has_eos(output, eos_token_ids: List[int]) -> bool:
    """Check if output contains an EOS token.
    
    Args:
        output: Output tensor or array
        eos_token_ids: List of EOS token IDs
        
    Returns:
        True if output contains an EOS token, False otherwise
    """
    flat_output = output.flatten()
    for token_id in eos_token_ids:
        if token_id in flat_output:
            return True
    return False