
from typing import List, Union
import torch

def has_eos(output: torch.Tensor, eos_token: List[int]) -> bool:
    """
    Checks if the output contains any of the EOS token IDs.
    """
    return any(x for x in eos_token if x in output)

def truncate_output_to_eos(output: torch.Tensor, eos_token: List[int]) -> torch.Tensor:
    """
    Truncates the generated output up to the first detected EOS token.
    """
    for i in range(output.shape[1]):
        if has_eos(output[:, :i], eos_token):
            return output[:, :i]
    return output