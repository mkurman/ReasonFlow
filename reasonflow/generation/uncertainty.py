import torch
from typing import Optional

def measure_uncertainty(logits: torch.Tensor, top_k: int = 5) -> float:
    """Measure model uncertainty based on logits distribution.
    
    Args:
        logits: Logits tensor from the model
        top_k: Number of top tokens to consider
        
    Returns:
        Uncertainty score (0.0-1.0), higher means more uncertain
    """
    # Cache tensor size and prepare computation
    batch_size = logits.size(0)
    seq_len = logits.size(1)
    
    # Skip softmax when possible (for pure logits-based metrics)
    if batch_size * seq_len > 128:  # For larger batches, use optimized approach
        # Optimized computation for large batches - use log_softmax for better numerical stability
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get top-k without materializing full softmax
        top_log_probs, _ = torch.topk(log_probs, top_k, dim=-1)
        top_probs = torch.exp(top_log_probs)
        
        # Vectorized computation of probability ratio
        prob_ratio = top_probs[..., 0] / (top_probs[..., 1] + 1e-6)
    else:
        # Direct approach for smaller tensors
        probs = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probs, top_k, dim=-1)
        prob_ratio = top_probs[..., 0] / (top_probs[..., 1] + 1e-6)
    
    # More efficient certainty calculation (avoid redundant operations)
    certainty = torch.clamp(prob_ratio / 3.0, 0, 1).mean().item()
    
    return 1.0 - certainty


def adapt_acceptance_threshold(
    config,
    iteration: int,
    uncertainty: float,
    prev_acceptance_rate: Optional[float] = None
) -> float:
    """Adapt the acceptance threshold based on context.
    
    Args:
        config: The ReasonFlowConfig object
        iteration: Current iteration number
        uncertainty: Uncertainty score (0.0-1.0)
        prev_acceptance_rate: Previous acceptance rate, if available
        
    Returns:
        Adjusted acceptance threshold
    """
    base_threshold = config.acceptance_threshold
    
    # Early iterations: be more lenient
    if iteration < 3:
        return max(
            config.min_acceptance_threshold,
            base_threshold * 0.8
        )
    
    # High uncertainty: lower threshold
    if uncertainty > config.uncertainty_threshold:
        return max(
            config.min_acceptance_threshold,
            base_threshold * 0.85
        )
    
    # Low uncertainty: can be more selective
    if uncertainty < 0.3:
        return min(
            config.max_acceptance_threshold,
            base_threshold * 1.2
        )
    
    # If previous acceptance rate is very low, reduce threshold
    if prev_acceptance_rate is not None and prev_acceptance_rate < 0.2:
        return max(
            config.min_acceptance_threshold,
            base_threshold * 0.9
        )
    
    return base_threshold