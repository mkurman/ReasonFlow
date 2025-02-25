import torch
from typing import Tuple, Optional

def select_best_thinkers(
    config,
    outputs: torch.Tensor,
    scores: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select best thinkers using both quality and diversity metrics.
    
    Args:
        config: The ReasonFlowConfig object
        outputs: Output tensors from all thinkers
        scores: Quality scores for each thinker
        hidden_states: Optional hidden states for diversity calculation
    
    Returns:
        Tuple of (best_thinker_indices, best_scores)
    """
    # Fast path for single thinker (common case optimization)
    if len(outputs) == 1:
        return torch.tensor([0], device=scores.device), scores
        
    # Quality component (normalized scores) - with improved numerical stability
    quality_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    
    # Only calculate diversity if weight > 0
    if config.diversity_weight > 0.001:  # Skip near-zero weights for performance
        # Diversity component calculation - optimized to avoid loops when possible
        diversity_scores = torch.zeros_like(quality_scores)
        if len(outputs) > 1:
            # Use hidden states if available (typically more efficient)
            output_embeds = hidden_states if hidden_states is not None else outputs
            
            # Preallocate output tensor for better memory efficiency
            num_thinkers = len(output_embeds)
            diversity_scores = torch.zeros(num_thinkers, device=quality_scores.device)
            
            # Pre-compute all pairwise similarities in one operation if not too many thinkers
            if num_thinkers <= 16:  # Threshold where batched is more efficient
                # Normalize embeddings once for cosine similarity
                norms = torch.norm(output_embeds, dim=-1, keepdim=True)
                normalized_embeds = output_embeds / (norms + 1e-6)
                
                # Compute full similarity matrix at once (more efficient)
                similarity_matrix = torch.mm(
                    normalized_embeds.view(num_thinkers, -1), 
                    normalized_embeds.view(num_thinkers, -1).t()
                )
                
                # Remove self-similarities from diagonal
                similarity_matrix.fill_diagonal_(0)
                
                # Compute mean similarities for each thinker (excluding self)
                non_zeros = (num_thinkers - 1)
                mean_similarities = similarity_matrix.sum(dim=1) / non_zeros
                diversity_scores = 1 - mean_similarities
            else:
                # Fall back to loop for very large number of thinkers
                for i in range(len(outputs)):
                    other_outputs = torch.cat([output_embeds[:i], output_embeds[i+1:]])
                    similarities = torch.cosine_similarity(
                        output_embeds[i].unsqueeze(0),
                        other_outputs,
                        dim=-1
                    )
                    diversity_scores[i] = 1 - similarities.mean()
        
        # Combined score with weights
        combined_scores = (
            (1 - config.diversity_weight) * quality_scores + 
            config.diversity_weight * diversity_scores
        )
    else:
        # Fast path: just use quality scores when diversity weight is negligible
        combined_scores = quality_scores
    
    # Optimized top-k selection
    topk = min(config.topk_thinkers, len(combined_scores))
    if topk == 1:
        # Faster version for common case (single best thinker)
        best_idx = torch.argmax(combined_scores).unsqueeze(0)
        return best_idx, combined_scores[best_idx]
    else:
        # Standard topk for multiple thinkers
        best_indices = combined_scores.topk(topk, sorted=True).indices
        return best_indices, combined_scores[best_indices]