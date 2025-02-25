import torch
from typing import List, Optional

def dynamic_temperature_scheduling(
    iteration: int, 
    config,
    confidence_scores: Optional[torch.Tensor] = None
) -> List[float]:
    """Adjust temperature dynamically based on generation progress and confidence.
    
    Args:
        iteration: The current iteration number
        config: The ReasonFlowConfig object
        confidence_scores: Optional tensor of confidence scores
        
    Returns:
        List of temperature values to use
    """
    base_temps = config.temperatures.copy()
    
    if not config.dynamic_temperature:
        return base_temps
        
    if iteration < 3:  # Early iterations: promote exploration
        return [min(config.max_temperature, t * 1.2) for t in base_temps]
    
    if confidence_scores is not None and confidence_scores.max().item() < 0.3:
        # Low confidence: increase temperature diversity
        return [
            max(config.min_temperature, 
                min(config.max_temperature, t * (1.0 + (i * 0.15))))
            for i, t in enumerate(base_temps)
        ]
        
    if iteration > 10:  # Later iterations: reduce temperature
        return [max(config.min_temperature, t * 0.9) for t in base_temps]
        
    return base_temps