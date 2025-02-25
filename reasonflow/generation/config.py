from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ReasonFlowConfig:
    """Configuration for ReasonFlow generation.
    
    This class defines all parameters that control the behavior of ReasonFlow's
    multi-path generation process.
    """
    num_of_thinkers: int = 1
    num_of_thoughts: int = 3
    topk_thinkers: int = 1
    acceptance_threshold: float = 0.5
    temperatures: Optional[List[float]] = field(default_factory=lambda: [0.7, 1.3])
    diversity_weight: float = 0.3
    min_temperature: float = 0.5
    max_temperature: float = 1.5
    dynamic_temperature: bool = True
    adaptive_thoughts: bool = False  # Disabled due to stability issues
    uncertainty_threshold: float = 0.7
    min_acceptance_threshold: float = 0.3
    max_acceptance_threshold: float = 0.8