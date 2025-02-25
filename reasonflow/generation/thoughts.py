def determine_thought_length(
    config,
    input_complexity: int, 
    iteration: int,
    uncertainty: float
) -> int:
    """Determine the number of thoughts to use.
    
    Note: The adaptive thoughts feature has been disabled due to stability issues.
    This function now always returns the base number of thoughts from the config.
    
    Args:
        config: The ReasonFlowConfig object
        input_complexity: Complexity of input (e.g., input length)
        iteration: Current iteration number
        uncertainty: Uncertainty score (0.0-1.0)
    
    Returns:
        Number of thoughts to generate
    """
    # Always return the base number of thoughts
    return config.num_of_thoughts