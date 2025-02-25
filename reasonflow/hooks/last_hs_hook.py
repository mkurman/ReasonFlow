def get_module_name(module, model):
    for name, mod in model.named_modules():
        if mod == module:
            return name
    return None


def register_non_norm_hook(model):
    """
    Registers an optimized forward hook to normalization layers.
    Performance-focused implementation that minimizes data movement.

    Args:
        model (torch.nn.Module): The model to register the hook to.
    """
    import torch
    from functools import lru_cache

    # Pre-allocate empty tensor for efficiency
    empty_tensor = torch.zeros(0, 0, 0, device='cpu')
    
    # Initialize with empty tensor
    model.non_normalized_hidden_states = dict(
        tensor=empty_tensor,
        name="not_initialized"
    )
    
    # Cache module names to avoid repeated lookups
    @lru_cache(maxsize=128)
    def get_module_name_cached(module_id):
        for name, mod in model.named_modules():
            if id(mod) == module_id:
                return name
        return None

    def hook_fn(module, input, output):
        """
        Optimized hook function with minimal overhead.
        Only extracts the last hidden state which is what we need.
        """
        if not input or len(input) == 0 or input[0] is None:
            return  # Skip if input is empty

        try:
            # Extract only the last token's hidden state and detach
            # This is much more efficient than copying the whole tensor
            with torch.no_grad():  # Ensure we don't track gradients
                hs_last = input[0][:, -1:, :].detach()
                
                # Store without unnecessary copies
                model.non_normalized_hidden_states = dict(
                    tensor=hs_last,  # Keep on GPU for faster access later
                    name=get_module_name_cached(id(module)),
                )
        except Exception:
            # Fast error recovery
            model.non_normalized_hidden_states = dict(
                tensor=empty_tensor,
                name="error_capturing_hidden_states",
            )

    # Register hook with proper error handling
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        # Register with direct function reference for better performance
        hook = model.model.norm.register_forward_hook(hook_fn)
        # Store hook handle for potential cleanup
        model._norm_hook_handle = hook
        print("Optimized hook registered for model.norm")
    else:
        print("No 'norm' attribute found in the model.")
