def get_module_name(module, model):
    for name, mod in model.named_modules():
        if mod == module:
            return name
    return None


def register_non_norm_hook(model):
    """
    Registers a forward hook to all normalization layers in the model.

    Args:
        model (torch.nn.Module): The model to register the hook to.
    """

    def hook_fn(module, input, output):
        """
        This hook function will be called after the forward pass of the module.

        Args:
            module (torch.nn.Module): The module that triggered the hook.
            input (tuple): The input to the module.
            output (torch.Tensor): The output of the module.
        """
        # module_name = get_module_name(module, model)
        model.non_normalized_hidden_states = dict(
            tensor=input[0].detach().cpu(),
            name=get_module_name(module, model),
        )  # Store the input (non-normalized)

    if hasattr(model.model, "norm"):
        # For models with a 'norm' attribute (like Llama)
        model.model.norm.register_forward_hook(
            lambda module, input, output: hook_fn(module, input, output)
        )
        print("Hook registered for model.norm")
    else:
        print("No 'norm' attribute found in the model.")
