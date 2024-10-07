from transformers import TrainerCallback
import torch
import gc


class EvaluateFirstStepCallback(TrainerCallback):
    """
    Run eval after the first training step.
    Used to have a more precise evaluation learning curve.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param}"
    )


def tear_down_torch():
    """
    Teardown for PyTorch.
    Clears GPU cache for both CUDA and MPS.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
