import torch
import numpy as np

def gradient_attribution(model, input_tensor: torch.Tensor, target_output: torch.Tensor,
                         baseline: torch.Tensor = None) -> np.ndarray:
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    steps = 50
    input_tensor.requires_grad_(True)
    total_grads = 0
    for alpha in torch.linspace(0, 1, steps):
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        output = model(interpolated)
        if isinstance(output, tuple):
            output = output[0]
        model.zero_grad()
        output[:, target_output].sum().backward(retain_graph=True)
        grads = interpolated.grad.clone()
        total_grads += grads
    avg_grads = total_grads / steps
    attribution = (input_tensor - baseline) * avg_grads
    return attribution.detach().cpu().numpy()

def text_gradient_attribution(model, tokenizer, text: str, target_class: int = 0) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt")
    embeddings = model.get_input_embeddings()(inputs["input_ids"])
    return gradient_attribution(model, embeddings, target_class)
