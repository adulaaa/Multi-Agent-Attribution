"""Input-embedding attribution (Integrated Gradients), corrected.

Replaces the previous version, which (a) called requires_grad_() on a non-leaf
interpolated tensor and (b) passed embeddings as the positional input_ids arg to
an HF causal LM. This version uses leaf interpolation points and inputs_embeds.
"""
import torch

def integrated_gradients_embeddings(model, tokenizer, text, target_id=None,
                                    target_pos=-1, steps=50, device=None):
    """Per-token saliency of `text` for the model's prediction at target_pos,
    via Integrated Gradients over input embeddings. Returns [seq] scores."""
    device = device or next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt").to(device)
    embed = model.get_input_embeddings()
    inp = embed(enc["input_ids"]).detach()
    baseline = torch.zeros_like(inp)
    total = torch.zeros_like(inp)

    for alpha in torch.linspace(0, 1, steps, device=device):
        x = (baseline + alpha * (inp - baseline)).detach().requires_grad_(True)
        out = model(inputs_embeds=x, attention_mask=enc.get("attention_mask"))
        step_logits = out.logits[0, target_pos, :]
        tid = int(step_logits.argmax()) if target_id is None else int(target_id)
        model.zero_grad(set_to_none=True)
        step_logits[tid].backward()
        total = total + x.grad.detach()

    avg_grads = total / steps
    attribution = (inp - baseline) * avg_grads     
    return attribution.sum(-1).squeeze(0).cpu()