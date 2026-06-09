"""TracIn-style gradient attribution.

Per-example loss gradients are the shared primitive behind TracIn and influence
functions. This module computes them over a chosen parameter subset and turns a
set of them into a similarity (influence) matrix.

  TracIn(i, j)  ~=  <g_i, g_j>            (single-checkpoint form)
  Influence(i, j) = - g_j^T H^{-1} g_i    (adds an iHVP term -- separate step)
"""
import torch


_HEAD_ALIASES = {"lm_head", "head", "output"}


def _output_head_params(model):
    """The output-projection weight, resolved by identity. Handles tied
    embeddings (small Qwen3 models have no standalone 'lm_head.weight';
    the head IS model.embed_tokens.weight)."""
    head = model.get_output_embeddings()
    w = getattr(head, "weight", None)
    if w is None:
        return []
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and p is w]


def select_param_subset(model, name_filter="lm_head"):
    """Choose which parameters to attribute over. The output head ('lm_head')
    or a layer filter like 'layers.27' carry real task signal; the final norm
    does not. The head is resolved by identity so tied embeddings still work."""
    if name_filter is None or name_filter in _HEAD_ALIASES:
        params = _output_head_params(model)
        if params:
            return params
    params = [(n, p) for n, p in model.named_parameters()
              if p.requires_grad and name_filter and name_filter in n]
    if not params:
        last = getattr(getattr(model, "config", None), "num_hidden_layers", "N")
        last = last - 1 if isinstance(last, int) else "N"
        raise ValueError(
            f"name_filter '{name_filter}' matched no parameters. "
            f"For the output head use name_filter='lm_head' (tied heads are now "
            f"handled), or target a block like 'layers.{last}'.")
    return params


def sequence_nll(model, prompt_ids, gen_ids, device):
    """Teacher-forced negative log-likelihood of gen_ids following prompt_ids."""
    full = torch.cat([prompt_ids[0], gen_ids]).unsqueeze(0).to(device)
    p_len = prompt_ids.shape[1]
    logits = model(full).logits
    logp = torch.log_softmax(logits[0, p_len - 1:-1, :], dim=-1)
    return -logp.gather(-1, gen_ids.to(device).unsqueeze(-1)).squeeze(-1).sum()


def per_example_param_grad(model, prompt_ids, gen_ids, device,
                           name_filter="lm_head", create_graph=False):
    """Flat gradient of the sequence NLL w.r.t. the selected parameters.
    Set create_graph=True when you need to differentiate again (HVP / iHVP)."""
    params = [p for _, p in select_param_subset(model, name_filter)]
    nll = sequence_nll(model, prompt_ids, gen_ids, device)
    model.zero_grad(set_to_none=True)
    grads = torch.autograd.grad(nll, params, create_graph=create_graph)
    flat = torch.cat([g.reshape(-1) for g in grads])
    return (flat if create_graph else flat.detach()), nll.item()


def tracin_matrices(grads):
    """Influence (dot) and cosine matrices from per-example grads.

    Accepts a [n, dim] tensor OR a list of [dim] vectors. Upcasts at most two
    vectors to fp32 at a time, so it stays memory-frugal even for a very large
    subset like an untied lm_head (~600M dims on Qwen3-8B), where materialising
    a full fp32 [n, dim] matrix would OOM.
    """
    if isinstance(grads, torch.Tensor):
        grads = [grads[i] for i in range(grads.shape[0])]
    n = len(grads)
    dot = torch.zeros(n, n)
    flat = [g.reshape(-1) for g in grads]
    for i in range(n):
        gi = flat[i].float()
        for j in range(i, n):
            v = torch.dot(gi, flat[j].float())
            dot[i, j] = dot[j, i] = v
    d = dot.diagonal().clamp_min(1e-12).sqrt()
    cos = dot / torch.outer(d, d).clamp_min(1e-12)
    return dot, cos


def random_projection(grad_matrix, out_dim=4096, seed=0):
    """Johnson-Lindenstrauss projection to make large grads storable/comparable
    at scale (standard TracIn trick). Approximately preserves dot products."""
    g = torch.Generator().manual_seed(seed)
    in_dim = grad_matrix.shape[1]
    R = torch.randn(in_dim, out_dim, generator=g) / (out_dim ** 0.5)
    return grad_matrix.float() @ R