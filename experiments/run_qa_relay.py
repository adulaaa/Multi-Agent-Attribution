#!/usr/bin/env python3
"""
Two-agent QA relay -- FRAMING ABLATION (+ empty-prompt fix, + content-only test).

Findings so far (1.7B, layers.27, 50 seeds):
    control (bare statements) ........ 0.80 top-1
    FRAMING=preamble ................. 0.54   (preamble dents, signal survives)
    FRAMING=full (preamble+roles) .... 0.18   (chance: signal destroyed)
    FRAMING=none / roles ............. crashed on an empty-prompt edge case
                                        in tracin.sequence_nll (first turn starts
                                        at token 0; Qwen3 prepends no BOS).

So the role delimiters ('Agent A:'/'Agent B:'), not the preamble, are the prime
suspect: every turn opens with identical delimiter tokens, putting the late-layer
gradient in a repeated-template regime whose dominant direction is shared across
all turns -- the same failure mode as the output head.

This version:
  * prepends a sentinel token so every attributed span has >=1 token of left
    context (fixes the crash without editing tracin.py).
  * adds CONTENT_ONLY: attribute each turn's CONTENT tokens only, masking the
    'Agent A:' delimiter out of the gradient target (it stays in context). This
    tests the obvious fix and ports directly to the debate (attribute argument
    content, not speaker/turn-boundary tokens).

CONFIG (env)
    MODEL_ID  default Qwen/Qwen3-0.6B   (use 1.7B; control = 0.80 there)
    DTYPE     float32 (default) | bfloat16
    K, SEEDS  default 6, 20             (SEEDS=50 to match the control)
    ATTR_SUBSET   name_filter for tracin.select_param_subset; default last block
    FRAMING       none | roles | preamble | full   (default none)
    CONTENT_ONLY  0 (default) | 1       attribute content tokens only
"""

import os
import sys
import random
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from src.attribution.tracin import per_example_param_grad, tracin_matrices  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
    os.environ.get("DTYPE", "float32")
]
K = int(os.environ.get("K", "6"))
SEEDS = int(os.environ.get("SEEDS", "20"))
ATTR_SUBSET = os.environ.get("ATTR_SUBSET", "")
FRAMING = os.environ.get("FRAMING", "none")
CONTENT_ONLY = os.environ.get("CONTENT_ONLY", "0") == "1"
HETERO = os.environ.get("HETERO", "0") == "1"  # heterogeneous (non-repeated) turns
CENTER = os.environ.get("CENTER", "none")  # none | mean | pc1 | pc2  (remove shared direction)
RECALL = os.environ.get("RECALL", "0") == "1"  # greedily check the model actually recalls the gold code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_PREAMBLE = FRAMING in ("preamble", "full")
USE_ROLES = FRAMING in ("roles", "full")

PROJECT_NAMES = [
    "Zephyr", "Onyx", "Larch", "Vesper", "Quill",
    "Marlow", "Cinder", "Halcyon", "Bramble", "Pertwee",
]

TEMPLATES = [  # each embeds {n} and {c}; breaks the K-identical repetition
    "The codeword for project {n} is {c}.",
    "Project {n} was assigned access code {c}.",
    "{c} is the registry key for project {n}.",
    "We logged {c} as the clearance string for project {n}.",
    "The access token for project {n} reads {c}.",
    "Records show project {n} uses the code {c}.",
    "Operator note: project {n} maps to {c}.",
    "Under the new scheme, {c} secures project {n}.",
]


def rand_code(rng):
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(2))
    digits = "".join(rng.choice(string.digits) for _ in range(4))
    return f"{letters}-{digits}"


def _build_episode(rng):
    """Returns chunks, fact_slot, turn_pairs[(prefix_idx, content_idx)], answer_idx.
    Splitting each turn into (delimiter, content) chunks lets CONTENT_ONLY attribute
    just the content while the delimiter remains in context."""
    names = rng.sample(PROJECT_NAMES, K)
    codes = [rand_code(rng) for _ in range(K)]
    fact_slot = rng.randrange(K)
    target_name, target_code = names[fact_slot], codes[fact_slot]

    a_prefix = "Agent A: " if USE_ROLES else ""
    b_prefix = "Agent B: " if USE_ROLES else ""
    # template assignment is independent of fact_slot, so phrasing can't
    # systematically favor the queried turn. rng drawn only when HETERO so
    # the non-HETERO stream (and the ladder numbers) stays bit-identical.
    tpls = rng.sample(TEMPLATES, K) if HETERO else ["The codeword for project {n} is {c}."] * K

    chunks = []
    if USE_PREAMBLE:
        chunks.append("Two agents share a codeword registry. "
                      "Agent A reports entries; Agent B answers queries.\n")
    turn_pairs = []
    for i, (n, c) in enumerate(zip(names, codes)):
        pidx = len(chunks); chunks.append(a_prefix)            # "" if no roles
        cidx = len(chunks); chunks.append(tpls[i].format(n=n, c=c) + "\n")
        turn_pairs.append((pidx, cidx))
    chunks.append(b_prefix + f"Question: What is the codeword for project "
                             f"{target_name}?\nAnswer: ")
    chunks.append(target_code)
    return chunks, fact_slot, turn_pairs, len(chunks) - 1


def _encode_with_spans(tokenizer, chunks):
    # Always lead with a sentinel so every span has >=1 token of left context.
    lead = tokenizer.bos_token_id
    if lead is None:
        lead = tokenizer.eos_token_id
    ids = [lead] if lead is not None else []
    spans = []
    for text in chunks:
        toks = tokenizer(text, add_special_tokens=False).input_ids
        start = len(ids)
        ids.extend(toks)
        spans.append((start, len(ids)))
    return torch.tensor([ids], device=DEVICE), spans


def _grad_for_span(model, full_ids, span, subset):
    a, b = span
    g, nll = per_example_param_grad(model, full_ids[:, :a], full_ids[0, a:b],
                                    DEVICE, name_filter=subset)
    return g.detach().cpu(), nll, (b - a)


def _attribute(grads, center):
    """answer-vs-turn cosines, optionally after removing a shared direction.
    mean: subtract the turn-mean from answer and turns (removes common baseline).
    pcK : also project out the top-K principal directions of the turn set
          (removes the dominant shared/positional axis, e.g. recency)."""
    answer = grads[0].float()
    turns = [g.float() for g in grads[1:]]
    X = torch.stack(turns)                       # [K, dim]
    mean = X.mean(0)
    if center != "none":
        answer = answer - mean
        turns = [t - mean for t in turns]
    if center.startswith("pc"):
        r = int(center[2:] or "1")
        Xc = torch.stack(turns)                  # already mean-removed
        G = Xc @ Xc.t()                          # [K, K] Gram (rank<=K, cheap at huge dim)
        _, evecs = torch.linalg.eigh(G)          # ascending eigenvalues
        V = Xc.t() @ evecs[:, -r:]               # [dim, r] top-r feature directions
        V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)
        proj = lambda v: v - V @ (V.t() @ v)
        answer = proj(answer)
        turns = [proj(t) for t in turns]
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    # diagnostic: mean off-diagonal RAW turn-turn cosine (shared-direction strength)
    raw = [g.float() for g in grads[1:]]
    n = len(raw)
    pair = [cos(raw[i], raw[j]) for i in range(n) for j in range(i + 1, n)]
    return [cos(answer, t) for t in turns], (sum(pair) / len(pair) if pair else 0.0)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
    model.eval()
    subset = ATTR_SUBSET or f"layers.{model.config.num_hidden_layers - 1}"
    print(f"model={MODEL_ID} dtype={DTYPE} device={DEVICE} K={K} seeds={SEEDS} "
          f"subset={subset} FRAMING={FRAMING} CONTENT_ONLY={CONTENT_ONLY} HETERO={HETERO} CENTER={CENTER} "
          f"(preamble={USE_PREAMBLE}, roles={USE_ROLES})")

    top1, rr, pos, raw_cos_sum, ans_nll_sum, recall = 0, 0.0, [0] * K, 0.0, 0.0, 0
    for seed in range(SEEDS):
        rng = random.Random(seed)
        chunks, fact_slot, turn_pairs, answer_idx = _build_episode(rng)
        full_ids, spans = _encode_with_spans(tokenizer, chunks)

        turn_spans = []
        for pidx, cidx in turn_pairs:
            if CONTENT_ONLY:
                turn_spans.append(spans[cidx])                       # content only
            else:
                turn_spans.append((spans[pidx][0], spans[cidx][1]))  # delimiter+content

        g_ans, ans_nll, ans_ntok = _grad_for_span(model, full_ids, spans[answer_idx], subset)
        grads = [g_ans]
        grads += [_grad_for_span(model, full_ids, s, subset)[0] for s in turn_spans]
        sims, raw_cos = _attribute(grads, CENTER)
        raw_cos_sum += raw_cos
        ans_nll_sum += ans_nll / max(ans_ntok, 1)        # mean per-token NLL of gold answer

        if RECALL:                                       # does the model actually answer right?
            astart, aend = spans[answer_idx]
            gold = full_ids[0, astart:aend]
            with torch.no_grad():
                out = model.generate(full_ids[:, :astart],
                                     max_new_tokens=int(gold.shape[0]) + 2,
                                     do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            recall += int(out[0, astart:astart + gold.shape[0]].tolist() == gold.tolist())

        order = sorted(range(K), key=lambda j: sims[j], reverse=True)
        rank = order.index(fact_slot)
        top1 += int(rank == 0)
        rr += 1.0 / (rank + 1)
        if rank == 0:
            pos[fact_slot] += 1

    print(f"\n{'metric':<14}{'value':>8}")
    print("-" * 24)
    print(f"{'top1 acc':<14}{top1 / SEEDS:>8.2f}")
    print(f"{'MRR':<14}{rr / SEEDS:>8.2f}")
    print(f"per-position top1 (slot0..K-1): {pos}")
    print(f"mean raw turn-turn cosine: {raw_cos_sum / SEEDS:.3f}  "
          f"(high => one shared direction dominates the gradients)")
    print(f"mean answer per-token NLL: {ans_nll_sum / SEEDS:.3f}  "
          f"(high => model would NOT produce the gold code: nothing to attribute)")
    if RECALL:
        print(f"greedy recall of gold code: {recall / SEEDS:.2f}  "
              f"(can the model even answer correctly under this framing?)")
    print(f"\nchance top1 = {1.0 / K:.2f}. FRAMING=none should match the control "
          f"(~0.80 @ 1.7B); CONTENT_ONLY=1 under FRAMING=full tests whether masking "
          f"the delimiter from the gradient target recovers the signal.")


if __name__ == "__main__":
    main()