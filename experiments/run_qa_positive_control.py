#!/usr/bin/env python3
"""
QA positive-control for white-box TracIn attribution.

WHY THIS EXISTS
---------------
The debate testbed produced a null result (near-orthogonal lm_head gradients,
same-side not > cross-side). A null is uninterpretable without a positive
control: we cannot tell whether the *method* is broken (length/loss/tie/norm
bug), whether we're reading the *wrong subset* (the head encodes which tokens
were emitted, not routed content), or whether debate *stance* simply isn't a
strong gradient signal.

This script builds a task with KNOWN ground truth and NO agent machinery, so it
isolates "is the attribution math correct?" from "is the relay plumbing correct?".
Get this green first; only then wrap the segments in the two-agent relay.

DESIGN
------
K structurally identical statements, one per "project":
    "The codeword for project {NAME} is {CODE}."
CODEs are random / out-of-distribution so the model cannot answer from its
weights (the gradient must route through the statement that supplied the fact).
Statements are shuffled (fact position randomized). Then a query:
    "Question: What is the codeword for project {TARGET}?\nAnswer: {CODE}"
We attribute the MEAN per-token NLL of the answer-CODE span back to each
statement via TracIn = cosine of per-subset parameter gradients.
Ground truth: the TARGET project's statement must rank #1.

CONTROLS BAKED IN
-----------------
- length:   one shared template => segments are length-matched; loss is MEAN
            per-token NLL (not summed) so a long turn can't masquerade as
            influential. Cosine is also norm-invariant as a second guard.
- position: fact slot randomized each seed; per-position top-1 is reported so
            recency bias is visible rather than hidden.
- subset:   ATTR_SUBSET=sweep crosses the control with several parameter subsets
            in ONE run (tied head/embed, early/mid/late layer MLPs). This is the
            "lm_head vs layers.N" question and the positive control together.

SUCCESS CRITERION
-----------------
Top-1 accuracy >> 1/K and MRR near 1.0 for at least one subset, stable across
fact position. If even this fails, the attribution code is wrong before any
relay is worth building. If it passes on layers.N but not the head, that both
validates the method AND explains the debate null (head = token identity,
layers = routed content).

INTEGRATION NOTE
----------------
The TracIn primitive is reimplemented inline to stay self-contained and avoid
guessing src/attribution/tracin.py signatures. Once green, swap the inline
grad/cosine for tracin.py and wrap segments in experiments/run_qa_relay.py.

CONFIG (env vars, matching the existing run-config convention)
    MODEL_ID     default Qwen/Qwen3-0.6B
    DTYPE        float32 (default; clean grads on small models) | bfloat16
    K            number of statements / segments (default 6)
    SEEDS        number of randomized episodes (default 20)
    ATTR_SUBSET  'sweep' (default) | any substring of a parameter name
                 (e.g. 'lm_head', 'embed_tokens', 'layers.13.mlp')
"""

import os
import random
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
    os.environ.get("DTYPE", "float32")
]
K = int(os.environ.get("K", "6"))
SEEDS = int(os.environ.get("SEEDS", "20"))
ATTR_SUBSET = os.environ.get("ATTR_SUBSET", "sweep")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_NAMES = [
    "Zephyr", "Onyx", "Larch", "Vesper", "Quill",
    "Marlow", "Cinder", "Halcyon", "Bramble", "Pertwee",
]


def rand_code(rng):
    """Out-of-distribution codeword the model cannot know parametrically."""
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(2))
    digits = "".join(rng.choice(string.digits) for _ in range(4))
    return f"{letters}-{digits}"


def build_episode(rng):
    """Return shuffled statements + a query targeting one of them.

    chunks layout: [stmt_0 .. stmt_{K-1}, query_prefix, answer_code]
    The fact's slot among the statements is randomized (recency control).
    """
    names = rng.sample(PROJECT_NAMES, K)
    codes = [rand_code(rng) for _ in range(K)]
    fact_slot = rng.randrange(K)  # which shuffled position holds the queried fact
    target_name, target_code = names[fact_slot], codes[fact_slot]

    statements = [
        f"The codeword for project {n} is {c}.\n" for n, c in zip(names, codes)
    ]
    query_prefix = (
        f"Question: What is the codeword for project {target_name}?\nAnswer: "
    )
    answer = target_code
    chunks = statements + [query_prefix, answer]
    return chunks, fact_slot


def encode_with_spans(tokenizer, chunks):
    """Tokenize chunks separately, concatenate, and return exact [start,end) spans."""
    ids = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    spans = []
    for text in chunks:
        toks = tokenizer(text, add_special_tokens=False).input_ids
        start = len(ids)
        ids.extend(toks)
        spans.append((start, len(ids)))
    input_ids = torch.tensor([ids], device=DEVICE)
    return input_ids, spans


def select_subsets(model):
    """Build {label: [(name, param), ...]} for the requested subset(s).

    Dedups tied parameters by id (small Qwen3 ties lm_head to embed_tokens).
    """
    n = model.config.num_hidden_layers
    if ATTR_SUBSET == "sweep":
        early, mid, late = 1, n // 2, n - 1
        patterns = {
            "head/embed(tied)": "embed_tokens",
            f"layer{early}.mlp": f"layers.{early}.mlp",
            f"layer{mid}.mlp": f"layers.{mid}.mlp",
            f"layer{late}.mlp": f"layers.{late}.mlp",
            f"layer{late}.attn": f"layers.{late}.self_attn",
        }
    else:
        patterns = {ATTR_SUBSET: ATTR_SUBSET}

    subsets = {}
    for label, pat in patterns.items():
        seen, params = set(), []
        for name, p in model.named_parameters():
            if pat in name and id(p) not in seen:
                seen.add(id(p))
                params.append((name, p))
        params.sort(key=lambda x: x[0])  # fixed order so vectors align across segments
        if params:
            subsets[label] = params
        else:
            print(f"[warn] subset '{label}' (pattern '{pat}') matched no params")
    return subsets


def span_loss(model, input_ids, span):
    """MEAN per-token NLL over the target span, conditioned on its left context.

    HF shifts internally: labels[p] is predicted from logits[p-1], so masking
    labels outside the span restricts the (mean-reduced) loss to that span.
    """
    labels = torch.full_like(input_ids, -100)
    a, b = span
    labels[0, a:b] = input_ids[0, a:b]
    return model(input_ids=input_ids, labels=labels).loss


def grads_for_subsets(model, loss, subsets):
    """One backward, read each subset's flattened gradient. Stored on CPU/fp32."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    out = {}
    for label, params in subsets.items():
        vec = torch.cat([p.grad.reshape(-1) for _, p in params])
        out[label] = vec.detach().float().cpu().clone()
    return out


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    print(f"model={MODEL_ID} dtype={DTYPE} device={DEVICE} K={K} seeds={SEEDS} "
          f"subset={ATTR_SUBSET}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    model.eval()  # no dropout; deterministic grads

    subsets = select_subsets(model)
    union_ids = {id(p) for ps in subsets.values() for _, p in ps}
    for p in model.parameters():
        p.requires_grad_(id(p) in union_ids)  # freeze everything else: speed + memory
    print("subsets:", {k: f"{sum(p.numel() for _, p in v):,} params"
                       for k, v in subsets.items()})

    # metrics[label] -> dict of running tallies
    metrics = {label: {"top1": 0, "rr": 0.0, "pos": [0] * K} for label in subsets}

    for seed in range(SEEDS):
        rng = random.Random(seed)
        chunks, fact_slot = build_episode(rng)
        input_ids, spans = encode_with_spans(tokenizer, chunks)
        seg_spans = spans[:K]          # the K statements
        answer_span = spans[-1]        # the codeword to attribute

        # query gradient (stored), then each segment compared against it
        q_grads = grads_for_subsets(model, span_loss(model, input_ids, answer_span),
                                    subsets)
        sims = {label: [] for label in subsets}
        for span in seg_spans:
            s_grads = grads_for_subsets(model, span_loss(model, input_ids, span),
                                        subsets)
            for label in subsets:
                sims[label].append(cosine(q_grads[label], s_grads[label]))

        for label in subsets:
            order = sorted(range(K), key=lambda j: sims[label][j], reverse=True)
            rank = order.index(fact_slot)  # 0 == top-1
            metrics[label]["top1"] += int(rank == 0)
            metrics[label]["rr"] += 1.0 / (rank + 1)
            if rank == 0:
                metrics[label]["pos"][fact_slot] += 1

    print(f"\n{'subset':<20}{'top1':>8}{'MRR':>8}   per-position top1 (slot0..K-1)")
    print("-" * 70)
    chance = 1.0 / K
    for label in subsets:
        m = metrics[label]
        top1 = m["top1"] / SEEDS
        mrr = m["rr"] / SEEDS
        flag = "  <-- recovers fact" if top1 > 0.5 else ""
        print(f"{label:<20}{top1:>8.2f}{mrr:>8.2f}   {m['pos']}{flag}")
    print(f"\nchance top1 = {chance:.2f}.  A subset is validated if top1 >> chance "
          f"and MRR ~ 1.0, evenly across positions (no recency artifact).")


if __name__ == "__main__":
    main()
