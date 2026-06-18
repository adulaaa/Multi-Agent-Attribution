#!/usr/bin/env python3
"""White-box two-agent experiment with gradient attribution.

Runs a deterministic two-agent interaction using WhiteBoxAgent (shared model),
logs the messages and per-turn parameter gradients, and prints the TracIn-style
influence matrices. This is the white-box counterpart to run_baseline.py.

Env vars:
  MODEL_ID     default Qwen/Qwen3-1.7B   (use Qwen/Qwen3-8B with DTYPE=bfloat16)
  DTYPE        default float32           (bfloat16 for >=4B models)
  ATTR_SUBSET  default lm_head           (or e.g. layers.27 for the last block)
  N_TURNS      default 4
  LOG_DIR      default results
"""
import os
import sys
import json
import time

sys.path.append(".")
import torch

from src.agents.white_box_agent import WhiteBoxAgent, load_shared_model
from src.attribution.tracin import per_example_param_grad, tracin_matrices

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DTYPE = os.environ.get("DTYPE", "float32")
ATTR_SUBSET = os.environ.get("ATTR_SUBSET", "lm_head")
N_TURNS = int(os.environ.get("N_TURNS", "4"))
LOG_DIR = os.environ.get("LOG_DIR", "results")


def frame_incoming(turn, topic, opponent_text):
    if turn == 0:
        return (f'The debate proposition is: "{topic}"\n'
                f'State your position in 2-3 sentences. Begin with a clear '
                f'"Yes" or "No" according to your assigned side.')
    return (f'Your opponent just argued:\n"{opponent_text}"\n\n'
            f'Rebut their specific points in 2-3 sentences and defend your own '
            f'side. Do not repeat their wording; make a new argument.')


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    tok, model, device = load_shared_model(MODEL_ID, dtype=DTYPE)
    print(f"{MODEL_ID} on {device} ({DTYPE}) | attr_subset={ATTR_SUBSET}")

    topic = ("Language-model agents should be allowed to call external tools "
             "without human approval.")
    pro = WhiteBoxAgent("PRO", "You are PRO in a debate. You ALWAYS argue in "
                        "FAVOR of the proposition. Be concise. Never concede.",
                        tok, model, device)
    con = WhiteBoxAgent("CON", "You are CON in a debate. You ALWAYS argue "
                        "AGAINST the proposition. Be concise. Never concede.",
                        tok, model, device)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    msg_path = os.path.join(LOG_DIR, f"wb-debate-{run_id}.jsonl")
    grads, labels = [], []
    opponent_text = ""
    speaker = pro

    with open(msg_path, "w") as log:
        for turn in range(N_TURNS):
            text, prompt_ids, gen_ids = speaker.speak(
                frame_incoming(turn, topic, opponent_text))
            g, nll = per_example_param_grad(
                model, prompt_ids, gen_ids, device, name_filter=ATTR_SUBSET)
            g = g.cpu()                       # free GPU before next turn
            grads.append(g)
            labels.append(f"turn{turn}_{speaker.name}")

            log.write(json.dumps({
                "turn": turn, "agent": speaker.name, "text": text,
                "n_gen_tokens": int(gen_ids.numel()), "nll": nll,
                "grad_dim": int(g.numel()), "grad_norm": float(g.norm()),
            }) + "\n")
            log.flush()
            print(f"[turn {turn}] {speaker.name} (nll={nll:.2f}, "
                  f"||g||={g.norm():.4f}):\n    {text}\n")

            opponent_text = text
            speaker = con if speaker is pro else pro

    dot, cos = tracin_matrices(grads)
    grad_norms = dot.diagonal().sqrt()
    torch.save({"labels": labels, "dot": dot, "cos": cos,
                "grad_norms": grad_norms},
               os.path.join(LOG_DIR, f"wb-debate-{run_id}-attribution.pt"))

    print("cosine similarity (direction only):")
    print(" " * 12 + "".join(f"{l:>14}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>12}" + "".join(f"{cos[i, j]:14.4f}" for j in range(len(labels))))
    print(f"\nmessages -> {msg_path}")
    print("(raw grads not saved; set SAVE_GRADS in your own fork if needed)")


if __name__ == "__main__":
    main()
