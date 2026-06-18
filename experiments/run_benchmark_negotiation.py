#!/usr/bin/env python3
"""
Synthetic negotiation attribution – no external dataset.
Computes LOO, perturbation, and exact Shapley for 100 synthetic negotiations.
All methods use the same baseline (empty string) and the same sampled response pair.
All attributions are computed with respect to the same reference outcome.
"""
import os
import sys
import re
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Set seeds for reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Load model once ---
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/Phi-3.5-mini-instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True
).to(DEVICE)
model.eval()

# --- Helper: generate response from model with chat template ---
def generate_response(prompt, max_new_tokens=100):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant's response (after the last turn)
    if "assistant" in full:
        response = full.split("assistant")[-1].strip()
    else:
        response = full[len(input_text):].strip()
    return response

# --- Shared agent class ---
class SharedDialogueAgent:
    def __init__(self, name):
        self.name = name
        self.history = []
    def respond(self, message, max_new_tokens=100):
        self.history.append(("user", message))
        prompt = f"{self.name}: {message}\n{self.name}:"
        response = generate_response(prompt, max_new_tokens)
        self.history.append(("assistant", response))
        return response
    def reset(self):
        self.history = []

# --- Run a negotiation and return final price ---
# fixed utterances are used if provided; otherwise they are generated.
def run_negotiation(buyer, seller, item, start_price,
                    fixed_buyer_response=None, fixed_seller_response=None):
    buyer.reset()
    seller.reset()
    opening = f"Item: {item}\nAsking price: ${start_price:.2f}. Make an offer."
    if fixed_buyer_response is not None:
        buyer_resp = fixed_buyer_response
        buyer.history.append(("user", opening))
        buyer.history.append(("assistant", buyer_resp))
    else:
        buyer_resp = buyer.respond(opening)
    if fixed_seller_response is not None:
        seller_resp = fixed_seller_response
        seller.history.append(("user", buyer_resp))
        seller.history.append(("assistant", seller_resp))
    else:
        seller_resp = seller.respond(buyer_resp)
    # Extract price – fallback to start_price if not found
    match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", seller_resp)
    if match:
        return float(match.group(1))
    else:
        return start_price

# --- Outcome function for attribution (deterministic given pair) ---
def make_outcome_fn(buyer, seller, item, start_price):
    def outcome_fn(resp_pair):
        # resp_pair = [buyer_text, seller_text]
        return run_negotiation(
            buyer, seller, item, start_price,
            fixed_buyer_response=resp_pair[0],
            fixed_seller_response=resp_pair[1]
        )
    return outcome_fn

# --- Attribution methods – all use the SAME baseline (empty string) ---
BASELINE = ""   # common baseline for both agents

def leave_one_out_attribution(resp_pair, outcome_fn):
    # resp_pair = [buyer_text, seller_text]
    full = outcome_fn(resp_pair)
    # LOO buyer: replace buyer's utterance with baseline
    loo_b = full - outcome_fn([BASELINE, resp_pair[1]])
    # LOO seller: replace seller's utterance with baseline
    loo_s = full - outcome_fn([resp_pair[0], BASELINE])
    return loo_b, loo_s

def perturbation_attribution(resp_pair, outcome_fn, agent_idx):
    # agent_idx: 0 for buyer, 1 for seller
    full = outcome_fn(resp_pair)
    if agent_idx == 0:
        alt = outcome_fn([BASELINE, resp_pair[1]])
    else:
        alt = outcome_fn([resp_pair[0], BASELINE])
    return full - alt

def exact_shapley_2_agents(resp_pair, outcome_fn):
    b, s = resp_pair[0], resp_pair[1]
    # Empty coalition: both replaced by baseline
    v_empty = outcome_fn([BASELINE, BASELINE])
    v_b = outcome_fn([b, BASELINE])
    v_s = outcome_fn([BASELINE, s])
    v_bs = outcome_fn([b, s])
    shap_b = 0.5*(v_b - v_empty) + 0.5*(v_bs - v_s)
    shap_s = 0.5*(v_s - v_empty) + 0.5*(v_bs - v_b)
    return shap_b, shap_s

# --- Main loop ---
def main():
    n = 100
    loo_b_list, loo_s_list = [], []
    pert_b_list, pert_s_list = [], []
    shap_b_list, shap_s_list = [], []

    for i in range(n):
        buyer = SharedDialogueAgent("Buyer")
        seller = SharedDialogueAgent("Seller")
        items = ["laptop", "phone", "bicycle", "watch", "headphones", "tablet", "camera", "speaker"]
        item = random.choice(items)
        start_price = random.randint(50, 500)

        # Generate ONE representative pair of responses (both utterances)
        buyer.reset(); seller.reset()
        opening = f"Item: {item}\nAsking price: ${start_price:.2f}. Make an offer."
        buyer_resp = buyer.respond(opening)
        seller_resp = seller.respond(buyer_resp)
        resp_pair = [buyer_resp, seller_resp]

        # Deterministic outcome function
        outcome_fn = make_outcome_fn(buyer, seller, item, start_price)

        # ---- Compute all attributions using the SAME baseline and SAME pair ----
        loo_b, loo_s = leave_one_out_attribution(resp_pair, outcome_fn)
        pert_b = perturbation_attribution(resp_pair, outcome_fn, agent_idx=0)
        pert_s = perturbation_attribution(resp_pair, outcome_fn, agent_idx=1)
        shap_b, shap_s = exact_shapley_2_agents(resp_pair, outcome_fn)

        loo_b_list.append(loo_b); loo_s_list.append(loo_s)
        pert_b_list.append(pert_b); pert_s_list.append(pert_s)
        shap_b_list.append(shap_b); shap_s_list.append(shap_s)

        if (i+1) % 20 == 0:
            print(f"Processed {i+1} negotiations.")

    print("\n=== Negotiation Attribution Summary (100 negotiations) ===")
    print(f"Method            Buyer      Seller")
    print(f"LOO               {np.mean(loo_b_list):.4f}    {np.mean(loo_s_list):.4f}")
    print(f"Perturbation      {np.mean(pert_b_list):.4f}    {np.mean(pert_s_list):.4f}")
    print(f"Exact Shapley     {np.mean(shap_b_list):.4f}    {np.mean(shap_s_list):.4f}")

if __name__ == "__main__":
    main()
