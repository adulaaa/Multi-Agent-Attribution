#!/usr/bin/env python3
"""
Synthetic negotiation attribution with stats – no external dataset.
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
from scipy import stats

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

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
    if "assistant" in full:
        response = full.split("assistant")[-1].strip()
    else:
        response = full[len(input_text):].strip()
    return response

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
    match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", seller_resp)
    if match:
        return float(match.group(1))
    else:
        return start_price

BASELINE = "I don't know."

def make_outcome_fn(buyer, seller, item, start_price):
    def outcome_fn(resp_pair):
        return run_negotiation(
            buyer, seller, item, start_price,
            fixed_buyer_response=resp_pair[0],
            fixed_seller_response=resp_pair[1]
        )
    return outcome_fn

def leave_one_out_attribution(resp_pair, outcome_fn):
    full = outcome_fn(resp_pair)
    loo_b = full - outcome_fn([BASELINE, resp_pair[1]])
    loo_s = full - outcome_fn([resp_pair[0], BASELINE])
    return loo_b, loo_s

def perturbation_attribution(resp_pair, outcome_fn, agent_idx):
    full = outcome_fn(resp_pair)
    if agent_idx == 0:
        alt = outcome_fn([BASELINE, resp_pair[1]])
    else:
        alt = outcome_fn([resp_pair[0], BASELINE])
    return full - alt

def exact_shapley_2_agents(resp_pair, outcome_fn):
    b, s = resp_pair[0], resp_pair[1]
    v_empty = outcome_fn([BASELINE, BASELINE])
    v_b = outcome_fn([b, BASELINE])
    v_s = outcome_fn([BASELINE, s])
    v_bs = outcome_fn([b, s])
    shap_b = 0.5*(v_b - v_empty) + 0.5*(v_bs - v_s)
    shap_s = 0.5*(v_s - v_empty) + 0.5*(v_bs - v_b)
    return shap_b, shap_s

def compute_ci(data):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=std_err)
    return mean, std_err, ci

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

        buyer.reset(); seller.reset()
        opening = f"Item: {item}\nAsking price: ${start_price:.2f}. Make an offer."
        buyer_resp = buyer.respond(opening)
        seller_resp = seller.respond(buyer_resp)
        resp_pair = [buyer_resp, seller_resp]

        outcome_fn = make_outcome_fn(buyer, seller, item, start_price)

        loo_b, loo_s = leave_one_out_attribution(resp_pair, outcome_fn)
        pert_b = perturbation_attribution(resp_pair, outcome_fn, agent_idx=0)
        pert_s = perturbation_attribution(resp_pair, outcome_fn, agent_idx=1)
        shap_b, shap_s = exact_shapley_2_agents(resp_pair, outcome_fn)

        loo_b_list.append(loo_b); loo_s_list.append(loo_s)
        pert_b_list.append(pert_b); pert_s_list.append(pert_s)
        shap_b_list.append(shap_b); shap_s_list.append(shap_s)

        if (i+1) % 20 == 0:
            print(f"Processed {i+1} negotiations.")

    methods = {
        "LOO": (loo_b_list, loo_s_list),
        "Perturbation": (pert_b_list, pert_s_list),
        "Exact Shapley": (shap_b_list, shap_s_list)
    }

    print("\n=== Negotiation Attribution Summary (100 negotiations) ===")
    print(f"{'Method':<15} {'Buyer mean':>12} {'Buyer CI':>20} {'Seller mean':>12} {'Seller CI':>20}")
    for name, (b_list, s_list) in methods.items():
        b_mean, b_se, b_ci = compute_ci(b_list)
        s_mean, s_se, s_ci = compute_ci(s_list)
        print(f"{name:<15} {b_mean:12.4f} [{b_ci[0]:.4f}, {b_ci[1]:.4f}] {s_mean:12.4f} [{s_ci[0]:.4f}, {s_ci[1]:.4f}]")
        
if __name__ == "__main__":
    main()
