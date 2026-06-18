#!/usr/bin/env python3
"""
Synthetic negotiation attribution – no external dataset.
Computes LOO, perturbation, and exact Shapley for 100 synthetic negotiations.
All methods use the actual final price as the outcome.
"""
import os
import sys
import re
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

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

# --- Shared agent class ---
class SharedDialogueAgent:
    def __init__(self, name):
        self.name = name
        self.history = []
    def respond(self, message, max_new_tokens=100):
        self.history.append(("user", message))
        prompt = f"{self.name}: {message}\n{self.name}:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        self.history.append(("assistant", response))
        return response
    def reset(self):
        self.history = []

# --- Run a negotiation and return final price ---
def run_negotiation(buyer, seller, item="laptop", start_price=100,
                    fixed_buyer_response=None, fixed_seller_response=None):
    buyer.reset()
    seller.reset()
    opening = f"Item: {item}\nAsking price: ${start_price:.2f}. Make an offer."
    if fixed_buyer_response is not None:
        buyer_resp = fixed_buyer_response
    else:
        buyer_resp = buyer.respond(opening)
    if fixed_seller_response is not None:
        seller_resp = fixed_seller_response
    else:
        seller_resp = seller.respond(buyer_resp)
    match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", seller_resp)
    return float(match.group(1)) if match else 0.0

# --- Compute all attributions for one negotiation ---
def compute_attributions(buyer, seller):
    # Randomise item and price for each negotiation
    items = ["laptop", "phone", "bicycle", "watch", "headphones", "tablet", "camera", "speaker"]
    item = random.choice(items)
    start_price = random.randint(50, 500)

    # Original price
    orig_price = run_negotiation(buyer, seller, item, start_price)

    # ---- LOO ----
    # Ablate buyer
    orig_buyer_respond = buyer.respond
    buyer.respond = lambda _: "I accept your price."
    price_no_buyer = run_negotiation(buyer, seller, item, start_price)
    buyer.respond = orig_buyer_respond
    loo_buyer = orig_price - price_no_buyer

    # Ablate seller
    orig_seller_respond = seller.respond
    seller.respond = lambda _: ""
    price_no_seller = run_negotiation(buyer, seller, item, start_price)
    seller.respond = orig_seller_respond
    loo_seller = orig_price - price_no_seller

    # ---- Perturbation and Shapley ----
    # Get fresh responses from both agents
    buyer.reset()
    seller.reset()
    opening = f"Item: {item}\nAsking price: ${start_price:.2f}. Make an offer."
    buyer_resp = buyer.respond(opening)
    seller_resp = seller.respond(buyer_resp)

    # Outcome function that re‑runs the negotiation with fixed responses
    def outcome_fn(resp_pair):
        # resp_pair = [buyer_text, seller_text]
        # Re‑run the negotiation with these fixed utterances
        return run_negotiation(
            buyer, seller, item, start_price,
            fixed_buyer_response=resp_pair[0],
            fixed_seller_response=resp_pair[1]
        )

    pert_buyer = perturbation_attribution(
        buyer_resp, seller_resp, outcome_fn, agent_idx=0, baseline_value=""
    )
    pert_seller = perturbation_attribution(
        buyer_resp, seller_resp, outcome_fn, agent_idx=1, baseline_value=""
    )
    shapley_scores = exact_shapley_2_agents(
        [buyer_resp, seller_resp], outcome_fn, baseline=""
    )

    return {
        "LOO": (loo_buyer, loo_seller),
        "Perturbation": (pert_buyer, pert_seller),
        "Exact Shapley": (shapley_scores[0], shapley_scores[1])
    }

# --- Main loop ---
def main():
    n = 100
    loo_b, loo_s = [], []
    pert_b, pert_s = [], []
    shap_b, shap_s = [], []

    for i in range(n):
        buyer = SharedDialogueAgent("Buyer")
        seller = SharedDialogueAgent("Seller")
        attrs = compute_attributions(buyer, seller)
        loo_b.append(attrs["LOO"][0]); loo_s.append(attrs["LOO"][1])
        pert_b.append(attrs["Perturbation"][0]); pert_s.append(attrs["Perturbation"][1])
        shap_b.append(attrs["Exact Shapley"][0]); shap_s.append(attrs["Exact Shapley"][1])
        if (i+1) % 20 == 0:
            print(f"Processed {i+1} negotiations.")

    print("\n=== Negotiation Attribution Summary (100 negotiations) ===")
    print(f"Method            Buyer      Seller")
    print(f"LOO               {np.mean(loo_b):.4f}    {np.mean(loo_s):.4f}")
    print(f"Perturbation      {np.mean(pert_b):.4f}    {np.mean(pert_s):.4f}")
    print(f"Exact Shapley     {np.mean(shap_b):.4f}    {np.mean(shap_s):.4f}")

if __name__ == "__main__":
    main()
