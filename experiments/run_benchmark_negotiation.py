#!/usr/bin/env python3
"""
Craigslist Bargaining – negotiation attribution.
Uses a shared model to avoid OOM.
Computes LOO, perturbation, and exact Shapley for 100 negotiations.
"""
import os
import sys
import re
import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from src.environment.two_agent_dialogue import TwoAgentEnv
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
        self.conversation_history = []
    def respond(self, message, max_new_tokens=100):
        self.conversation_history.append(("user", message))
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
        self.conversation_history.append(("assistant", response))
        return response
    def reset(self):
        self.conversation_history = []

def run_negotiation(buyer, seller, item_title, item_desc, start_price, 
                    fixed_buyer_response=None, fixed_seller_response=None):
    buyer.reset()
    seller.reset()
    opening = f"Item: {item_title}\n{item_desc}\nAsking price: ${start_price:.2f}. Make an offer."
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

def compute_attributions(item_title, item_desc, start_price, buyer, seller):
    orig_price = run_negotiation(buyer, seller, item_title, item_desc, start_price)

    # LOO
    orig_buyer_respond = buyer.respond
    buyer.respond = lambda _: "I accept your price."
    price_no_buyer = run_negotiation(buyer, seller, item_title, item_desc, start_price)
    buyer.respond = orig_buyer_respond

    orig_seller_respond = seller.respond
    seller.respond = lambda _: ""
    price_no_seller = run_negotiation(buyer, seller, item_title, item_desc, start_price)
    seller.respond = orig_seller_respond

    loo_buyer = orig_price - price_no_buyer
    loo_seller = orig_price - price_no_seller

    # Perturbation and Shapley
    buyer.reset()
    seller.reset()
    opening = f"Item: {item_title}\n{item_desc}\nAsking price: ${start_price:.2f}. Make an offer."
    buyer_resp = buyer.respond(opening)
    seller_resp = seller.respond(buyer_resp)

    def outcome_fn(responses):
        return run_negotiation(
            buyer, seller, item_title, item_desc, start_price,
            fixed_buyer_response=responses[0],
            fixed_seller_response=responses[1]
        )

    pert_buyer = perturbation_attribution(buyer_resp, seller_resp, outcome_fn, agent_idx=0, baseline_value="")
    pert_seller = perturbation_attribution(buyer_resp, seller_resp, outcome_fn, agent_idx=1, baseline_value="")
    shapley_scores = exact_shapley_2_agents([buyer_resp, seller_resp], outcome_fn, baseline="")

    return {
        "LOO": (loo_buyer, loo_seller),
        "Perturbation": (pert_buyer, pert_seller),
        "Exact Shapley": (shapley_scores[0], shapley_scores[1])
    }

def main():
    dataset = load_dataset("craigslist_bargaining", split="train")
    examples = [dataset[i] for i in range(100)]

    all_loo_b, all_loo_s = [], []
    all_pert_b, all_pert_s = [], []
    all_shap_b, all_shap_s = [], []

    for idx, ex in enumerate(examples):
        title = ex.get("title", "item")
        desc = ex.get("description", "")
        price = float(ex.get("price", 100.0))
        buyer = SharedDialogueAgent("Buyer")
        seller = SharedDialogueAgent("Seller")
        attrs = compute_attributions(title, desc, price, buyer, seller)
        all_loo_b.append(attrs["LOO"][0]); all_loo_s.append(attrs["LOO"][1])
        all_pert_b.append(attrs["Perturbation"][0]); all_pert_s.append(attrs["Perturbation"][1])
        all_shap_b.append(attrs["Exact Shapley"][0]); all_shap_s.append(attrs["Exact Shapley"][1])
        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1} negotiations.")

    print("\n=== Craigslist Attribution Summary (100 negotiations) ===")
    print(f"Method            Buyer      Seller")
    print(f"LOO               {np.mean(all_loo_b):.4f}    {np.mean(all_loo_s):.4f}")
    print(f"Perturbation      {np.mean(all_pert_b):.4f}    {np.mean(all_pert_s):.4f}")
    print(f"Exact Shapley     {np.mean(all_shap_b):.4f}    {np.mean(all_shap_s):.4f}")

if __name__ == "__main__":
    main()
