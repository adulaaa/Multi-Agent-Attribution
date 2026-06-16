#!/usr/bin/env python3
"""
Craigslist Bargaining – negotiation attribution.
Re‑runs the negotiation after ablating each agent.
Requires `datasets` package.
Computes LOO, perturbation, and exact Shapley.
"""


import os
import sys
import re
import numpy as np
from datasets import load_dataset

sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

def negotiate(buyer, seller, item_title, item_desc, start_price):
    """Run negotiation, return final price."""
    buyer.reset(); seller.reset()
    opening = f"Item: {item_title}\n{item_desc}\nAsking price: ${start_price:.2f}. Make an offer."
    buyer_resp = buyer.respond(opening)
    seller_resp = seller.respond(buyer_resp)
    match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", seller_resp)
    return float(match.group(1)) if match else 0.0

def outcome_from_responses(responses, item_title, item_desc, start_price):
    """
    Given [buyer_resp, seller_resp], simulate negotiation with these as the only messages.
    """
    # We treat buyer_resp as the offer, seller_resp as final price.
    # For consistency, we need to re-run the negotiation with these exact responses.
    # Simplified: we assume seller_resp contains the final price.
    match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", responses[1])
    return float(match.group(1)) if match else 0.0

def compute_attributions(item_title, item_desc, start_price, buyer, seller):
    """
    Compute LOO, perturbation, and exact Shapley for one negotiation.
    """
    # Original price
    orig_price = negotiate(buyer, seller, item_title, item_desc, start_price)

    # LOO
    orig_buyer_respond = buyer.respond
    buyer.respond = lambda _: "I accept."
    price_no_buyer = negotiate(buyer, seller, item_title, item_desc, start_price)
    buyer.respond = orig_buyer_respond

    orig_seller_respond = seller.respond
    seller.respond = lambda _: ""
    price_no_seller = negotiate(buyer, seller, item_title, item_desc, start_price)
    seller.respond = orig_seller_respond

    loo_buyer = orig_price - price_no_buyer
    loo_seller = orig_price - price_no_seller

    # Perturbation and Shapley need the actual responses
    buyer_resp = buyer.respond(f"Make an offer on {item_title}")
    seller_resp = seller.respond(buyer_resp)
    def outcome_pair(responses):
        return outcome_from_responses(responses, item_title, item_desc, start_price)

    pert_buyer = perturbation_attribution(buyer_resp, seller_resp, outcome_pair, agent_idx=0, baseline_value="")
    pert_seller = perturbation_attribution(buyer_resp, seller_resp, outcome_pair, agent_idx=1, baseline_value="")
    shap_scores = exact_shapley_2_agents([buyer_resp, seller_resp], outcome_pair, baseline="")

    return {
        "LOO": (loo_buyer, loo_seller),
        "Perturbation": (pert_buyer, pert_seller),
        "Exact Shapley": (shap_scores[0], shap_scores[1])
    }

def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    dataset = load_dataset("craigslist_bargaining", split="train")
    examples = [dataset[i] for i in range(100)]

    all_loo_b = []; all_loo_s = []
    all_pert_b = []; all_pert_s = []
    all_shap_b = []; all_shap_s = []

    for idx, ex in enumerate(examples):
        title = ex.get("title", "item")
        desc = ex.get("description", "")
        price = float(ex.get("price", 100.0))
        buyer = DialogueAgent("Buyer", model_name=model_name)
        seller = DialogueAgent("Seller", model_name=model_name)
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
