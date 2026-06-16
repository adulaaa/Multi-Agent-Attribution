#!/usr/bin/env python3
"""
Craigslist Bargaining – negotiation attribution.
Re‑runs the negotiation after ablating each agent.
Requires `datasets` package.
Computes LOO, perturbation, and exact Shapley for 100 negotiations.
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


def run_negotiation(buyer, seller, item_title, item_desc, start_price, 
                    fixed_buyer_response=None, fixed_seller_response=None):
    """
    Run a negotiation. If fixed_buyer_response is given, use that as the buyer's
    first response instead of generating it. Similarly for seller.
    Returns the final price (0 if no deal).
    """
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
    """
    Compute LOO, perturbation, and exact Shapley for one negotiation.
    """
    # --- Original outcome ---
    orig_price = run_negotiation(buyer, seller, item_title, item_desc, start_price)

    # --- LOO ---
    # LOO for buyer: buyer accepts immediately
    orig_buyer_respond = buyer.respond
    buyer.respond = lambda _: "I accept your price."
    price_no_buyer = run_negotiation(buyer, seller, item_title, item_desc, start_price)
    buyer.respond = orig_buyer_respond

    # LOO for seller: seller gives no response (deal fails)
    orig_seller_respond = seller.respond
    seller.respond = lambda _: ""
    price_no_seller = run_negotiation(buyer, seller, item_title, item_desc, start_price)
    seller.respond = orig_seller_respond

    loo_buyer = orig_price - price_no_buyer
    loo_seller = orig_price - price_no_seller

    # --- For perturbation and Shapley, reset both agents and get fresh responses
    buyer.reset()
    seller.reset()
    opening = f"Item: {item_title}\n{item_desc}\nAsking price: ${start_price:.2f}. Make an offer."
    buyer_resp = buyer.respond(opening)          # Buyer's actual offer
    seller_resp = seller.respond(buyer_resp)     # Seller's actual response

    # Outcome function that takes [buyer_resp, seller_resp] and returns final price
    def outcome_fn(responses):
        # Re‑run the negotiation with these fixed responses
        return run_negotiation(
            buyer, seller, item_title, item_desc, start_price,
            fixed_buyer_response=responses[0],
            fixed_seller_response=responses[1]
        )

    # --- Perturbation ---
    pert_buyer = perturbation_attribution(
        buyer_resp, seller_resp, outcome_fn, agent_idx=0, baseline_value=""
    )
    pert_seller = perturbation_attribution(
        buyer_resp, seller_resp, outcome_fn, agent_idx=1, baseline_value=""
    )

    # --- Exact Shapley ---
    shapley_scores = exact_shapley_2_agents(
        [buyer_resp, seller_resp], outcome_fn, baseline=""
    )

    return {
        "LOO": (loo_buyer, loo_seller),
        "Perturbation": (pert_buyer, pert_seller),
        "Exact Shapley": (shapley_scores[0], shapley_scores[1])
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
