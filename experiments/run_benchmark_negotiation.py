#!/usr/bin/env python3
"""
Craigslist Bargaining – negotiation attribution.
Re‑runs the negotiation after ablating each agent.
Requires `datasets` package.
"""

import os
import sys
import re
import numpy as np
from datasets import load_dataset

sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent

def negotiate(buyer, seller, item_title, item_desc, start_price):
    """Run a single negotiation exchange, return final price (0 if no deal)."""
    buyer.reset()
    seller.reset()
    opening = f"Item: {item_title}\n{item_desc}\nAsking price: ${start_price:.2f}. Make an offer."
    buyer_response = buyer.respond(opening)
    seller_response = seller.respond(buyer_response)
    # Extract price from seller's response
    match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", seller_response)
    if match:
        return float(match.group(1))
    return 0.0

def leave_one_out_negotiation(buyer, seller, item_title, item_desc, start_price):
    """Compute LOO attribution using final price as outcome."""
    original_price = negotiate(buyer, seller, item_title, item_desc, start_price)
    # Ablate buyer
    orig_buyer_respond = buyer.respond
    buyer.respond = lambda _: "I accept your price."
    price_no_buyer = negotiate(buyer, seller, item_title, item_desc, start_price)
    buyer.respond = orig_buyer_respond
    # Ablate seller
    orig_seller_respond = seller.respond
    seller.respond = lambda _: ""
    price_no_seller = negotiate(buyer, seller, item_title, item_desc, start_price)
    seller.respond = orig_seller_respond
    return original_price - price_no_buyer, original_price - price_no_seller

def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    dataset = load_dataset("craigslist_bargaining", split="train")
    # Use a few examples
    examples = [dataset[i] for i in range(3)]
    buyer = DialogueAgent("Buyer", model_name=model_name)
    seller = DialogueAgent("Seller", model_name=model_name)

    for ex in examples:
        title = ex.get("title", "item")
        desc = ex.get("description", "")
        price = float(ex.get("price", 100.0))
        attr_buyer, attr_seller = leave_one_out_negotiation(buyer, seller, title, desc, price)
        print(f"{title}: attr(buyer)={attr_buyer:.2f}, attr(seller)={attr_seller:.2f}")

if __name__ == "__main__":
    main()
