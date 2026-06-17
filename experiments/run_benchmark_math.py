#!/usr/bin/env python3
"""
GSM8K math reasoning – two agents: Solver and Verifier.
Computes LOO, perturbation, and exact Shapley attribution.
Uses a shared model to avoid OOM.
"""
import os
import sys
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

# --- Helper functions ---
def get_solver_answer(problem, solver):
    prompt = f"Solve this math problem step by step and output only the final number:\n{problem}"
    return solver.respond(prompt)

def get_verifier_text(problem, solver_answer, verifier):
    prompt = (
        f"The problem was:\n{problem}\n"
        f"An agent gave this solution:\n{solver_answer}\n"
        f"Is the final answer correct? Answer only YES or NO."
    )
    return verifier.respond(prompt)

def outcome_from_responses(responses):
    # Directly use the Verifier's text
    if not responses[1]:
        return 0
    return 1 if "yes" in responses[1].lower() else 0

def compute_attributions(problem, solver, verifier):
    # Original outcome
    solver_resp = get_solver_answer(problem, solver)
    verifier_resp = get_verifier_text(problem, solver_resp, verifier)
    original_correct = outcome_from_responses([solver_resp, verifier_resp])

    # LOO for solver
    solver.reset()
    verifier.reset()
    orig_s = solver.respond
    solver.respond = lambda _: ""
    solver_empty = get_solver_answer(problem, solver)
    verifier_empty = get_verifier_text(problem, solver_empty, verifier)
    correct_no_solver = outcome_from_responses([solver_empty, verifier_empty])
    solver.respond = orig_s
    loo_solver = original_correct - correct_no_solver

    # LOO for verifier
    solver.reset()
    verifier.reset()
    orig_v = verifier.respond
    verifier.respond = lambda _: ""
    solver_resp2 = get_solver_answer(problem, solver)
    verifier_empty2 = get_verifier_text(problem, solver_resp2, verifier)
    correct_no_verifier = outcome_from_responses([solver_resp2, verifier_empty2])
    verifier.respond = orig_v
    loo_verifier = original_correct - correct_no_verifier

    # Perturbation and Shapley: use the actual recorded responses
    solver.reset()
    verifier.reset()
    solver_resp_clean = get_solver_answer(problem, solver)
    verifier_resp_clean = get_verifier_text(problem, solver_resp_clean, verifier)

    # Outcome function uses the Verifier's actual text (no fresh verifier)
    def outcome_fn(resp_pair):
        return outcome_from_responses(resp_pair)

    pert_solver = perturbation_attribution(
        solver_resp_clean, verifier_resp_clean, outcome_fn, agent_idx=0, baseline_value=""
    )
    pert_verifier = perturbation_attribution(
        solver_resp_clean, verifier_resp_clean, outcome_fn, agent_idx=1, baseline_value=""
    )
    shapley_scores = exact_shapley_2_agents(
        [solver_resp_clean, verifier_resp_clean], outcome_fn, baseline=""
    )

    return {
        "LOO": (loo_solver, loo_verifier),
        "Perturbation": (pert_solver, pert_verifier),
        "Exact Shapley": (shapley_scores[0], shapley_scores[1])
    }

def main():
    dataset = load_dataset("gsm8k", "main", split="train")
    indices = list(range(100))
    problems = [dataset[i]["question"] for i in indices]

    loo_s, loo_v = [], []
    pert_s, pert_v = [], []
    shap_s, shap_v = [], []

    for idx, prob in enumerate(problems):
        solver = SharedDialogueAgent("Solver")
        verifier = SharedDialogueAgent("Verifier")
        attrs = compute_attributions(prob, solver, verifier)
        loo_s.append(attrs["LOO"][0]); loo_v.append(attrs["LOO"][1])
        pert_s.append(attrs["Perturbation"][0]); pert_v.append(attrs["Perturbation"][1])
        shap_s.append(attrs["Exact Shapley"][0]); shap_v.append(attrs["Exact Shapley"][1])
        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1} problems.")

    print("\n=== GSM8K Attribution Summary (100 problems) ===")
    print(f"Method            Solver     Verifier")
    print(f"LOO               {np.mean(loo_s):.4f}    {np.mean(loo_v):.4f}")
    print(f"Perturbation      {np.mean(pert_s):.4f}    {np.mean(pert_v):.4f}")
    print(f"Exact Shapley     {np.mean(shap_s):.4f}    {np.mean(shap_v):.4f}")

if __name__ == "__main__":
    main()

