#!/usr/bin/env python3
"""
GSM8K math reasoning with stats and smart baseline.
"""
import os
import sys
import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

sys.path.append(".")
from src.environment.two_agent_dialogue import TwoAgentEnv
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

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
    if not responses[1]:
        return 0
    return 1 if "yes" in responses[1].lower() else 0

BASELINE = "I don't know the answer."

def compute_attributions(problem, solver, verifier):
    solver_resp = get_solver_answer(problem, solver)
    verifier_resp = get_verifier_text(problem, solver_resp, verifier)
    original_correct = outcome_from_responses([solver_resp, verifier_resp])

    # LOO solver
    solver.reset()
    verifier.reset()
    orig_s = solver.respond
    solver.respond = lambda _: ""
    solver_empty = get_solver_answer(problem, solver)
    verifier_empty = get_verifier_text(problem, solver_empty, verifier)
    correct_no_solver = outcome_from_responses([solver_empty, verifier_empty])
    solver.respond = orig_s
    loo_solver = original_correct - correct_no_solver

    # LOO verifier
    solver.reset()
    verifier.reset()
    orig_v = verifier.respond
    verifier.respond = lambda _: ""
    solver_resp2 = get_solver_answer(problem, solver)
    verifier_empty2 = get_verifier_text(problem, solver_resp2, verifier)
    correct_no_verifier = outcome_from_responses([solver_resp2, verifier_empty2])
    verifier.respond = orig_v
    loo_verifier = original_correct - correct_no_verifier

    # Perturbation and Shapley with smart baseline
    solver.reset()
    verifier.reset()
    solver_resp_clean = get_solver_answer(problem, solver)
    verifier_resp_clean = get_verifier_text(problem, solver_resp_clean, verifier)

    def outcome_fn(resp_pair):
        return outcome_from_responses(resp_pair)

    pert_solver = perturbation_attribution(
        solver_resp_clean, verifier_resp_clean, outcome_fn, agent_idx=0, baseline_value=BASELINE
    )
    pert_verifier = perturbation_attribution(
        solver_resp_clean, verifier_resp_clean, outcome_fn, agent_idx=1, baseline_value=BASELINE
    )
    shapley_scores = exact_shapley_2_agents(
        [solver_resp_clean, verifier_resp_clean], outcome_fn, baseline=BASELINE
    )

    return {
        "LOO": (loo_solver, loo_verifier),
        "Perturbation": (pert_solver, pert_verifier),
        "Exact Shapley": (shapley_scores[0], shapley_scores[1])
    }

def compute_ci(data):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=std_err)
    return mean, std_err, ci

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

    methods = {
        "LOO": (loo_s, loo_v),
        "Perturbation": (pert_s, pert_v),
        "Exact Shapley": (shap_s, shap_v)
    }

    print("\n=== GSM8K Attribution Summary (100 problems) ===")
    print(f"{'Method':<15} {'Solver mean':>12} {'Solver CI':>20} {'Verifier mean':>12} {'Verifier CI':>20}")
    for name, (s_list, v_list) in methods.items():
        s_mean, s_se, s_ci = compute_ci(s_list)
        v_mean, v_se, v_ci = compute_ci(v_list)
        print(f"{name:<15} {s_mean:12.4f} [{s_ci[0]:.4f}, {s_ci[1]:.4f}] {v_mean:12.4f} [{v_ci[0]:.4f}, {v_ci[1]:.4f}]")

if __name__ == "__main__":
    main()

