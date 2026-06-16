#!/usr/bin/env python3
"""
GSM8K math reasoning – two agents: Solver and Verifier.
Computes LOO, perturbation, and exact Shapley attribution.
"""
import os
import sys
import numpy as np
from datasets import load_dataset

sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

def solve_and_verify(problem, solver, verifier):
    """Return 1 if correct, 0 otherwise."""
    solver_prompt = f"Solve this math problem step by step and output only the final number:\n{problem}"
    solver_answer = solver.respond(solver_prompt)
    verifier_prompt = (
        f"The problem was:\n{problem}\n"
        f"An agent gave this solution:\n{solver_answer}\n"
        f"Is the final answer correct? Answer only YES or NO."
    )
    verifier_answer = verifier.respond(verifier_prompt)
    return 1 if "yes" in verifier_answer.lower() else 0

def outcome_from_responses(responses, problem, solver, verifier):
    """
    Given two responses [solver_resp, verifier_resp], evaluate correctness.
    We need to re-run the verifier on the solver response.
    """
    solver_answer = responses[0]
    if solver_answer == "":
        # If solver is ablated, we assume empty answer -> incorrect
        return 0
    verifier_prompt = (
        f"The problem was:\n{problem}\n"
        f"An agent gave this solution:\n{solver_answer}\n"
        f"Is the final answer correct? Answer only YES or NO."
    )
    # We need a fresh verifier agent because the original might have history.
    # We'll create a new one inside to avoid contamination.
    temp_verifier = DialogueAgent("TempVerifier", model_name=verifier.model_name)
    verifier_answer = temp_verifier.respond(verifier_prompt)
    return 1 if "yes" in verifier_answer.lower() else 0

def compute_attributions(problem, solver, verifier):
    """
    Compute LOO, perturbation, and exact Shapley for one problem.
    Returns dict of attributions.
    """
    # Original correctness
    orig_correct = solve_and_verify(problem, solver, verifier)

    # LOO: ablate solver and verifier separately
    orig_respond_s = solver.respond
    solver.respond = lambda _: ""
    correct_no_solver = solve_and_verify(problem, solver, verifier)
    solver.respond = orig_respond_s

    orig_respond_v = verifier.respond
    verifier.respond = lambda _: "YES"
    correct_no_verifier = solve_and_verify(problem, solver, verifier)
    verifier.respond = orig_respond_v

    loo_solver = orig_correct - correct_no_solver
    loo_verifier = orig_correct - correct_no_verifier

    # Perturbation: replace response with empty string and re-evaluate
    # We need the actual responses first
    solver_resp = solver.respond(f"Solve: {problem}")
    verifier_resp = verifier.respond(f"Verify: {problem}\n{solver_resp}")
    # Define outcome function for the two responses
    def outcome_pair(responses):
        return outcome_from_responses(responses, problem, solver, verifier)
    pert_solver = perturbation_attribution(solver_resp, verifier_resp, outcome_pair, agent_idx=0, baseline_value="")
    pert_verifier = perturbation_attribution(solver_resp, verifier_resp, outcome_pair, agent_idx=1, baseline_value="")

    # Exact Shapley
    shapley_scores = exact_shapley_2_agents([solver_resp, verifier_resp], outcome_pair, baseline="")

    return {
        "LOO": (loo_solver, loo_verifier),
        "Perturbation": (pert_solver, pert_verifier),
        "Exact Shapley": (shapley_scores[0], shapley_scores[1])
    }

def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    dataset = load_dataset("gsm8k", "main", split="train")
    indices = list(range(100))  # 100 problems
    problems = [dataset[i]["question"] for i in indices]

    all_loo_s = []; all_loo_v = []
    all_pert_s = []; all_pert_v = []
    all_shap_s = []; all_shap_v = []

    for idx, prob in enumerate(problems):
        solver = DialogueAgent("Solver", model_name=model_name)
        verifier = DialogueAgent("Verifier", model_name=model_name)
        attrs = compute_attributions(prob, solver, verifier)
        all_loo_s.append(attrs["LOO"][0]); all_loo_v.append(attrs["LOO"][1])
        all_pert_s.append(attrs["Perturbation"][0]); all_pert_v.append(attrs["Perturbation"][1])
        all_shap_s.append(attrs["Exact Shapley"][0]); all_shap_v.append(attrs["Exact Shapley"][1])
        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1} problems.")

    print("\n=== GSM8K Attribution Summary (100 problems) ===")
    print(f"Method            Solver     Verifier")
    print(f"LOO               {np.mean(all_loo_s):.4f}    {np.mean(all_loo_v):.4f}")
    print(f"Perturbation      {np.mean(all_pert_s):.4f}    {np.mean(all_pert_v):.4f}")
    print(f"Exact Shapley     {np.mean(all_shap_s):.4f}    {np.mean(all_shap_v):.4f}")

if __name__ == "__main__":
    main()
