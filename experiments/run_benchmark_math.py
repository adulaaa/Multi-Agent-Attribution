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
    """responses = [solver_text, verifier_text] -> 1 if verifier says YES, else 0."""
    if not responses[1]:
        return 0
    return 1 if "yes" in responses[1].lower() else 0


def compute_attributions(problem, solver, verifier):
    # --- Original outcome (clean run) ---
    solver_resp = get_solver_answer(problem, solver)
    verifier_resp = get_verifier_text(problem, solver_resp, verifier)
    original_correct = outcome_from_responses([solver_resp, verifier_resp])

    # --- LOO for solver ---
    solver.reset()
    verifier.reset()
    orig_s = solver.respond
    solver.respond = lambda _: ""
    solver_empty = get_solver_answer(problem, solver)
    verifier_empty = get_verifier_text(problem, solver_empty, verifier)
    correct_no_solver = outcome_from_responses([solver_empty, verifier_empty])
    solver.respond = orig_s
    loo_solver = original_correct - correct_no_solver

    # --- LOO for verifier ---
    solver.reset()
    verifier.reset()
    orig_v = verifier.respond
    verifier.respond = lambda _: "YES"
    solver_resp2 = get_solver_answer(problem, solver)
    verifier_yes = get_verifier_text(problem, solver_resp2, verifier)
    correct_no_verifier = outcome_from_responses([solver_resp2, verifier_yes])
    verifier.respond = orig_v
    loo_verifier = original_correct - correct_no_verifier

    # --- Fresh clean run for perturbation and Shapley ---
    solver.reset()
    verifier.reset()
    solver_resp_clean = get_solver_answer(problem, solver)
    verifier_resp_clean = get_verifier_text(problem, solver_resp_clean, verifier)

    def outcome_fn(resp_pair):
        # Re-evaluate correctness using the given solver and verifier texts.
        #Fresh verifier to avoid history contamination.
        temp_verifier = DialogueAgent("TempVerifier", model_name=verifier.model_name)
        temp_verifier.reset()
        prompt = (
            f"The problem was:\n{problem}\n"
            f"An agent gave this solution:\n{resp_pair[0]}\n"
            f"Is the final answer correct? Answer only YES or NO."
        )
        temp_response = temp_verifier.respond(prompt)
        return 1 if "yes" in temp_response.lower() else 0

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
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    dataset = load_dataset("gsm8k", "main", split="train")
    indices = list(range(100))
    problems = [dataset[i]["question"] for i in indices]

    loo_s = []; loo_v = []
    pert_s = []; pert_v = []
    shap_s = []; shap_v = []

    for idx, prob in enumerate(problems):
        solver = DialogueAgent("Solver", model_name=model_name)
        verifier = DialogueAgent("Verifier", model_name=model_name)
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
