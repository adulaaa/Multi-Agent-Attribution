
#!/usr/bin/env python3
"""
GSM8K math reasoning – two agents: Solver and Verifier.
Computes LOO, perturbation, and exact Shapley using the verifier's judgment as outcome.
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
    solver_prompt = f"Solve this math problem step by step and output only the final number:\n{problem}"
    return solver.respond(solver_prompt)

def get_verifier_judgment(problem, solver_answer, verifier):
    verifier_prompt = (
        f"The problem was:\n{problem}\n"
        f"An agent gave this solution:\n{solver_answer}\n"
        f"Is the final answer correct? Answer only YES or NO."
    )
    verifier_answer = verifier.respond(verifier_prompt)
    return 1 if "yes" in verifier_answer.lower() else 0

def outcome_from_responses(responses):
    """
    responses = [solver_response, verifier_response]
    Outcome is the verifier's judgment (1 if YES, 0 otherwise).
    If verifier_response is empty, we treat it as NO (0).
    """
    verifier_response = responses[1]
    if verifier_response == "":
        return 0
    return 1 if "yes" in verifier_response.lower() else 0

def compute_attributions(problem, solver, verifier):
    # Original correctness: solver answer + verifier judgment
    solver_answer = get_solver_answer(problem, solver)
    original_correct = get_verifier_judgment(problem, solver_answer, verifier)

    # LOO: ablate solver or verifier and recompute correctness
    orig_solver_respond = solver.respond
    solver.respond = lambda _: ""
    solver_answer_empty = get_solver_answer(problem, solver)
    correct_no_solver = get_verifier_judgment(problem, solver_answer_empty, verifier)
    solver.respond = orig_solver_respond

    orig_verifier_respond = verifier.respond
    verifier.respond = lambda _: "YES"  
    correct_no_verifier = get_verifier_judgment(problem, solver_answer, verifier)
    verifier.respond = orig_verifier_respond

    loo_solver = original_correct - correct_no_solver
    loo_verifier = original_correct - correct_no_verifier

    # For perturbation and Shapley, we need actual responses from both agents
    solver_resp = get_solver_answer(problem, solver)
    verifier_resp = get_verifier_judgment(problem, solver_resp, verifier)  # returns 0/1
    verifier_prompt = (
        f"The problem was:\n{problem}\n"
        f"An agent gave this solution:\n{solver_resp}\n"
        f"Is the final answer correct? Answer only YES or NO."
    )
    verifier_text = verifier.respond(verifier_prompt)

    # Outcome function for perturbation and Shapley: uses both responses
    def outcome_fn(resp_pair):
        # resp_pair = [solver_text, verifier_text]
        return outcome_from_responses(resp_pair)

    pert_solver = perturbation_attribution(
        solver_resp, verifier_text, outcome_fn, agent_idx=0, baseline_value=""
    )
    pert_verifier = perturbation_attribution(
        solver_resp, verifier_text, outcome_fn, agent_idx=1, baseline_value=""
    )
    shapley_scores = exact_shapley_2_agents(
        [solver_resp, verifier_text], outcome_fn, baseline=""
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
