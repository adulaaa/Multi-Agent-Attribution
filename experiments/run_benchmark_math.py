#!/usr/bin/env python3
"""
GSM8K math reasoning – two agents: Solver and Verifier.
Attribution recomputes correctness after ablating each agent.
Requires `datasets` package.
"""

import os
import sys
import numpy as np
from datasets import load_dataset

sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent

def solve_and_verify(problem, agent_solver, agent_verifier):
    """Run solver, then verifier, return 1 if correct else 0."""
    solver_prompt = f"Solve this math problem step by step and output only the final number:\n{problem}"
    solver_answer = agent_solver.respond(solver_prompt)
    verifier_prompt = (
        f"The problem was:\n{problem}\n"
        f"An agent gave this solution:\n{solver_answer}\n"
        f"Is the final answer correct? Answer only YES or NO."
    )
    verifier_answer = agent_verifier.respond(verifier_prompt)
    return 1 if "yes" in verifier_answer.lower() else 0

def leave_one_out_gsm8k(problem, agent_solver, agent_verifier):
    """Compute LOO attribution for one problem."""
    original_outcome = solve_and_verify(problem, agent_solver, agent_verifier)
    # Ablate solver
    orig_respond_solver = agent_solver.respond
    agent_solver.respond = lambda _: ""
    outcome_no_solver = solve_and_verify(problem, agent_solver, agent_verifier)
    agent_solver.respond = orig_respond_solver
    # Ablate verifier (verifier always says YES when ablated, giving maximal chance)
    orig_respond_verifier = agent_verifier.respond
    agent_verifier.respond = lambda _: "YES"
    outcome_no_verifier = solve_and_verify(problem, agent_solver, agent_verifier)
    agent_verifier.respond = orig_respond_verifier
    return original_outcome - outcome_no_solver, original_outcome - outcome_no_verifier

def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    dataset = load_dataset("gsm8k", "main", split="train")
    # Use a small subset for demonstration (change as needed)
    indices = [0, 1, 2, 3, 4]
    problems = [dataset[i]["question"] for i in indices]
    agent_solver = DialogueAgent("Solver", model_name=model_name)
    agent_verifier = DialogueAgent("Verifier", model_name=model_name)

    attrs_solver, attrs_verifier = [], []
    for prob in problems:
        a, b = leave_one_out_gsm8k(prob, agent_solver, agent_verifier)
        attrs_solver.append(a)
        attrs_verifier.append(b)
        print(f"Problem: {prob[:60]}... -> attr(solver)={a:.3f}, attr(verifier)={b:.3f}")

    print(f"\nMean attribution (Solver): {np.mean(attrs_solver):.4f}")
    print(f"Mean attribution (Verifier): {np.mean(attrs_verifier):.4f}")

if __name__ == "__main__":
    main()
