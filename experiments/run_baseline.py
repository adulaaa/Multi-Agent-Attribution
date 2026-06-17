#!/usr/bin/env python3
import sys
import os
sys.path.append(".")

from src.agents.dialogue_agent import DialogueAgent
from src.environment.two_agent_dialogue import TwoAgentEnv
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

def main():
    model_name = os.environ.get("MODEL_NAME", "microsoft/Phi-3.5-mini-instruct")
    agent_a = DialogueAgent("Alice", model_name=model_name)
    agent_b = DialogueAgent("Bob", model_name=model_name)
    env = TwoAgentEnv(agent_a, agent_b)

    prompt = "Should we invest more in renewable energy? Answer briefly."
    env.step(prompt)
    _, resp_a, resp_b = env.get_last_exchange()

    # Outcome function expects 'env' and returns total length
    def length_outcome(env):
        _, ra, rb = env.get_last_exchange()
        return len(ra) + len(rb)

    # LOO (uses length_outcome)
    scores_loo = leave_one_out_attribution([agent_a, agent_b], env, length_outcome)
    
    # Perturbation and Shapley use the pre-recorded responses with a simple length metric
    # that takes a list of strings.
    def length_pair(responses):
        return len(responses[0]) + len(responses[1])

    attr_a_pert = perturbation_attribution(resp_a, resp_b, length_pair, agent_idx=0, baseline_value="")
    attr_b_pert = perturbation_attribution(resp_a, resp_b, length_pair, agent_idx=1, baseline_value="")
    shapley_scores = exact_shapley_2_agents([resp_a, resp_b], length_pair, baseline="")

    print(f"\nBaseline attribution results:")
    print(f"Method            Alice      Bob")
    print(f"Leave-One-Out     {scores_loo[0]:.2f}      {scores_loo[1]:.2f}")
    print(f"Perturbation      {attr_a_pert:.2f}      {attr_b_pert:.2f}")
    print(f"Exact Shapley     {shapley_scores[0]:.2f}      {shapley_scores[1]:.2f}")

if __name__ == "__main__":
    main()
