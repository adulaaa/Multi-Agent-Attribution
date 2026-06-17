#!/usr/bin/env python3
import sys
sys.path.append(".")

from src.agents.dialogue_agent import DialogueAgent
from src.environment.two_agent_dialogue import TwoAgentEnv
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

def main():
    model_name = "microsoft/Phi-3.5-mini-instruct"
    agent_a = DialogueAgent("Alice", model_name=model_name)
    agent_b = DialogueAgent("Bob", model_name=model_name)
    env = TwoAgentEnv(agent_a, agent_b)

    prompt = "Should we invest more in renewable energy? Answer briefly."
    env.step(prompt)
    _, resp_a, resp_b = env.get_last_exchange()

    def length_outcome(env):
        _, ra, rb = env.get_last_exchange()
        return len(ra) + len(rb)

    # 1. Leave-One-Out
    scores_loo = leave_one_out_attribution([agent_a, agent_b], env, length_outcome)
    # 2. Perturbation
    attr_a_pert = perturbation_attribution(resp_a, resp_b, length_outcome, agent_idx=0, baseline_value="")
    attr_b_pert = perturbation_attribution(resp_a, resp_b, length_outcome, agent_idx=1, baseline_value="")
    # 3. Exact Shapley (two agents)
    shapley_scores = exact_shapley_2_agents([resp_a, resp_b], length_outcome, baseline="")

    print(f"\nBaseline attribution results:")
    print(f"Method            Alice      Bob")
    print(f"Leave-One-Out     {scores_loo[0]:.2f}      {scores_loo[1]:.2f}")
    print(f"Perturbation      {attr_a_pert:.2f}      {attr_b_pert:.2f}")
    print(f"Exact Shapley     {shapley_scores[0]:.2f}      {shapley_scores[1]:.2f}")

if __name__ == "__main__":
    main()
