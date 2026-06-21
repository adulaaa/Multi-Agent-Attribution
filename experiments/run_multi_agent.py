#!/usr/bin/env python3
import sys
sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent
from src.environment.multi_agent_env import MultiAgentEnv
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.shapley_approx import shapley_approximation

def main():
    model = "microsoft/Phi-3.5-mini-instruct"
    agents = [DialogueAgent(f"Agent{i}", model) for i in range(3)]
    env = MultiAgentEnv(agents)
    prompt = "What is 2+2? Answer briefly."
    env.step(prompt)
    def outcome_fn(env):
        return sum(len(r) for r in env.get_last_exchange()[1:])
    loo = leave_one_out_attribution(agents, env, outcome_fn)
    responses = env.get_last_exchange()[1:]
    shap = shapley_approximation(responses, lambda x: sum(len(r) for r in x), n_samples=50)
    print("LOO:", loo)
    print("Shapley:", shap)
if __name__ == "__main__":
    main()
