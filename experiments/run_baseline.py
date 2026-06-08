#!/usr/bin/env python3
import sys
sys.path.append(".")

from src.agents.dialogue_agent import DialogueAgent
from src.environment.two_agent_dialogue import TwoAgentEnv
from src.attribution.removal_based import leave_one_out_attribution

def main():
    model_name = "Qwen/Qwen-7B-Chat"
    agent_a = DialogueAgent("Alice", model_name=model_name)
    agent_b = DialogueAgent("Bob", model_name=model_name)
    env = TwoAgentEnv(agent_a, agent_b)
    
    prompt = "Should we invest more in renewable energy? Answer briefly."
    env.step(prompt)
    
    def outcome(env):
        _, ra, rb = env.get_last_exchange()
        return len(ra) + len(rb)
    
    scores = leave_one_out_attribution([agent_a, agent_b], env, outcome)
    print(f"Attribution: Alice={scores[0]:.2f}, Bob={scores[1]:.2f}")

if __name__ == "__main__":
    main()
