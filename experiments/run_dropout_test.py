#!/usr/bin/env python3
import sys
import os
import json
import time
sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent
from src.environment.two_agent_dialogue import TwoAgentEnv
from src.metrics.stability import sensitivity_to_dropout

def main():
    model_name = "Qwen/Qwen-7B-Chat"  # or read from env
    agent_a = DialogueAgent("Alice", model_name=model_name)
    agent_b = DialogueAgent("Bob", model_name=model_name)
    env = TwoAgentEnv(agent_a, agent_b)
    def outcome_fn(env):
        env.step("Test prompt for dropout")
        _, ra, rb = env.get_last_exchange()
        return len(ra) + len(rb)
    sensitivity_a = sensitivity_to_dropout(env, agent_idx=0, outcome_fn=outcome_fn, dropout_rounds=3)
    sensitivity_b = sensitivity_to_dropout(env, agent_idx=1, outcome_fn=outcome_fn, dropout_rounds=3)
    print(f"Sensitivity to dropping Alice: {sensitivity_a:.2f}")
    print(f"Sensitivity to dropping Bob: {sensitivity_b:.2f}")
    # Log to file
    os.makedirs("results", exist_ok=True)
    log_file = f"results/dropout_{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump({
            "model": model_name,
            "sensitivity_Alice": sensitivity_a,
            "sensitivity_Bob": sensitivity_b
        }, f)
    print(f"Logged to {log_file}")

if __name__ == "__main__":
    main()
