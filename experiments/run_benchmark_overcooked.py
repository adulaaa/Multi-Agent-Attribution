#!/usr/bin/env python3
"""
Overcooked‑AI integration – cooperative task attribution.
Loads the actual Overcooked environment, drives two LLM agents,
and recomputes episode reward after ablating each agent to get attribution.
Requires `overcooked-ai` package.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent

try:
    from overcooked_ai import OvercookedEnv
    from overcooked_ai import OvercookedGridworld
except ImportError:
    raise ImportError("Please install overcooked-ai: pip install overcooked-ai")

def create_overcooked_layout(layout_name="cramped_room"):
    """Create standard Overcooked gridworld."""
    return OvercookedGridworld.from_layout_name(layout_name)

class OvercookedWrapper:
    """Wrap Overcooked to expose a text‑based API for LLM agents."""
    def __init__(self, layout):
        self.env = OvercookedEnv.from_gridworld(layout)
        self.step_count = 0
        self.max_steps = 400

    def reset(self):
        self.env.reset()
        self.step_count = 0
        return self._state_text()

    def step(self, action_a_str, action_b_str):
        """Map agent strings to action indices and step environment."""
        action_a = self._parse_action(action_a_str)
        action_b = self._parse_action(action_b_str)
        state, reward, done, _ = self.env.step([action_a, action_b])
        self.step_count += 1
        done = done or self.step_count >= self.max_steps
        return self._state_text(), reward, done

    def _parse_action(self, text):
        """Convert natural language action to Overcooked action index."""
        text = text.lower()
        if "up" in text:
            return 1
        elif "down" in text:
            return 2
        elif "left" in text:
            return 3
        elif "right" in text:
            return 4
        elif "interact" in text:
            return 5
        return 0  # stay

    def _state_text(self):
        """Return human‑readable state description."""
        # Use environment's built‑in string representation
        return str(self.env.state)

class OvercookedAgent(DialogueAgent):
    """Agent that outputs Overcooked actions."""
    def respond(self, message):
        system = "You are playing Overcooked. Output one action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT, or STAY."
        return super().respond(system + "\n" + message)

def run_episode(agent_a, agent_b, env_wrapper):
    """Run a full episode and return total reward."""
    state = env_wrapper.reset()
    total_reward = 0.0
    done = False
    while not done:
        action_a = agent_a.respond(state)
        action_b = agent_b.respond(state)
        state, reward, done = env_wrapper.step(action_a, action_b)
        total_reward += reward
    return total_reward

def leave_one_out_attribution_overcooked(agent_a, agent_b, env_wrapper):
    """Compute LOO attribution by re‑running episodes."""
    original_reward = run_episode(agent_a, agent_b, env_wrapper)
    # Ablate agent A
    orig_respond_a = agent_a.respond
    agent_a.respond = lambda _: "STAY"
    reward_without_a = run_episode(agent_a, agent_b, env_wrapper)
    agent_a.respond = orig_respond_a
    # Ablate agent B
    orig_respond_b = agent_b.respond
    agent_b.respond = lambda _: "STAY"
    reward_without_b = run_episode(agent_a, agent_b, env_wrapper)
    agent_b.respond = orig_respond_b
    return original_reward - reward_without_a, original_reward - reward_without_b

def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    layout = create_overcooked_layout("cramped_room")
    all_attr_a, all_attr_b = [], []
    for ep in range(50):
        env = OvercookedWrapper(layout)
        agent_a = OvercookedAgent("ChefA", model_name=model_name)
        agent_b = OvercookedAgent("ChefB", model_name=model_name)
        attr_a, attr_b = leave_one_out_attribution_overcooked(agent_a, agent_b, env)
        all_attr_a.append(attr_a)
        all_attr_b.append(attr_b)
        print(f"Episode {ep+1}: ChefA={attr_a:.2f}, ChefB={attr_b:.2f}")
    print(f"\nMean attribution over 50 episodes: ChefA={np.mean(all_attr_a):.2f}, ChefB={np.mean(all_attr_b):.2f}")

if __name__ == "__main__":
    main()
