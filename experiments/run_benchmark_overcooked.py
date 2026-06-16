#!/usr/bin/env python3
"""
Overcooked‑AI integration – cooperative task attribution.
Loads the actual Overcooked environment, drives two LLM agents,
and recomputes episode reward after ablating each agent to get attribution.
Requires `overcooked-ai` package.
Computes LOO, perturbation, and exact Shapley for each episode.
"""


import os
import sys
import numpy as np
import torch
from datasets import load_dataset  

sys.path.append(".")
from src.agents.dialogue_agent import DialogueAgent
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

try:
    from overcooked_ai import OvercookedEnv
    from overcooked_ai import OvercookedGridworld
except ImportError:
    raise ImportError("Please install overcooked-ai: pip install overcooked-ai")

def create_overcooked_layout(layout_name="cramped_room"):
    return OvercookedGridworld.from_layout_name(layout_name)

class OvercookedWrapper:
    def __init__(self, layout):
        self.env = OvercookedEnv.from_gridworld(layout)
        self.step_count = 0
        self.max_steps = 400

    def reset(self):
        self.env.reset()
        self.step_count = 0
        return self._state_text()

    def step(self, action_a_str, action_b_str):
        action_a = self._parse_action(action_a_str)
        action_b = self._parse_action(action_b_str)
        state, reward, done, _ = self.env.step([action_a, action_b])
        self.step_count += 1
        done = done or self.step_count >= self.max_steps
        return self._state_text(), reward, done

    def _parse_action(self, text):
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
        return 0

    def _state_text(self):
        return str(self.env.state)

class OvercookedAgent(DialogueAgent):
    def respond(self, message):
        system = "You are playing Overcooked. Output one action: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT, or STAY."
        return super().respond(system + "\n" + message)

def run_episode(agent_a, agent_b, env_wrapper):
    state = env_wrapper.reset()
    total_reward = 0.0
    done = False
    while not done:
        action_a = agent_a.respond(state)
        action_b = agent_b.respond(state)
        state, reward, done = env_wrapper.step(action_a, action_b)
        total_reward += reward
    return total_reward

def outcome_from_responses(responses, env_wrapper, agent_a, agent_b):
    """
    Given two action strings [action_a, action_b], simulate the episode with these fixed actions
    (but actually we need to use them as the first actions and then continue normally).
    Simplified: we just return the total reward of the episode starting with those actions.
    """
    # We'll create a new environment and use the given actions as first step.
    env = OvercookedWrapper(env_wrapper.env.gridworld)  # recreate
    state = env.reset()
    # Use the provided actions
    state, reward, done = env.step(responses[0], responses[1])
    total = reward
    # Continue the episode with the original agents' responses
    agent_a_copy = OvercookedAgent("A", model_name=agent_a.model_name)
    agent_b_copy = OvercookedAgent("B", model_name=agent_b.model_name)
    while not done:
        action_a = agent_a_copy.respond(state)
        action_b = agent_b_copy.respond(state)
        state, reward, done = env.step(action_a, action_b)
        total += reward
    return total

def compute_attributions(env_wrapper, agent_a, agent_b):
    """
    Compute LOO, perturbation, and exact Shapley for one episode.
    """
    # Original reward
    orig_reward = run_episode(agent_a, agent_b, env_wrapper)

    # LOO
    orig_respond_a = agent_a.respond
    agent_a.respond = lambda _: "STAY"
    reward_no_a = run_episode(agent_a, agent_b, env_wrapper)
    agent_a.respond = orig_respond_a

    orig_respond_b = agent_b.respond
    agent_b.respond = lambda _: "STAY"
    reward_no_b = run_episode(agent_a, agent_b, env_wrapper)
    agent_b.respond = orig_respond_b

    loo_a = orig_reward - reward_no_a
    loo_b = orig_reward - reward_no_b

    # Perturbation and Shapley: we need the first actions from each agent
    state = env_wrapper.reset()
    action_a = agent_a.respond(state)
    action_b = agent_b.respond(state)
    def outcome_pair(responses):
        # For exact Shapley, we need to evaluate all subsets; we'll use the outcome_from_responses function
        return outcome_from_responses(responses, env_wrapper, agent_a, agent_b)

    pert_a = perturbation_attribution(action_a, action_b, outcome_pair, agent_idx=0, baseline_value="STAY")
    pert_b = perturbation_attribution(action_a, action_b, outcome_pair, agent_idx=1, baseline_value="STAY")
    shap_scores = exact_shapley_2_agents([action_a, action_b], outcome_pair, baseline="STAY")

    return {
        "LOO": (loo_a, loo_b),
        "Perturbation": (pert_a, pert_b),
        "Exact Shapley": (shap_scores[0], shap_scores[1])
    }

def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    layout = create_overcooked_layout("cramped_room")
    episodes = 50

    all_loo_a = []; all_loo_b = []
    all_pert_a = []; all_pert_b = []
    all_shap_a = []; all_shap_b = []

    for ep in range(episodes):
        env = OvercookedWrapper(layout)
        agent_a = OvercookedAgent("ChefA", model_name=model_name)
        agent_b = OvercookedAgent("ChefB", model_name=model_name)
        attrs = compute_attributions(env, agent_a, agent_b)
        all_loo_a.append(attrs["LOO"][0]); all_loo_b.append(attrs["LOO"][1])
        all_pert_a.append(attrs["Perturbation"][0]); all_pert_b.append(attrs["Perturbation"][1])
        all_shap_a.append(attrs["Exact Shapley"][0]); all_shap_b.append(attrs["Exact Shapley"][1])
        if (ep+1) % 10 == 0:
            print(f"Processed {ep+1} episodes.")

    print("\n=== Overcooked Attribution Summary (50 episodes) ===")
    print(f"Method            ChefA      ChefB")
    print(f"LOO               {np.mean(all_loo_a):.4f}    {np.mean(all_loo_b):.4f}")
    print(f"Perturbation      {np.mean(all_pert_a):.4f}    {np.mean(all_pert_b):.4f}")
    print(f"Exact Shapley     {np.mean(all_shap_a):.4f}    {np.mean(all_shap_b):.4f}")

if __name__ == "__main__":
    main()
