#!/usr/bin/env python3
"""
Test attribution stability under input noise (reworded prompts).
Uses a shared model to avoid OOM.
"""
import os
import sys
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from src.environment.two_agent_dialogue import TwoAgentEnv
from src.attribution.perturbation import perturbation_attribution
from src.attribution.shapley_approx import exact_shapley_2_agents

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
ATTRIB_METHOD = os.environ.get("ATTRIB_METHOD", "perturbation")
N_NOISE = int(os.environ.get("N_NOISE", "50"))
SEEDS = int(os.environ.get("SEEDS", "10"))
PROMPT_ORIG = os.environ.get("PROMPT_ORIG", "Should we increase AI safety funding? Answer in one sentence.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model once globally ---
print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True
).to(DEVICE)
model.eval()

# --- Shared agent class (does NOT load its own model) ---
class SharedDialogueAgent:
    def __init__(self, name):
        self.name = name
        self.conversation_history = []
    def respond(self, message, max_new_tokens=100):
        self.conversation_history.append(("user", message))
        prompt = f"{self.name}: {message}\n{self.name}:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=0.7,
                                     pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        self.conversation_history.append(("assistant", response))
        return response
    def reset(self):
        self.conversation_history = []

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rephrase_prompt(prompt, rng):
    variants = [
        prompt.replace("?", "?").lower(),
        prompt.replace("Should", "Is it advisable to"),
        prompt.replace("increase", "boost"),
        prompt.replace("funding", "investment"),
        prompt.replace("Answer in one sentence.", "Respond with one sentence."),
        prompt.replace("?", "? " + rng.choice(["Please", "Kindly", ""])),
    ]
    if rng.random() < 0.3 and len(prompt) > 0:
        idx = rng.randint(0, len(prompt)-1)
        noisy = list(prompt)
        noisy[idx] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        variants.append("".join(noisy))
    return rng.choice(variants)

def run_attribution(agent_a, agent_b, env, prompt, method):
    env.reset()
    env.step(prompt)
    _, ra, rb = env.get_last_exchange()
    def length_outcome(responses):
        return len(responses[0]) + len(responses[1])
    if method == "perturbation":
        a = perturbation_attribution(ra, rb, length_outcome, agent_idx=0, baseline_value="")
        b = perturbation_attribution(ra, rb, length_outcome, agent_idx=1, baseline_value="")
        return np.array([a, b])
    elif method == "shapley":
        return exact_shapley_2_agents([ra, rb], length_outcome, baseline="")
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    print(f"Stability test: method={ATTRIB_METHOD}, N_NOISE={N_NOISE}, SEEDS={SEEDS}")
    all_stability_scores = []
    for seed in range(SEEDS):
        set_seed(seed)
        rng = random.Random(seed)
        agent_a = SharedDialogueAgent("Alice")
        agent_b = SharedDialogueAgent("Bob")
        env = TwoAgentEnv(agent_a, agent_b)
        orig_scores = run_attribution(agent_a, agent_b, env, PROMPT_ORIG, ATTRIB_METHOD)
        perturbed_scores = []
        for _ in range(N_NOISE):
            noisy_prompt = rephrase_prompt(PROMPT_ORIG, rng)
            agent_a2 = SharedDialogueAgent("Alice")
            agent_b2 = SharedDialogueAgent("Bob")
            env2 = TwoAgentEnv(agent_a2, agent_b2)
            scores = run_attribution(agent_a2, agent_b2, env2, noisy_prompt, ATTRIB_METHOD)
            perturbed_scores.append(scores)
        perturbed_scores = np.array(perturbed_scores)
        variance = np.var(perturbed_scores, axis=0).mean()
        max_var = np.var(orig_scores) if np.var(orig_scores) > 1e-6 else 1.0
        stability = 1.0 - min(1.0, variance / max_var)
        all_stability_scores.append(stability)
        print(f"Seed {seed}: orig={orig_scores}, stability={stability:.3f}")
    overall = np.mean(all_stability_scores)
    print(f"\nOverall stability score (higher = more stable): {overall:.4f}")

if __name__ == "__main__":
    main()
