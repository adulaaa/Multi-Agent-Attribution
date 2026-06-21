#!/usr/bin/env python3
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(".")
from src.agents.base_agent import BaseAgent
from src.environment.multi_agent_env import MultiAgentEnv
from src.attribution.removal_based import leave_one_out_attribution
from src.attribution.shapley_approx import shapley_approximation

# Shared model class (loads once)
class SharedDialogueAgent(BaseAgent):
    def __init__(self, name, tokenizer, model, device):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
    def respond(self, message, max_new_tokens=100):
        self.conversation_history.append(("user", message))
        prompt = f"{self.name}: {message}\n{self.name}:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        self.conversation_history.append(("assistant", response))
        return response
    def reset(self):
        super().reset()

def main():
    model_name = "microsoft/Phi-3.5-mini-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model ONCE
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    # Create 3 agents sharing the same model
    agents = [
        SharedDialogueAgent(f"Agent{i}", tokenizer, model, device)
        for i in range(3)
    ]
    
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
