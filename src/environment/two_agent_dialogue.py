from typing import Tuple, List, Optional
from src.agents.base_agent import BaseAgent

class TwoAgentEnv:
    def __init__(self, agent_a: BaseAgent, agent_b: BaseAgent):
        self.agents = [agent_a, agent_b]
        self.history: List[Tuple[str, str, str]] = []
    
    def step(self, prompt: str) -> str:
        response_a = self.agents[0].respond(prompt)
        response_b = self.agents[1].respond(response_a)
        self.history.append((prompt, response_a, response_b))
        return response_b
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.history.clear()
    
    def get_last_exchange(self) -> Optional[Tuple[str, str, str]]:
        return self.history[-1] if self.history else None
    
    def get_outcome(self, metric: str = "length") -> float:
        if not self.history:
            return 0.0
        _, resp_a, resp_b = self.history[-1]
        if metric == "length":
            return len(resp_a) + len(resp_b)
        elif metric == "agreement":
            return 1.0 if any(word in resp_b.lower() for word in ["agree", "yes", "correct"]) else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

