from typing import List, Optional
from src.agents.base_agent import BaseAgent

class MultiAgentEnv:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.history: List[List[str]] = []
        self.last_prompt = ""

    def step(self, prompt: str) -> str:
        self.last_prompt = prompt
        messages = [prompt]
        for agent in self.agents:
            response = agent.respond(messages[-1])
            messages.append(response)
        self.history.append(messages)
        return messages[-1]

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.history.clear()
        self.last_prompt = ""

    def get_last_exchange(self) -> Optional[List[str]]:
        return self.history[-1] if self.history else None
