from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.conversation_history = []
    
    @abstractmethod
    def respond(self, message: str) -> str:
        pass
    
    def reset(self):
        self.conversation_history = []
