import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_agent import BaseAgent

class DialogueAgent(BaseAgent):
    def __init__(self, name: str, model_name: str = "Qwen/Qwen-7B-Chat", device: str = None):
        super().__init__(name)
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
    
    def respond(self, message: str, max_new_tokens: int = 100) -> str:
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
