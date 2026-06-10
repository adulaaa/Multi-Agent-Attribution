"""White-box dialogue agent.

Unlike a string-only agent, this SHARES one model across agents and exposes the
tensors gradient-based attribution needs (logits, hidden states, token ids).
It still implements BaseAgent.respond() so it remains compatible with the
existing black-box environment/attribution code; the white-box path is speak().
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_agent import BaseAgent

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_shared_model(model_name, dtype="float32", device=None):
    """Load tokenizer + model ONCE and share across agents (so two 8B agents
    don't cost 2x VRAM). Use dtype='bfloat16' for >=4B models on a single GPU."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=_DTYPES[dtype]).to(device).eval()
    return tok, model, device


class WhiteBoxAgent(BaseAgent):
    def __init__(self, name, system_prompt, tokenizer, model, device, gen_kwargs=None):
        super().__init__(name)
        self.system_prompt = system_prompt
        self.tok = tokenizer
        self.model = model
        self.device = device
        # Greedy + anti-repeat by default: deterministic => reproducible attribution.
        self.gen_kwargs = gen_kwargs or dict(
            max_new_tokens=160, do_sample=False,
            repetition_penalty=1.3, no_repeat_ngram_size=3)
        self.messages = [{"role": "system", "content": system_prompt}]

    def _render(self):
        return self.tok.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

    # --- BaseAgent (string-only) API: keeps the black-box pipeline working ---
    def respond(self, message):
        text, _, _ = self.speak(message)
        return text

    # --- White-box API: text + the tensors to attribute over ---
    @torch.no_grad()
    def speak(self, message):
        self.messages.append({"role": "user", "content": message})
        enc = self.tok(self._render(), return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, **self.gen_kwargs)
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": text})
        return text, enc["input_ids"].detach(), gen_ids.detach()

    def reset(self):
        super().reset()
        self.messages = [{"role": "system", "content": self.system_prompt}]