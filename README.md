# Multi-Agent Attribution

Interpretable and stable attribution for multi-agent interactive systems.

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-attribution.git
cd multi-agent-attribution
pip install -r requirements.txt
```

## Quick Start
```python
from src.agents.dialogue_agent import DialogueAgent
from src.environment.two_agent_dialogue import TwoAgentEnv
from src.attribution.removal_based import leave_one_out_attribution

agent_a = DialogueAgent("Alice", model_name="Qwen/Qwen-7B-Chat")
agent_b = DialogueAgent("Bob", model_name="Qwen/Qwen-7B-Chat")
env = TwoAgentEnv(agent_a, agent_b)

prompt = "Should we increase AI safety funding?"
env.step(prompt)

def outcome(env):
    _, ra, rb = env.get_last_exchange()
    return len(ra) + len(rb)

scores = leave_one_out_attribution([agent_a, agent_b], env, outcome)
print(f"Attribution: Alice={scores[0]:.2f}, Bob={scores[1]:.2f}")
```

# Experiments

- python experiments/run_baseline.py – baseline attribution
- python experiments/run_dropout_test.py – stability under agent dropout

# License

MIT

