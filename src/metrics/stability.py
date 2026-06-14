import numpy as np
from typing import Callable, List, Any

def attribution_stability(attribution_fn: Callable, inputs: List[Any],
                          noise_level: float = 0.1, n_samples: int = 10) -> float:
    """
    Measure stability of attribution under input noise (reworded prompts).
    Returns a score between 0 and 1 (higher = more stable).
    """
    base_attr = np.array(attribution_fn(*inputs))
    perturbations = []
    for _ in range(n_samples):
        # Apply simple noise: add random character to first input string if it is a string
        noisy_inputs = list(inputs)
        if isinstance(noisy_inputs[0], str):
            s = noisy_inputs[0]
            idx = np.random.randint(0, max(1, len(s)))
            noisy_inputs[0] = s[:idx] + np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) + s[idx:]
        pert_attr = np.array(attribution_fn(*noisy_inputs))
        perturbations.append(pert_attr)
    perturbations = np.array(perturbations)
    variance = np.var(perturbations, axis=0).mean()
    max_var = np.var(base_attr) if np.var(base_attr) > 0 else 1.0
    return 1.0 - min(1.0, variance / max_var)

def sensitivity_to_dropout(env, agent_idx: int, outcome_fn: Callable,
                           dropout_rounds: int = 5) -> float:
    """
    Sensitivity of outcome to agent dropout. Re‑runs the environment after
    replacing the agent's respond method with a no‑op.
    """
    changes = []
    for _ in range(dropout_rounds):
        original_outcome = outcome_fn(env)
        original_respond = env.agents[agent_idx].respond
        env.agents[agent_idx].respond = lambda _: ""
        new_outcome = outcome_fn(env)
        changes.append(abs(original_outcome - new_outcome))
        env.agents[agent_idx].respond = original_respond
    return np.mean(changes)
