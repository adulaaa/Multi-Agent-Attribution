import numpy as np
from typing import Callable, List

def leave_one_out_attribution(agents: List, environment, metric_fn: Callable, baseline="") -> np.ndarray:
    """
    LOO attribution: re‑runs the dialogue after ablating each agent.
    Requires the environment to have a `last_prompt` attribute (set in `step`).
    """
    original_responds = [agent.respond for agent in agents]
    n = len(agents)
    scores = np.zeros(n)
    # Original outcome
    original_outcome = metric_fn(environment)
    for i in range(n):
        # Ablate agent i
        agents[i].respond = lambda _: baseline
        # Re‑run from saved prompt
        prompt = getattr(environment, 'last_prompt', None)
        if prompt is None:
            raise ValueError("Environment missing 'last_prompt'. Add `self.last_prompt = prompt` in environment.step().")
        environment.reset()
        environment.step(prompt)
        perturbed_outcome = metric_fn(environment)
        scores[i] = original_outcome - perturbed_outcome
        # Restore original method
        agents[i].respond = original_responds[i]
        # Reset back to original state (re‑run with all agents)
        environment.reset()
        environment.step(prompt)
    # Restore all (already done in loop)
    for i, orig in enumerate(original_responds):
        agents[i].respond = orig
    return scores

def removal_protocol_attribution(agents: List, environment, metric_fn: Callable,
                                  removal_type: str = "ablation") -> np.ndarray:
    if removal_type == "ablation":
        return leave_one_out_attribution(agents, environment, metric_fn, baseline="")
    else:
        raise NotImplementedError(f"{removal_type} not implemented")
