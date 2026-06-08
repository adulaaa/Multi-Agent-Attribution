import numpy as np
from typing import Callable, List

def leave_one_out_attribution(agents: List, environment, metric_fn: Callable, baseline="") -> np.ndarray:
    original_outcome = metric_fn(environment)
    n = len(agents)
    scores = np.zeros(n)
    for i in range(n):
        original_respond = agents[i].respond
        agents[i].respond = lambda _: baseline
        perturbed_outcome = metric_fn(environment)
        scores[i] = original_outcome - perturbed_outcome
        agents[i].respond = original_respond
    return scores

def removal_protocol_attribution(agents: List, environment, metric_fn: Callable, 
                                  removal_type: str = "ablation") -> np.ndarray:
    if removal_type == "ablation":
        return leave_one_out_attribution(agents, environment, metric_fn, baseline="")
    else:
        raise NotImplementedError(f"{removal_type} not implemented")
