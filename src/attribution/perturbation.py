import numpy as np
from typing import Callable, List

def perturbation_attribution(response_a: str, response_b: str, 
                             outcome_fn: Callable[[List[str]], float], 
                             agent_idx: int,
                             baseline_value: str = "") -> float:
    original_inputs = [response_a, response_b]
    original_outcome = outcome_fn(original_inputs)
    perturbed_inputs = original_inputs.copy()
    perturbed_inputs[agent_idx] = baseline_value
    perturbed_outcome = outcome_fn(perturbed_inputs)
    return original_outcome - perturbed_outcome

def leave_one_out(responses: List[str], outcome_fn: Callable[[List[str]], float], 
                  baseline: str = "") -> np.ndarray:
    n = len(responses)
    scores = np.zeros(n)
    original_outcome = outcome_fn(responses)
    for i in range(n):
        perturbed = responses.copy()
        perturbed[i] = baseline
        scores[i] = original_outcome - outcome_fn(perturbed)
    return scores
