import numpy as np
from typing import Callable, List

def shapley_approximation(responses: List[str], outcome_fn: Callable[[List[str]], float],
                          n_samples: int = 100, baseline: str = "") -> np.ndarray:
    n = len(responses)
    shapley_values = np.zeros(n)
    for _ in range(n_samples):
        perm = np.random.permutation(n)
        current_outcome = outcome_fn([baseline] * n)
        for i, idx in enumerate(perm):
            coalition = [baseline] * n
            for j in perm[:i+1]:
                coalition[j] = responses[j]
            new_outcome = outcome_fn(coalition)
            marginal = new_outcome - current_outcome
            shapley_values[idx] += marginal / n_samples
            current_outcome = new_outcome
    return shapley_values

def data_shapley_style_attribution(agents: List, outcome_fn: Callable, 
                                   n_samples: int = 100, baseline: str = "") -> np.ndarray:
    n = len(agents)
    shapley = np.zeros(n)
    for _ in range(n_samples):
        perm = np.random.permutation(n)
        current_outcome = outcome_fn([])
        for i, idx in enumerate(perm):
            coalition = perm[:i+1].tolist()
            new_outcome = outcome_fn(coalition)
            marginal = new_outcome - current_outcome
            shapley[idx] += marginal / n_samples
            current_outcome = new_outcome
    return shapley

def exact_shapley_2_agents(responses: List, outcome_fn: Callable, baseline: str = "") -> np.ndarray:
    v0 = outcome_fn([baseline, baseline])
    v1 = outcome_fn([responses[0], baseline])
    v2 = outcome_fn([baseline, responses[1]])
    v12 = outcome_fn([responses[0], responses[1]])
    phi1 = 0.5 * (v1 - v0) + 0.5 * (v12 - v2)
    phi2 = 0.5 * (v2 - v0) + 0.5 * (v12 - v1)
    return np.array([phi1, phi2])
