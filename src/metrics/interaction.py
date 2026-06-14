"""
Metrics for interaction‑level interpretability.

Provides functions to measure which agent‑to‑agent messages most influenced
the joint decision, e.g., influence of each utterance on final outcome.
"""

import numpy as np
from typing import List, Callable, Any

def message_influence_matrix(messages: List[str], outcome_fn: Callable[[List[str]], float],
                             baseline: str = "") -> np.ndarray:
    """
    Compute pairwise influence: how much removing each message changes outcome.
    Returns a matrix M where M[i,j] = change in outcome when message i is removed
    while evaluating outcome on the full set (simplified). Actually returns a
    vector of per‑message influence, not a full matrix.
    """
    n = len(messages)
    orig_outcome = outcome_fn(messages)
    influence = np.zeros(n)
    for i in range(n):
        perturbed = messages.copy()
        perturbed[i] = baseline
        new_outcome = outcome_fn(perturbed)
        influence[i] = orig_outcome - new_outcome
    return influence

def most_influential_message(messages: List[str], outcome_fn: Callable[[List[str]], float],
                             baseline: str = "") -> int:
    """Return index of the message whose removal causes largest absolute outcome change."""
    n = len(messages)
    orig_outcome = outcome_fn(messages)
    deltas = []
    for i in range(n):
        perturbed = messages.copy()
        perturbed[i] = baseline
        new_outcome = outcome_fn(perturbed)
        deltas.append(abs(orig_outcome - new_outcome))
    return int(np.argmax(deltas))

def interaction_strength(messages: List[str], outcome_fn: Callable[[List[str]], float],
                         baseline: str = "") -> float:
    """
    Measure how much the outcome depends on interactions between messages.
    High value = joint effect > sum of individuals (synergy).
    """
    orig = outcome_fn(messages)
    sum_individual = 0.0
    for i in range(len(messages)):
        single = [baseline] * len(messages)
        single[i] = messages[i]
        sum_individual += outcome_fn(single)
    baseline_outcome = outcome_fn([baseline] * len(messages))
    return orig - sum_individual + (len(messages)-1) * baseline_outcome
