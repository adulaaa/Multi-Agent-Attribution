import numpy as np
from typing import List, Dict, Any, Callable, Tuple

class CausalAttribution:
    def __init__(self, trace: List[Dict[str, Any]], outcome_fn: Callable):
        self.trace = trace
        self.outcome_fn = outcome_fn
        self.n = len(trace)
    
    def compute_causal_responsibility_scores(self) -> np.ndarray:
        crs = np.zeros(self.n)
        original_outcome = self.outcome_fn(self.trace)
        if original_outcome == 1:
            print("Trace succeeded – CRS only defined for failures")
            return crs
        for i in range(self.n):
            intervened_trace = self.trace.copy()
            intervened_trace[i] = self._minimal_repair(intervened_trace[i])
            new_outcome = self.outcome_fn(intervened_trace)
            crs[i] = 1.0 if new_outcome == 1 else 0.0
        return crs
    
    def _minimal_repair(self, step: Dict) -> Dict:
        repaired = step.copy()
        repaired['output'] = "[REPAIRED] " + step['output']
        return repaired
    
    def identify_failure_steps(self, threshold: float = 0.5) -> List[int]:
        scores = self.compute_causal_responsibility_scores()
        return [i for i, score in enumerate(scores) if score >= threshold]

def causal_attribution_for_dialogue(history: List[Tuple[str, str, str]], 
                                     success_fn: Callable) -> np.ndarray:
    trace = []
    for i, (prompt, resp_a, resp_b) in enumerate(history):
        trace.append({'step': i, 'agent': 'User', 'action': 'prompt', 'input': prompt, 'output': prompt})
        trace.append({'step': i, 'agent': 'AgentA', 'action': 'respond', 'input': prompt, 'output': resp_a})
        trace.append({'step': i, 'agent': 'AgentB', 'action': 'respond', 'input': resp_a, 'output': resp_b})
    causal = CausalAttribution(trace, success_fn)
    return causal.compute_causal_responsibility_scores()
