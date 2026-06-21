import numpy as np
from scipy import stats

def mean_std_ci(data, confidence=0.95):
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    n = len(arr)
    if n > 1:
        se = stats.sem(arr)
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    else:
        ci = (mean, mean)
    return mean, std, ci

def t_test(a, b):
    return stats.ttest_ind(a, b)
