
import numpy as np
from scipy.stats import chi2, entropy, chisquare
from typing import Dict, List, Any

def calculate_mcnemar_test(y_true: np.ndarray, y_pred_1: np.ndarray, y_pred_2: np.ndarray) -> float:
    """
    Calculates McNemar's test p-value for comparing two classifiers.
    """
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)
    
    n10 = np.sum(correct_1 & ~correct_2) # M1 correct, M2 wrong
    n01 = np.sum(~correct_1 & correct_2) # M1 wrong, M2 correct
    
    if (n10 + n01) == 0:
        return 1.0
        
    statistic = ((np.abs(n10 - n01) - 1) ** 2) / (n10 + n01) # Continuity correction
    
    p_value = chi2.sf(statistic, 1)
    
    return p_value

def calculate_kl_divergence(p_counts: Dict[Any, int], q_probs: Dict[Any, float]) -> float:
    """
    Calculates KL Divergence D_KL(P || Q) to validate data fidelity.
    P: Observed distribution (from generated counts)
    Q: Expected distribution (theoretical)
    
    Tucker et al. (2020): D_KL < 0.05 indicates high fidelity.
    """
    # Align keys
    all_keys = set(p_counts.keys()) | set(q_probs.keys())
    
    p_vals = []
    q_vals = []
    
    total_count = sum(p_counts.values())
    if total_count == 0: return 0.0
    
    for k in sorted(list(all_keys)):
        # P (Observed Prob)
        p = p_counts.get(k, 0) / total_count
        # Handle zero prob in P for KL (0 * log(0) = 0, but scipy handles it if input is prob)
        # Actually standard KL is sum P log P/Q. P=0 -> 0 contribution.
        # But we create array. 
        # Add epsilon to P if needed? No, 0 is fine if P is 0.
        
        # Q (Expected Prob)
        q = q_probs.get(k, 1e-9) # Avoid division by zero if Q is 0 but P is not
        
        p_vals.append(p)
        q_vals.append(q)
        
    p_vals = np.array(p_vals)
    q_vals = np.array(q_vals)
    
    # Normalize Q just in case
    q_vals = q_vals / np.sum(q_vals)
    
    # Scipy entropy(pk, qk) calculates S = sum(pk * log(pk / qk))
    return entropy(p_vals, q_vals)

def calculate_chi_squared_test(p_counts: Dict[Any, int], q_probs: Dict[Any, float]) -> float:
    """
    Performs Chi-Squared Goodness of Fit Test.
    Returns p-value. p > 0.05 indicates no significant difference (Good Fit).
    """
    all_keys = set(p_counts.keys()) | set(q_probs.keys())
    total_count = sum(p_counts.values())
    if total_count == 0: return 0.0
    
    f_obs = []
    f_exp = []
    
    for k in sorted(list(all_keys)):
        obs = p_counts.get(k, 0)
        exp = q_probs.get(k, 0.0) * total_count
        
        # Chi-square requires exp >= 5 usually. 
        # We assume large N (10,000) so usually fine.
        f_obs.append(obs)
        f_exp.append(exp)
        
    s, p = chisquare(f_obs, f_exp)
    return p

def calculate_wilson_score_interval(count: int, total: int, confidence: float = 0.95):
    """
    Calculates Wilson Score Interval for binomial proportion.
    Returns (lower_bound, upper_bound).
    """
    if total == 0: return (0.0, 0.0)
    
    from scipy.stats import norm
    p = count / total
    n = total
    z = norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2/n
    center_adjusted_probability = p + z**2 / (2*n)
    adjusted_standard_deviation = z * np.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    
    lower_bound = (center_adjusted_probability - adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + adjusted_standard_deviation) / denominator
    
    return (max(0.0, lower_bound), min(1.0, upper_bound))

