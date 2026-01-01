
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from .data_schema import BenchmarkResult
from .stats_utils import calculate_mcnemar_test, calculate_wilson_score_interval

def calculate_iso_metrics(sim_data: dict, mcisaac_threshold: int = 3) -> BenchmarkResult:
    y_true = np.array(sim_data['truth'])
    y_mc = np.array(sim_data['mc_score'])
    y_ai = np.array(sim_data['ai_prob'])
    
    # 1. McIsaac Sensitivity
    mc_mask = (y_mc >= mcisaac_threshold)
    mc_tp = np.sum((y_true == 1) & mc_mask)
    mc_tn = np.sum((y_true == 0) & (~mc_mask))
    mc_p = np.sum(y_true == 1)
    mc_n = np.sum(y_true == 0)
    
    mc_sens = mc_tp / mc_p if mc_p > 0 else 0
    mc_spec = mc_tn / mc_n if mc_n > 0 else 0
    
    # 2. AI Iso-Sensitivity
    fpr, tpr, thresholds = roc_curve(y_true, y_ai)
    
    idx = np.searchsorted(tpr, mc_sens, side='left')
    if idx >= len(tpr): idx = len(tpr) - 1
    
    iso_threshold = thresholds[idx]
    # Ensure threshold is reasonable (sometimes searchsorted gives index 0 which is threshold > 1)
    # If possible, pick best match
    
    ai_sens_iso = tpr[idx]
    ai_fpr_iso = fpr[idx]
    ai_spec_iso = 1.0 - ai_fpr_iso
    
    # 3. McNemar's
    y_pred_mc = (y_mc >= mcisaac_threshold).astype(int)
    y_pred_ai = (y_ai >= iso_threshold).astype(int)
    
    p_val = calculate_mcnemar_test(y_true, y_pred_mc, y_pred_ai)
    
    return BenchmarkResult(
        scenario="Overall",
        mcisaac_sensitivity=mc_sens,
        ai_iso_sensitivity=ai_sens_iso,
        mcisaac_specificity=mc_spec,
        ai_specificity=ai_spec_iso,
        p_value=p_val,
        break_point=iso_threshold 
    )

def _fmt_ci(ci_tuple):
    if not ci_tuple: return ""
    return f"({ci_tuple[0]:.3f}-{ci_tuple[1]:.3f})"

def calculate_subgroup_metrics(sim_data: dict, ai_threshold: float, mc_threshold: int = 3) -> pd.DataFrame:
    """
    Calculates Sens/Spec for subgroups (Age, McIsaac Score Bins).
    """
    df = pd.DataFrame({
        'truth': sim_data['truth'],
        'ai_prob': sim_data['ai_prob'],
        'mc_score': sim_data['mc_score'],
        'age_group': sim_data.get('age_group', ['Unknown']*len(sim_data['truth']))
    })
    
    # Stratify by Age
    results = []
    
    # 1. Age Groups
    for age in df['age_group'].unique():
        sub = df[df['age_group'] == age]
        if sub.empty: continue
        
        y_true = sub['truth']
        y_pred_ai = (sub['ai_prob'] >= ai_threshold).astype(int)
        y_pred_mc = (sub['mc_score'] >= mc_threshold).astype(int)
        
        res_ai = _calc_sens_spec(y_true, y_pred_ai)
        res_mc = _calc_sens_spec(y_true, y_pred_mc)
        
        results.append({
            'Group_Type': 'Age',
            'Group_Value': age,
            'N': len(sub),
            'AI_Sens': res_ai['sens'],
            'AI_Sens_CI': _fmt_ci(res_ai['sens_ci']),
            'AI_Spec': res_ai['spec'],
            'AI_Spec_CI': _fmt_ci(res_ai['spec_ci']),
            'Mc_Sens': res_mc['sens'],
            'Mc_Sens_CI': _fmt_ci(res_mc['sens_ci']),
            'Mc_Spec': res_mc['spec'],
            'Mc_Spec_CI': _fmt_ci(res_mc['spec_ci'])
        })
        
    # 2. Severity (McIsaac Score Bins)
    # Low (0-1), Mid (2-3), High (4+)
    # Note: Using McIsaac Score as a proxy for severity/clinical presentation
    bins = [
        ('Low (0-1)', lambda s: s <= 1),
        ('Mid (2-3)', lambda s: (s >= 2) & (s <= 3)),
        ('High (4+)', lambda s: s >= 4)
    ]
    
    for label, func in bins:
        sub = df[func(df['mc_score'])]
        if sub.empty: continue
        
        y_true = sub['truth']
        y_pred_ai = (sub['ai_prob'] >= ai_threshold).astype(int)
        
        # McIsaac prediction is naturally tied to bins, so comparing McIsaac to McIsaac bins is tautological for Spec at Low, Sens at High?
        # But we can compare AI accuracy within these bins.
        
        res_ai = _calc_sens_spec(y_true, y_pred_ai)
        
        results.append({
            'Group_Type': 'McIsaac_Severity',
            'Group_Value': label,
            'N': len(sub),
            'AI_Sens': res_ai['sens'],
            'AI_Sens_CI': _fmt_ci(res_ai['sens_ci']),
            'AI_Spec': res_ai['spec'],
            'AI_Spec_CI': _fmt_ci(res_ai['spec_ci']),
            'Mc_Sens': None, 
            'Mc_Sens_CI': None,
            'Mc_Spec': None,
            'Mc_Spec_CI': None
        })
        
    return pd.DataFrame(results)

def calculate_hybrid_metrics(sim_data: dict, default_threshold: float, override_rules: dict) -> dict:
    """
    Calculates overall metrics using a hybrid threshold strategy.
    override_rules: {'Age_Group': {'Child': 0.50}}
    """
    truths = np.array(sim_data['truth'])
    probs = np.array(sim_data['ai_prob'])
    
    # Defaults
    preds = (probs >= default_threshold).astype(int)
    
    # Apply Overrides
    for key_obs, rule in override_rules.items():
        # Get observation data (assuming sim_data has list of dicts or parallel lists)
        # sim_data keys: 'truth', 'ai_prob', 'age_group'... 
        # For this benchmark we specifically saved 'age_group'.
        key_k = key_obs.lower() # sim_data keys are lowercase usually
        if key_k not in sim_data: 
             # Try mapping 'Age_Group' -> 'age_group'
             continue
             
        group_data = np.array(sim_data[key_k])
        
        for group_val, specific_thresh in rule.items():
            mask = (group_data == group_val)
            preds[mask] = (probs[mask] >= specific_thresh).astype(int)
            
    # Calculate Metrics
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(truths, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'hybrid_sensitivity': sens,
        'hybrid_specificity': spec,
        'hybrid_accuracy': (tp + tn) / len(truths)
    }

def _calc_sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    sens_ci = calculate_wilson_score_interval(tp, tp + fn)
    spec_ci = calculate_wilson_score_interval(tn, tn + fp)
    
    return {'sens': sens, 'spec': spec, 'sens_ci': sens_ci, 'spec_ci': spec_ci}

def calculate_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i+1]
        if i == n_bins - 1:
            mask = (y_prob >= start) & (y_prob <= end)
        else:
            mask = (y_prob >= start) & (y_prob < end)
        
        count = np.sum(mask)
        if count > 0:
            avg_pred = np.mean(y_prob[mask])
            avg_true = np.mean(y_true[mask])
            ece += (count / total) * np.abs(avg_pred - avg_true)
            
    return ece

def calculate_net_benefit(y_true, y_prob, threshold):
    N = len(y_true)
    if N == 0: return 0.0
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    if threshold >= 1.0:
        weight = np.inf
    else:
        weight = threshold / (1.0 - threshold)
        
    net_benefit = (tp / N) - (fp / N) * weight
    return net_benefit

