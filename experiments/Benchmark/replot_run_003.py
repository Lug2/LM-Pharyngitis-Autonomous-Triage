
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import roc_curve, auc

# Adjust path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.evaluation import calculate_iso_metrics
from core.stats_utils import calculate_mcnemar_test

def replot_run_003():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "results", "run_003", "standard", "standard_simulation_data.csv")
    output_path = os.path.join(base_dir, "results", "run_003", "standard", "standard_report.png")
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Reconstruct data dict for calc function
    # 'truth': [], 'mc_score': [], 'ai_prob': []
    data = {
        'truth': df['truth'].values,
        'mc_score': df['mc_score'].values,
        'ai_prob': df['ai_prob'].values
    }
    
    print("Calculating metrics...")
    # Thresh=3 hardcoded as per standard
    result = calculate_iso_metrics(data, mcisaac_threshold=3)
    
    print("Generating plot...")
    # Load robustness data?
    # Actually robustness plot is separate in ax2. 
    # We need robustness data for ax2 to reproduce the full plot.
    # But robustness data is not saved as CSV in standard_bench.py currently! 
    # Wait, save_csv(df, "standard_simulation_data.csv") only saves simulation data (Part 1).
    # Robustness data is NOT saved in the current implementation of standard_bench.py!
    # Checking code...
    # `robustness` variable is calculated but not saved to CSV. 
    # So I cannot perfectly reproduce Ax2 (Robustness) without re-running robustness sim.
    # However, robustness sim is N=2000, faster than Iso N=10000.
    
    # Since I cannot reproduce Ax2 easily without re-running, 
    # and the user complained about Iso-Point (Ax1).
    # I will recreate the plot. For Ax2, I will just put a placeholder or try to read if I saved it?
    # I did NOT save robustness data to CSV in standard_bench.py. 
    
    # Plan: Just plot Ax1 (ROC) for now to show the user the Iso-Point fix? 
    # Or asking user to re-run is better?
    # The user asked "Is it missing?". 
    # I fixed the code.
    # I will admit I cannot perfectly replot without re-run for the Robustness part.
    # BUT! I can check if I can quickly run robustness? 
    # No, that requires loading model etc.
    
    # Let's just fix the code and tell the user "I fixed it. Since robustness data wasn't saved separately, please run again or I can run it for you."
    # OR, I just plot the ROC curve (Ax1) to a separate file `roc_check.png` to prove it works?
    
    fig, ax1 = plt.subplots(figsize=(7, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(data['truth'], data['ai_prob'])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'AI (AUC={roc_auc:.2f})')
    ax1.scatter([1-result.mcisaac_specificity], [result.mcisaac_sensitivity], c='g', label='McIsaac')
    
    # Iso-Point
    ai_fpr = 1 - result.ai_specificity
    ai_tpr = result.ai_iso_sensitivity
    ax1.scatter([ai_fpr], [ai_tpr], color='blue', marker='x', s=100, label='Iso-sensitivity Point')
    
    ax1.text(0.6, 0.2, f"McNemar p={result.p_value:.1e}", transform=ax1.transAxes)
    ax1.set_title("ROC Curve (Regenerated)")
    ax1.legend(loc='lower right')
    
    check_path = os.path.join(base_dir, "results", "run_003", "standard", "roc_with_iso_point.png")
    plt.savefig(check_path)
    print(f"Saved check plot to {check_path}")

if __name__ == "__main__":
    replot_run_003()

