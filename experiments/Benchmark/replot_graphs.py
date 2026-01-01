
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.calibration import calibration_curve

def replot_stress_dca_run_003():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results", "run_003")
    
    # 1. Stress Robustness Bar Chart
    metrics_path = os.path.join(res_dir, "stress", "stress_metrics.csv")
    if os.path.exists(metrics_path):
        print(f"Replotted Stress Bar Chart from {metrics_path}")
        df = pd.read_csv(metrics_path)
        plt.figure(figsize=(10, 5))
        scenarios = df['scenario_id'].tolist()
        aucs = df['auc'].tolist()
        labels = [s.replace('Scenario_', '') for s in scenarios]
        
        plt.bar(labels, aucs, color='skyblue', edgecolor='black')
        plt.ylim(0, 1.05) # FIXED
        plt.ylabel('AUC Score')
        plt.title('Robustness Across Scenarios')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(res_dir, "stress", "stress_robustness_chart.png"))
        plt.close()

    # 2. Calibration Plot (Stress)
    # Need detailed data for this.
    detailed_path = os.path.join(res_dir, "stress", "stress_detailed.csv")
    if os.path.exists(detailed_path):
        print(f"Replotted Calibration Plot from {detailed_path}")
        df = pd.read_csv(detailed_path)
        
        # Logic from stress_test.py
        # Store data for calibration plot (Base vs Scenario D)
        # We need to filter by scenario
        
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        for sc in ['Baseline', 'Scenario_D_0.50']:
            sub = df[df['scenario_id'] == sc]
            if not sub.empty:
                yt = sub['gt_gas'].values
                yp = sub['ai_prob'].values
                prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10)
                plt.plot(prob_pred, prob_true, "s-", label=f"{sc}")
        
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title("Calibration Plot (Reliability Diagram)")
        plt.ylim(0, 1.05) # FIXED
        plt.legend()
        plt.savefig(os.path.join(res_dir, "stress", "calibration_plot.png"))
        plt.close()

if __name__ == "__main__":
    replot_stress_dca_run_003()

