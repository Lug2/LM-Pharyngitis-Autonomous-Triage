import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def main():
    base_dir = r"C:\Users\leonm\Desktop\喉診断\Benchmark\results\run_006\standard"
    
    # 1. Load Baseline (Iso-Threshold) Results
    subgroup_path = os.path.join(base_dir, "subgroup_analysis.csv")
    if os.path.exists(subgroup_path):
        sub_df = pd.read_csv(subgroup_path)
        child_base = sub_df[sub_df['Group_Value'] == 'Child'].iloc[0]
        print(f"Baseline (Iso-Threshold) Child Sensitivity: {child_base['AI_Sens']:.4f}")
        print(f"Baseline (Iso-Threshold) Child Specificity: {child_base['AI_Spec']:.4f}")
    else:
        print("Subgroup analysis file not found.")
        return

    # 2. Load Raw Data for Pediatric Mode (Threshold = 0.50)
    data_path = os.path.join(base_dir, "standard_simulation_data.csv")
    if not os.path.exists(data_path):
        print("Simulation data not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # Filter for Child
    child_df = df[df['age_group'] == 'Child']
    
    # Calculate Metrics at Threshold 0.50
    y_true = child_df['truth']
    y_pred = (child_df['ai_prob'] >= 0.50).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nPediatric Mode (Thresh=0.50) Child Sensitivity: {sens:.4f}")
    print(f"Pediatric Mode (Thresh=0.50) Child Specificity: {spec:.4f}")
    
    # Impact
    sens_diff = sens - child_base['AI_Sens']
    print(f"\nImprovement in Sensitivity: +{sens_diff:.4f}")

if __name__ == "__main__":
    main()

