
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Project Root


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from core.patient_generator import PatientGenerator
from core.observation_layer import ObservationLayer
from src.causal_brain_v6 import CausalBrainV6

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_random_brain(config_path):
    """
    Creates a brain with TOTALLY RANDOM CPTs (Destructive Noise).
    This should break the model completely (AUC ~ 0.5).
    """
    brain = CausalBrainV6(config_path)
    model = brain.model
    
    for cpd in model.get_cpds():
        # Replace values with complete random uniform [0, 1]
        random_values = np.random.random(cpd.values.shape)
        
        # Normalize
        col_sums = random_values.sum(axis=0, keepdims=True)
        normalized_values = random_values / col_sums
        
        cpd.values = normalized_values
        
    from pgmpy.inference import VariableElimination
    brain.infer_engine = VariableElimination(brain.model)
    return brain

def run_sanity_check():
    logger.info("=== Starting Breaking Point Validity Check ===")
    
    config_path = "../../inference_model/model_config.yaml"
    
    # 1. Generate Data (Ground Truth) using NORMAL model
    logger.info("Generating Ground Truth Data (N=10000)...")
    generator = PatientGenerator(config_path)
    patients = generator.generate(10000)
    
    observer = ObservationLayer()
    observed_patients = [observer.apply(p) for p in patients]
    
    y_true = []
    for p in observed_patients:
        is_gas = 1 if (p.etiology in ['PURE_GAS', 'FUSO']) else 0
        y_true.append(is_gas)
        
    # Check if we have both classes
    if len(set(y_true)) < 2:
        logger.error("Data generation failed to produce both classes. Retrying...")
        return

    # 2. Test NORMAL Brain (Control)
    logger.info("Testing Normal Brain...")
    normal_brain = CausalBrainV6(config_path)
    y_prob_normal = []
    for p in observed_patients:
        res = normal_brain.diagnose(p.observation)
        prob = res.get('probs', {}).get('GAS', 0.0)
        y_prob_normal.append(prob)
        
    auc_normal = roc_auc_score(y_true, y_prob_normal)
    logger.info(f"Normal Brain AUC: {auc_normal:.4f}")
    

    # 3. Test RANDOM Brain (Destruction) - Multiple Trials
    logger.info("Testing RANDOM Brain (Destructive Noise) - 10 Trials...")
    
    auc_scores = []
    for i in range(10):
        random_brain = create_random_brain(config_path)
        y_prob_random = []
        for p in observed_patients:
            res = random_brain.diagnose(p.observation)
            prob = res.get('probs', {}).get('GAS', 0.0)
            y_prob_random.append(prob)
            
        auc = roc_auc_score(y_true, y_prob_random)
        auc_scores.append(auc)
        logger.info(f"Trial {i+1} AUC: {auc:.4f}")
        
    avg_auc = np.mean(auc_scores)
    logger.info(f"Average Random Brain AUC: {avg_auc:.4f}")
    
    if avg_auc < 0.6:
        logger.info("VALIDITY CONFIRMED: Random brain failed (Avg AUC ~ 0.5).")
        print("SUCCESS: Validity Check Passed. Random noise breaks the model.")
    else:
        logger.warning(f"VALIDITY WARNING: Random brain consistently performs well (Avg AUC={avg_auc:.4f}). Investigation needed.")
        print("FAILURE: Validity Check Failed. Random noise did not break the model.")

if __name__ == "__main__":
    run_sanity_check()

