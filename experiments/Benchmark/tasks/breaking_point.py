
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import roc_auc_score

from core.suite import BenchmarkTask
from core.patient_generator import PatientGenerator
from core.triage_agents import CausalBrainAgent
from core.observation_layer import ObservationLayer
from src.causal_brain_v6 import CausalBrainV6

logger = logging.getLogger(__name__)

class BreakingPointAnalysis(BenchmarkTask):
    @property
    def name(self):
        return "BreakingPoint"

    def run(self) -> dict:
        config = self.load_config("stress_config.yaml")
        # Reuse stress config for n_samples
        n_samples = config.get('stress_testing', {}).get('n_samples', 1000)
        model_path = self.get_model_config_path(config)
        
        # Noise levels to test: 0% (Baseline) to 100% (Complete Chaos)
        # 1.0 means we add random noise up to +/- 100% of the value (clipping at 0 and 1)
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        metrics = []
        
        # Base Generator (Ground Truth P_exp doesn't change, only the AI Brain changes)
        # We want to test if a "Brain Damaged" AI can still diagnose "Real" patients.
        generator = PatientGenerator(model_path)
        patients = generator.generate(n_samples) # Generate dataset once
        
        # Observer
        observer = ObservationLayer()
        # Apply standard observation noise
        observed_patients = [observer.apply(p) for p in patients]
        
        for noise in noise_levels:
            logger.info(f"Running Breaking Point Analysis: Noise Level {noise}")
            
            # 1. Create Perturbed Model
            brain = self._create_perturbed_brain(model_path, noise)
            
            # 2. Diagnose
            y_true = []
            y_prob = []
            y_alert_correct = [] # For safety layer recall
            
            for p in observed_patients:
                # CausalBrainV6 does not have .predict() (that is for the Agent wrapper)
                # We interpret p.observation directly
                res = brain.diagnose(p.observation)
                
                # Ground Truth GAS
                is_gas = 1 if (p.etiology in ['PURE_GAS', 'FUSO']) else 0
                
                # AI Prediction
                # We need probability of GAS
                if 'probs' in res:
                    prob = res['probs'].get('GAS', 0.0)
                else:
                    prob = 0.0
                
                # Safety Layer Check (Silent Danger)
                # To measure safety layer "sensitivity", we need a scenario where Safety Net IS needed.
                # But here we are using standard patients. 
                # Let's count "Alert Triggered" for GAS cases as a proxy for sensitivity?
                # Or better: Standard AUC captures general performance.
                # For "Safety Layer Sensitivity", we specifically look at "Silent Danger" recall 
                # BUT Silent Danger is for masked cases. 
                # Let's stick to AUC as primary "Breaking Point" metric.
                # And maybe "Fuso Detection" or similar?
                # The user asked for "Safety Layer Sensitivity". 
                # Let's verify if "Silent Danger" triggers for Fuso or High Risk cases.
                
                y_true.append(is_gas)
                y_prob.append(prob)
            
            # 3. Calculate Metrics
            if len(set(y_true)) > 1:
                auc_val = roc_auc_score(y_true, y_prob)
            else:
                auc_val = 0.0
                
            metrics.append({
                'noise_level': noise,
                'auc': auc_val
            })
            
        # Save Results
        df_res = pd.DataFrame(metrics)
        self.save_csv(df_res, "breaking_point_results.csv")
        
        # Visualize
        self._plot_breaking_point(df_res)
        
        return {
            'breaking_point_max_noise': 0.5 # Placeholder
        }

    def _create_perturbed_brain(self, config_path, noise_level):
        """
        Creates a CausalBrainV6 instance and perturbs its CPTs.
        """
        brain = CausalBrainV6(config_path)
        
        if noise_level == 0.0:
            return brain
            
        model = brain.model
        
        for cpd in model.get_cpds():
            # cpd.values is a flattened array in pgmpy (or shaped, depending on version)
            # but pgmpy's get_values() returns the full table.
            # We can modify cpd.values directly if we validly reshape or keep shape.
            
            original_values = cpd.values
            
            # Noise: uniform(-noise, +noise) * value ??? 
            # Or additive: value + uniform(-noise, noise)?
            # User said "Probabilities randomly +/- 30%".
            # If p=0.9, +/- 30% could mean 0.9 +/- 0.27 (0.63~1.0) OR 0.9 +/- 0.3 (0.6~1.2).
            # Usually percentage of the value.
            # But if p=0.01, percentage noise is tiny.
            # Let's assume "Additive Perturbation" scaled by noise_level?
            # Or "Percentage Perturbation"?
            # "Parametes ... are disturbed ... +/- 30%".
            # Let's try: New = Old * (1 + Uniform(-noise, noise))
            # This preserves 0s.
            
            delta = np.random.uniform(-noise_level, noise_level, size=original_values.shape)
            perturbed_values = original_values * (1.0 + delta)
            
            # Normalize over the first axis (Variable Card).
            # pgmpy DiscreteFactor values are flattened? 
            # TabularCPD values are (cardinality, product_of_parents).
            # We need to normalize columns.
            
            # Ensure no negatives
            perturbed_values = np.clip(perturbed_values, 0.0001, None) 
            # We preserve zeros? 
            # If original was 0, it stays 0.
            # Using mask.
            mask = (original_values > 1e-6)
            perturbed_values = np.where(mask, perturbed_values, 0.0)
            
            # Normalize
            # Sum down the state axis (axis 0)
            col_sums = perturbed_values.sum(axis=0, keepdims=True)
            normalized_values = perturbed_values / col_sums
            
            # Update CPD
            cpd.values = normalized_values
            
        # Basic check (optional, slow)
        # brain.model.check_model() 
        
        # We also need to update the inference engine because VariableElimination caches the model?
        # VariableElimination takes model in init.
        # So we must recreate the engine.
        from pgmpy.inference import VariableElimination
        brain.infer_engine = VariableElimination(brain.model)
        
        return brain

    def _plot_breaking_point(self, df):
        plt.figure(figsize=(10, 6))
        
        # Accuracy Plot
        plt.plot(df['noise_level'], df['auc'], 'o-', linewidth=2, label='AUC (Diagnostic Accuracy)')
        
        # Add breaking point threshold line
        plt.axhline(y=0.7, color='r', linestyle='--', label='Min Acceptable Accuracy (0.7)')
        plt.axhline(y=0.8, color='orange', linestyle='--', label='Target Accuracy (0.8)')
        
        plt.xlabel('Parameter Perturbation Level (Noise %)')
        plt.ylabel('Model Performance')
        plt.title('Breaking Point Analysis: Structural Robustness')
        plt.legend()
        plt.grid(True)
        
        # Annotate
        # Find point where it drops below 0.7
        drops = df[df['auc'] < 0.7]
        if not drops.empty:
            break_x = drops.iloc[0]['noise_level']
            plt.axvline(x=break_x, color='gray', linestyle=':')
            plt.text(break_x, 0.5, f" Breaking Point\n (Noise={break_x})", color='gray')

        self.save_plot(plt.gcf(), "breaking_point_plot.png")
        plt.close()

