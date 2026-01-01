
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from core.suite import BenchmarkTask
from core.patient_generator import PatientGenerator
from core.triage_agents import CausalBrainAgent
from core.evaluation import calculate_ece
from core.observation_layer import ObservationLayer

logger = logging.getLogger(__name__)

class StressTest(BenchmarkTask):
    @property
    def name(self):
        return "Stress"

    def run(self) -> dict:
        config = self.load_config("stress_config.yaml")
        model_path = self.get_model_config_path(config)
        
        generator = PatientGenerator(model_path)
        observer = ObservationLayer()
        brain = CausalBrainAgent(model_path)
        
        n_samples = config['stress_testing']['n_samples']
        scenarios = ['Baseline', 'Scenario_B', 'Scenario_C', 'Scenario_D_0.05', 'Scenario_D_0.15', 'Scenario_D_0.30', 'Scenario_D_0.50']
        
        all_results = []
        metrics = []
        calibration_data = {}
        
        for sc in scenarios:
            logger.info(f"Running Stress Scenario: {sc}")
            
            overrides = {}
            if sc.startswith('Scenario_D'):
                val = float(sc.split('_')[-1])
                overrides['GAS'] = {'True': val, 'False': 1.0 - val}
            
            generator.set_override(overrides)
            # generator.set_mutation(mutation_conf) # REMOVED: using ObservationLayer methods
            
            patients = generator.generate(n_samples)
            
            y_true = []
            y_prob = []
            
            for p in patients:
                # Standard Degradation
                p = observer.apply(p, base_noise=0.0)
                
                # Interventions
                if sc == 'Scenario_B':
                     rate = config['stress_testing']['nsaid_masking_rate']
                     observer.apply_latent_confounder(p, rate)
                elif sc == 'Scenario_C':
                     rate = config['stress_testing']['bias_miss_rate']
                     observer.apply_systematic_bias(p, rate, target_node='V_white')
                
                res = brain.predict(p)
                
                # Store
                row = p.ground_truth.copy()
                row['scenario_id'] = sc
                row['gt_gas'] = 1 if (p.etiology in ['PURE_GAS', 'FUSO']) else 0
                row['ai_prob'] = res['prob_treat']
                row['ai_diagnosis'] = res['diagnosis']
                row['mc_score'] = self._simple_mc(p.observation)
                row['alert_silent'] = 1 if res.get('is_silent_danger_alert') else 0
                
                all_results.append(row)
                y_true.append(row['gt_gas'])
                y_prob.append(res['prob_treat'])
            
            # Metrics
            if len(set(y_true)) > 1:
                auc_val = roc_auc_score(y_true, y_prob)
            else:
                auc_val = 0.0
                
            ece = calculate_ece(np.array(y_true), np.array(y_prob))
            
            # Store data for calibration plot (Base vs Scenario D)
            if sc == 'Baseline' or sc == 'Scenario_D_0.50':
                calibration_data[sc] = (y_true, y_prob)
            
            pdr = 0.0
            if sc != 'Baseline':
                base = next((m for m in metrics if m['scenario_id'] == 'Baseline'), None)
                if base and base['auc'] > 0:
                    pdr = (base['auc'] - auc_val) / base['auc'] * 100
            
            # Explainability: Alert Recall (only meaningful for NSAID scenario where we hide symptoms)
            alert_recall = None
            if sc == 'Scenario_B':
                # Ground Truth is GAS (Filtered in loop? No, p.etiology check)
                # We need recall among TRUE POSITIVES (GAS Cases)
                df_sc = pd.DataFrame(all_results[-len(y_true):])
                gas_cases = df_sc[df_sc['gt_gas'] == 1]
                if len(gas_cases) > 0:
                    alert_recall = gas_cases['alert_silent'].mean()
            
            metrics.append({
                'scenario_id': sc,
                'auc': auc_val,
                'ece_score': ece,
                'pdr_rate': pdr,
                'alert_recall': alert_recall
            })
            
        df_log = pd.DataFrame(all_results)
        self.save_csv(df_log, "stress_detailed.csv")
        
        df_sum = pd.DataFrame(metrics)
        self.save_csv(df_sum, "stress_metrics.csv")
        
        # --- Visualization ---
        self._plot_results(metrics, calibration_data)
        
        base_metric = next((m for m in metrics if m['scenario_id'] == 'Baseline'), {})
        return {
            'stress_base_auc': base_metric.get('auc', 0),
            'stress_base_ece': base_metric.get('ece_score', 0)
        }

    def _simple_mc(self, obs):
        score = 0
        if obs.get('C_temp') == 'High': score += 1
        if obs.get('C_cough') == 'Absent': score += 1
        if obs.get('C_lymph') in ['Anterior', 'Both']: score += 1
        if obs.get('V_white') in ['High', 'Low']: score += 1
        if obs.get('Age_Group') == 'Child': score += 1
        elif obs.get('Age_Group') in ['Adult', 'Senior']: score -= 1
        return score

    def _plot_results(self, metrics, calib_data):
        # 1. Calibration Plot
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for name, (yt, yp) in calib_data.items():
            prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10)
            plt.plot(prob_pred, prob_true, "s-", label=f"{name}")
        
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title("Calibration Plot (Reliability Diagram)")
        plt.ylim(0, 1.05)
        plt.legend()
        self.save_plot(plt.gcf(), "calibration_plot.png")
        plt.close()
        
        # 2. Robustness Bar Chart
        plt.figure(figsize=(10, 5))
        scenarios = [m['scenario_id'] for m in metrics]
        aucs = [m['auc'] for m in metrics]
        
        # Shorten names for x-axis
        labels = [s.replace('Scenario_', '') for s in scenarios]
        
        plt.bar(labels, aucs, color='skyblue', edgecolor='black')
        plt.ylim(0, 1.05) # Fixed scale 0-1
        plt.ylabel('AUC Score')
        plt.title('Robustness Across Scenarios')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        self.save_plot(plt.gcf(), "stress_robustness_chart.png")
        plt.close()

