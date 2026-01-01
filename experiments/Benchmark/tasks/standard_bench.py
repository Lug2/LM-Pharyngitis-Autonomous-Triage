
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from collections import Counter
from sklearn.metrics import roc_curve, auc

from core.suite import BenchmarkTask
from core.patient_generator import PatientGenerator
from core.observation_layer import ObservationLayer
from core.triage_agents import McIsaacAgent, CausalBrainAgent
from core.evaluation import calculate_iso_metrics
from core.stats_utils import calculate_kl_divergence, calculate_chi_squared_test

logger = logging.getLogger(__name__)

class StandardBenchmark(BenchmarkTask):
    @property
    def name(self):
        return "Standard"

    def run(self) -> dict:
        config = self.load_config("basic_config.yaml")
        model_path = self.get_model_config_path(config)
        
        # Init Components
        generator = PatientGenerator(model_path, alpha_leak=config['patient_generation']['alpha_leak'])
        observer = ObservationLayer()
        mcisaac = McIsaacAgent()
        ai = CausalBrainAgent(model_path)
        
        # --- PHASE 0: Fidelity Validation (Tucker et al. 2020) ---
        self._run_fidelity_check(generator)

        # Part 1: Iso-Sensitivity
        n_iso = config['simulation']['iso_samples']
        base_noise = config['simulation']['base_noise']
        
        data = self._run_simulation(generator, observer, mcisaac, ai, n_iso, base_noise)
        iso_result = calculate_iso_metrics(data, config['metrics']['mcisaac_threshold'])
        
        # Part 2: Robustness
        n_rob = config['simulation']['robustness_samples']
        s_rob = config['simulation']['robustness_steps']
        max_noise = config['simulation']['max_noise']
        
        robustness = self._analyze_robustness(generator, observer, mcisaac, ai, n_rob, s_rob, max_noise, iso_result.break_point)
        
        # Visualization
        self._generate_plots(data, robustness, iso_result)
        
        # Save detailed data
        df = pd.DataFrame(data)
        self.save_csv(df, "standard_simulation_data.csv")
        
        # Subgroup Analysis
        from core.evaluation import calculate_subgroup_metrics, calculate_hybrid_metrics
        subgroup_df = calculate_subgroup_metrics(data, iso_result.break_point, config['metrics']['mcisaac_threshold'])
        self.save_csv(subgroup_df, "subgroup_analysis.csv")
        logger.info("Subgroup Analysis saved.")

        # Pediatric Mode Analysis (Child Threshold = 0.50)
        hybrid_res = calculate_hybrid_metrics(
            data, 
            default_threshold=iso_result.break_point,
            override_rules={'age_group': {'Child': 0.50}}
        )
        logger.info(f"=== Pediatric Mode (Child Thresh=0.50) ===")
        logger.info(f"  Hybrid Sens: {hybrid_res['hybrid_sensitivity']:.4f}")
        logger.info(f"  Hybrid Spec: {hybrid_res['hybrid_specificity']:.4f}")
        
        # Save Hybrid Metrics
        pd.DataFrame([hybrid_res]).to_csv(os.path.join(self.output_dir, "pediatric_mode_metrics.csv"), index=False)

        return {
            'std_mc_sens': iso_result.mcisaac_sensitivity,
            'std_ai_sens': iso_result.ai_iso_sensitivity,
            'std_mc_spec': iso_result.mcisaac_specificity,
            'std_ai_spec': iso_result.ai_specificity,
            'std_p_value': iso_result.p_value,
            'pediatric_sens': hybrid_res['hybrid_sensitivity'],
            'pediatric_spec': hybrid_res['hybrid_specificity']
        }

    def _run_fidelity_check(self, generator):
        logger.info("=== Starting Fidelity Validation (Tucker et al. 2020) ===")
        # N=10,000 for statistical validity
        patients = generator.generate(1000)
        
        # 1. Age Group (Expectation from CPT)
        # Assuming CPT is loaded in generator.config
        # We need to extract the 'probs' for Age_Group root node
        age_cpt = next((n for n in generator.config['cpts'] if n['node'] == 'Age_Group'), None)
        fidelity_report = []
        
        if age_cpt:
            exp_probs = age_cpt['probs'] # {'Child': 0.5, ...}
            obs_counts = Counter([p.ground_truth['Age_Group'] for p in patients])
            
            kl = calculate_kl_divergence(obs_counts, exp_probs)
            p_chi = calculate_chi_squared_test(obs_counts, exp_probs)
            
            result = "PASS" if kl < 0.05 else "FAIL"
            logger.info(f"[Fidelity] Age_Group KL={kl:.4f} ({result}), Chi2_p={p_chi:.4f}")
            
            fidelity_report.append({
                'Variable': 'Age_Group',
                'KL_Divergence': kl,
                'Chi2_P_Value': p_chi,
                'Status': result
            })
            
        # 2. GAS Prevalence (Marginalized?) 
        # GAS is conditional on Age. So we cannot simply compare to one CPT unless we calculate marginal.
        # But we can check conditional fidelity?
        # Or check Root nodes mainly.
        # C_epidemic is root.
        epi_cpt = next((n for n in generator.config['cpts'] if n['node'] == 'C_epidemic'), None)
        if epi_cpt:
            exp_probs = epi_cpt['probs']
            obs_counts = Counter([p.ground_truth['C_epidemic'] for p in patients])
            
            kl = calculate_kl_divergence(obs_counts, exp_probs)
            p_chi = calculate_chi_squared_test(obs_counts, exp_probs)
            result = "PASS" if kl < 0.05 else "FAIL"
            logger.info(f"[Fidelity] C_epidemic KL={kl:.4f} ({result}), Chi2_p={p_chi:.4f}")
            fidelity_report.append({'Variable': 'C_epidemic', 'KL_Divergence': kl, 'Chi2_P_Value': p_chi, 'Status': result})

        # Save Report
        self.save_csv(pd.DataFrame(fidelity_report), "fidelity_report.csv")

    def _run_simulation(self, get, obs, mc, ai, n, noise):
        patients = get.generate(n)
        res = {'truth': [], 'mc_score': [], 'ai_prob': [], 'age_group': []}
        
        for p in patients:
            p = obs.apply(p, base_noise=noise)
            m_score = mc.predict(p)
            a_res = ai.predict(p)
            
            is_treat = (p.etiology in ['PURE_GAS', 'FUSO'])
            res['truth'].append(1 if is_treat else 0)
            res['mc_score'].append(m_score)
            res['ai_prob'].append(a_res['prob_treat'])
            res['age_group'].append(p.observation.get('Age_Group', 'Unknown'))
        return res

    def _analyze_robustness(self, get, obs, mc, ai, n, steps, max_noise, ai_thresh):
        noise_levels = np.linspace(0.0, max_noise, steps)
        history = {'noise': [], 'mc_sens': [], 'ai_sens': []}
        
        for noise in noise_levels:
            d = self._run_simulation(get, obs, mc, ai, n, noise)
            yt = np.array(d['truth'])
            if np.sum(yt==1) == 0: continue
            
            ym = np.array(d['mc_score'])
            mc_sens = np.sum((ym >= 3) & (yt==1)) / np.sum(yt==1)
            
            ya = np.array(d['ai_prob'])
            ai_sens = np.sum((ya >= ai_thresh) & (yt==1)) / np.sum(yt==1)
            
            history['noise'].append(noise)
            history['mc_sens'].append(mc_sens)
            history['ai_sens'].append(ai_sens)
        return history

    def _generate_plots(self, data, robustness, result):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC
        fpr, tpr, _ = roc_curve(data['truth'], data['ai_prob'])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'AI (AUC={roc_auc:.2f})')
        ax1.scatter([1-result.mcisaac_specificity], [result.mcisaac_sensitivity], c='g', label='McIsaac')
        
        # Add McNemar p-value text validation
        ax1.text(0.6, 0.2, f"McNemar p={result.p_value:.1e}", transform=ax1.transAxes)
        
        # Iso-Point
        ai_fpr = 1 - result.ai_specificity
        ai_tpr = result.ai_iso_sensitivity
        ax1.scatter([ai_fpr], [ai_tpr], color='blue', marker='x', s=100, label='Iso-sensitivity Point')
        
        ax1.set_title("ROC Curve")
        ax1.legend(loc='lower right')
        
        # Robustness
        ax2.plot(robustness['noise'], robustness['ai_sens'], label='AI (Causal)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
        ax2.plot(robustness['noise'], robustness['mc_sens'], label='McIsaac (Rule)', color='#ff7f0e', linewidth=2, linestyle='--', marker='^', markersize=4)
        
        ax2.set_xlabel('Latent Env Noise (Prob. of Missing Symptoms)', fontsize=12)
        ax2.set_ylabel('Sensitivity', fontsize=12)
        ax2.set_title("Robustness Analysis (MNAR)", fontsize=14)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize=10)
        
        self.save_plot(fig, "standard_report.png")
        
        # Save robustness data
        pd.DataFrame(robustness).to_csv(os.path.join(self.output_dir, "standard_robustness_data.csv"), index=False)

