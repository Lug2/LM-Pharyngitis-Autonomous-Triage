
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from core.suite import BenchmarkTask
from core.evaluation import calculate_net_benefit

class DCAAnalysis(BenchmarkTask):
    @property
    def name(self):
        return "DCA"

    def run(self) -> dict:
        config = self.load_config("dca_config.yaml")
        
        # Locate input data
        stress_file = os.path.join(self.output_dir, "stress", "stress_detailed.csv")
        if os.path.exists(stress_file):
            input_path = stress_file
        else:
            input_path = os.path.abspath(os.path.join(self.config_dir, config['dca_analysis']['input_file']))
            
        if not os.path.exists(input_path):
            print(f"DCA Warning: Input file not found {input_path}")
            return {}

        df = pd.read_csv(input_path)
        filter_sc = config['dca_analysis']['input_filter']['scenario_id']
        df = df[df['scenario_id'] == filter_sc]
        
        if df.empty:
            return {'dca_status': 'No Data'}
            
        # Calc DCA
        thresholds = np.arange(0.01, 1.00, 0.01)
        results = []
        
        y_true = df['gt_gas'].values
        ai_probs = df['ai_prob'].values
        mc_scores = df['mc_score'].values
        
        # Calibrate McIsaac (Simple mapping)
        risk_map = df.groupby('mc_score')['gt_gas'].mean().to_dict()
        mc_probs = df['mc_score'].map(risk_map).fillna(0).values
        
        treat_all = np.ones_like(y_true)
        
        for t in thresholds:
            results.append({
                'Threshold': t,
                'NB_AI': calculate_net_benefit(y_true, ai_probs, t),
                'NB_McIsaac': calculate_net_benefit(y_true, mc_probs, t),
                'NB_TreatAll': calculate_net_benefit(y_true, treat_all, t)
            })
            
        df_res = pd.DataFrame(results)
        self.save_csv(df_res, "dca_curves.csv")
        
        # --- Visualization ---
        self._plot_dca(df_res)
        
        return {'dca_status': 'Success', 'dca_baseline_n': len(df)}

    def _plot_dca(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df['Threshold'], df['NB_AI'], label='AI Model', linewidth=2)
        plt.plot(df['Threshold'], df['NB_McIsaac'], label='McIsaac Score', linestyle='--')
        plt.plot(df['Threshold'], df['NB_TreatAll'], label='Treat All', linestyle=':', color='gray')
        plt.axhline(0, color='black', linewidth=0.8)
        
        plt.xlim(0, 1.0)
        plt.ylim(bottom=-0.05, top=max(df['NB_AI'].max(), 0.1)+0.05)
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title('Decision Curve Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt.gcf(), "dca_plot.png")
        plt.close()

