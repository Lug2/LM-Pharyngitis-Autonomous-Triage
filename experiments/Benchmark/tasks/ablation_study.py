
import pandas as pd
import logging
from core.suite import BenchmarkTask
from core.triage_agents import CausalBrainAgent
from core.data_schema import PatientRecord

class AblationStudy(BenchmarkTask):
    @property
    def name(self):
        return "Ablation"

    def run(self) -> dict:
        config = self.load_config("ablation_config.yaml")
        model_path = self.get_model_config_path(config)
        brain = CausalBrainAgent(model_path)
        
        cases = []
        scenarios = config['ablation_study']['scenarios']
        
        if 'carrier_cases' in scenarios:
            cases.extend(self._load_cases(scenarios['carrier_cases']))
        if 'nsaid_cases' in scenarios:
            cases.extend(self._load_cases(scenarios['nsaid_cases']))
            
        results = []
        for case in cases:
            # Construct dummy PatientRecord from config evidence
            rec = PatientRecord(
                id=0,
                ground_truth=case['evidence'], 
                etiology=case['ground_truth'], 
                is_leakage=False, 
                q_env=1.0, 
                observation=case['evidence'] # Evidence is treated as observation
            )
            
            # Predict
            res = brain.predict(rec)
            
            results.append({
                'Scenario': case['scenario'],
                'GroundTruth': case['ground_truth'],
                'AI_Prob': res['prob_treat'],
                'Conflict': res['conflict_score'],
                'Triage': res['triage_type']
            })
            
        df = pd.DataFrame(results)
        self.save_csv(df, "ablation_results.csv")
        return {'ablation_cases': len(results)}

    def _load_cases(self, conf):
        cases = []
        for i in range(conf.get('n_samples', 10)):
            cases.append({
                'scenario': conf.get('scenario_id'),
                'ground_truth': conf.get('ground_truth'),
                'evidence': conf.get('evidence').copy()
            })
        return cases

