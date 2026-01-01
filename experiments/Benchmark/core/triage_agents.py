
from typing import Dict, Any
from .data_schema import PatientRecord
from .model_loader import ModelLoader

class McIsaacAgent:
    def __init__(self):
        pass

    def predict(self, record: PatientRecord) -> int:
        """
        Calculates Modified Centor (McIsaac) Score (0-4+)
        Input: record.observation (dict)
        Handling 'None': Treated as 0 (Negative/Absent)
        """
        obs = record.observation
        score = 0
        
        # 1. Temperature > 38
        temp = obs.get('C_temp')
        if temp == 'High':
            score += 1
            
        # 2. No Cough
        cough = obs.get('C_cough')
        if cough == 'Absent':
            score += 1
            
        # 3. Tender Anterior Cervical Adenopathy
        lymph = obs.get('C_lymph')
        if lymph in ['Anterior', 'Both']:
            score += 1
            
        # 4. Tonsillar Swelling or Exudate
        white = obs.get('V_white')
        if white in ['Low', 'High']:
            score += 1
            
        # 5. Age Modification
        age = obs.get('Age_Group')
        if age == 'Child': # 3-14
            score += 1
        elif age == 'YoungAdult': # 15-44 (15-30 in YAML)
            score += 0 
        elif age in ['Adult', 'Senior']:
            score -= 1
            
        return score

class CausalBrainAgent:
    def __init__(self, config_path: str):
        self.brain = ModelLoader.load_causal_brain(config_path)
    
    def predict(self, record: PatientRecord) -> Dict[str, Any]:
        """
        Returns: {
            'prob_gas': float,
            'conflict_score': float,
            'is_silent_danger_alert': bool
        }
        """
        obs = record.observation
        result = self.brain.infer_cognitive(obs)
        
        prob_gas = result['probs'].get('GAS', 0.0)
        prob_fuso = result['probs'].get('Fuso', 0.0)
        
        # Treatment Probability
        prob_treat = min(prob_gas + prob_fuso, 1.0)
        
        # Cognitive Info
        cognitive = result.get('cognitive', {})
        conflict = cognitive.get('conflict_score', 0.0)
        triage = cognitive.get('triage_type', '')
        
        return {
            'prob_gas': prob_gas,
            'prob_fuso': prob_fuso,
            'prob_treat': prob_treat,
            'conflict_score': conflict,
            'is_silent_danger_alert': 'Silent Danger' in triage,
            'diagnosis': result.get('diagnosis'),
            'triage_type': triage
        }

