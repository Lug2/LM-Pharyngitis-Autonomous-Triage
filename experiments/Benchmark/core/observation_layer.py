
from typing import Any, Dict
import random
from .data_schema import PatientRecord

class ObservationLayer:
    def __init__(self):
        # Vulnerability Hierarchy (beta_f)
        # Higher beta = Higher vulnerability to Latent Environmental Factor (L_env)
        self.VULNERABILITY = {
            'V_color': 1.2,   
            'V_vessel': 0.8,  
            'V_white': 0.5,   
            'C_rash': 1.0,
            'C_lymph': 0.2,   
        }
        self.BASE_NOISE_SCALE = 0.0 

    def apply(self, patient: PatientRecord, base_noise: float = 0.0) -> PatientRecord:
        """
        Applies MNAR (Missing Not At Random) degradation based on Latent Environmental Variable L_env.
        
        Mechanism:
        L_env ~ U(0,1)
        P(Obs_f = None | L_env, beta_f) = 1 if L_env < (base_noise * beta_f)
        """
        # 1. Sample Latent Environmental Factor (L_env)
        # Represents unmeasured structural quality (lighting, patient compliance, etc.)
        l_env = random.random()
        patient.q_env = l_env # Keeping legacy field name on schema but logically it is L_env
        
        obs = {}
        gt = patient.ground_truth
        
        # 2. Apply Structural Missingness (MNAR)
        for key, val in gt.items():
            if key in self.VULNERABILITY:
                beta = self.VULNERABILITY[key]
                # Age Interaction (MNAR)
                # Children are harder to examine (lower observability)
                age_modifier = 1.0
                if gt.get('Age_Group') == 'Child':
                    age_modifier = 1.5
                
                threshold = base_noise * beta * age_modifier
                
                # If Latent Env is below threshold -> Missing
                if l_env < threshold:
                    obs[key] = None
                else:
                    obs[key] = val
            elif key.startswith('C_') or key.startswith('V_'):
                obs[key] = val
            elif key in ['Age_Group', 'C_epidemic']:
                obs[key] = val
            else:
                pass
                
        patient.observation = obs
        return patient

    def apply_latent_confounder(self, patient: PatientRecord, rate: float):
        """
        Scenario B: Latent Confounder (NSAIDs) Intervention.
        
        Concept:
        L_NSAIDs is a hidden parent of C_temp and C_pain_sev.
        It is NOT recorded in the dataset (Latent).
        If Active, it suppresses the symptoms.
        
        Implementation:
        Sample L_NSAIDs ~ Bernoulli(rate).
        If True, modify the OBSERVATION (and effectively the phenotype) to masked state.
        """
        if random.random() < rate:
            # Latent Variable L_NSAIDs is ACTIVE
            # Causal Intervention: Set Child Nodes
            
            # 1. Mask Temperature
            if patient.observation.get('C_temp') == 'High':
                patient.observation['C_temp'] = 'Mild' # Milder presentation via API/Drug
            
            # 2. Mask Pain
            if patient.observation.get('C_pain_sev') == 'Severe':
                patient.observation['C_pain_sev'] = 'Mild'

    def apply_systematic_bias(self, patient: PatientRecord, rate: float, target_node: str = 'V_white'):
        """
        Scenario C: Systematic Bias via Miss Node Architecture.
        
        Concept:
        M_bias is a binary node representing a systematic process failure (e.g. Doctor Skip).
        If M_bias is Active, the target node becomes Missing (None).
        
        Implementation:
        Sample M_bias ~ Bernoulli(rate).
        If True, Observation[target] = None.
        """
        if random.random() < rate:
            # Miss Node Active
            if target_node in patient.observation:
                patient.observation[target_node] = None

