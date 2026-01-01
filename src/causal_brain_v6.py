from typing import Dict, Any, List
import logging
import copy
from .causal_brain_loader import CausalBrainLoader
from .reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)

class CausalBrainV6(CausalBrainLoader):
    """
    CausalBrainV6: Cognitive Conflict & Triage Guardrail
    Adds meta-cognition by splitting inference into Subjective and Objective streams
    to detect discrepancies (Over-reporting, Silent Danger).
    Integrates ReasoningEngine for full explanation generation.
    """
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.node_modality = self._build_modality_map()
        self.explainer = ReasoningEngine(self.config)
        
        # Load Thresholds
        thresh = self.config.get('thresholds', {})
        self.tau_silent = thresh.get('tau_silent', 0.419)
        self.tau_over = thresh.get('tau_over', 0.25)
        self.silent_prob_limit = thresh.get('silent_prob_limit', 0.6)
        # Age-Adaptive Dynamic Thresholding (AADT)
        self.gas_alert_threshold = thresh.get('gas_alert', 0.70)
        self.pediatric_threshold = thresh.get('pediatric_gas_threshold', 0.50)
        
        # Hardcoded map to align with Loader/V5 legacy hardcoding.
        # Ideally this should be in config, but for now we follow existing pattern.
        self.key_map = {
            'age_group': 'Age_Group',
            'v_white': 'V_white', 
            'v_vessel': 'V_vessel',
            'v_color': 'V_color',
            'c_temp': 'C_temp',
            'c_cough': 'C_cough',
            'c_eye': 'C_eye',
            'c_lymph': 'C_lymph',
            'c_pain_lat': 'C_pain_lat',
            'c_fatigue': 'C_fatigue',
            'c_epidemic': 'C_epidemic',
            'c_joint': 'C_joint',
            'c_duration': 'C_duration',
            'c_pain_sev': 'C_pain_sev',
            'c_onset': 'C_onset',
            'c_rash': 'C_rash' # Added missing mapping
        }

    def _build_modality_map(self) -> Dict[str, str]:
        """Reads 'modality' from node meta in config."""
        mapping = {}
        for node in self.config['nodes']:
            if 'meta' in node and 'modality' in node['meta']:
                mapping[node['name']] = node['meta']['modality']
        return mapping

    def infer_and_explain(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference and provides explanation with AADT (Age-Adaptive Dynamic Thresholding).
        """
        # 1. Infer (Standard Loader logic)
        result = super().infer(evidence)
        probs = result['probs']
        
        if not probs:
            return result # Error case
            
        # 2. Prepare normalized evidence for Explainer
        nodes_map = {n['name']: n for n in self.config['nodes']}
        mapped_evidence = {}
        
        for k, v in evidence.items():
            if v is None: continue
            node_name = self.key_map.get(k, k)
            if node_name not in nodes_map: continue
            
            states = nodes_map[node_name]['states']
            if isinstance(v, int) and 0 <= v < len(states):
                mapped_evidence[node_name] = states[v]
            elif isinstance(v, str) and v in states:
                mapped_evidence[node_name] = v
            elif isinstance(v, bool):
                 mapped_evidence[node_name] = str(v)
            else:
                 mapped_evidence[node_name] = v

        # 3. Explain
        explanation = self.explainer.generate(probs, mapped_evidence)
        
        # 4. Integrate Alerts
        combined_alerts = list(set(result.get('alert', []) + explanation.get('alerts', [])))
        explanation['alerts'] = combined_alerts
        
        winner = max(probs, key=probs.get)
        
        # --- Age-Adaptive Dynamic Thresholding (AADT) ---
        # Determine strictness based on patient age to prevent missed cases in children.
        age_group = mapped_evidence.get('Age_Group', 'Adult') # Default to Adult if unknown
        gas_prob = probs.get('GAS', 0.0)
        
        dynamic_threshold = self.gas_alert_threshold
        is_pediatric = (age_group == 'Child')
        
        if is_pediatric:
            dynamic_threshold = self.pediatric_threshold
            
        clinical_decision = "Monitor"
        if gas_prob >= dynamic_threshold:
            clinical_decision = "Treat (GAS)"
        elif probs.get('Fuso', 0) >= 0.20: # Fuso specific rule (simplified)
            clinical_decision = "Treat (Fuso)"
            
        # Add AADT info to result
        return {
            "diagnosis": winner,
            "probability": probs[winner],
            "probs": probs,
            "explanation": explanation,
            "clinical_decision": {
                "decision": clinical_decision,
                "threshold_used": dynamic_threshold,
                "is_pediatric_mode": is_pediatric,
                "logic": "Age-Adaptive Dynamic Thresholding (AADT)"
            }
        }

    def infer_cognitive(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs Cognitive Inference:
        1. Full Inference (Diagnosis)
        2. Subjective Stream (Patient Story)
        3. Objective Stream (Doctor View)
        4. Measure Conflict & Triage
        """
        # 1. Base Inference (Winner Determination)
        base_result = self.infer_and_explain(evidence)
        if "error" in base_result or not base_result.get('probs'):
            return base_result

        winner_diagnosis = base_result['diagnosis']
        
        # 2. Split Evidence
        ev_sub = {} # Context + Subjective
        ev_obj = {} # Context + Objective
        
        for k, v in evidence.items():
            node_name = self.key_map.get(k, k)
            modality = self.node_modality.get(node_name, None)
            
            if modality == 'context':
                ev_sub[k] = v
                ev_obj[k] = v
            elif modality == 'subjective':
                ev_sub[k] = v
            elif modality == 'objective':
                ev_obj[k] = v
            else:
                # Fallback: if not defined, maybe add to both or just ignore?
                # Default behavior: ignore if not explicitly categorized? 
                # Let's add to both to be safe for now if important.
                pass

        # 3. Stream Inference
        # We use self.infer(raw_evidence) which returns {'probs': {Pathogen: prob, ...}}
        res_sub = self.infer(ev_sub)
        res_obj = self.infer(ev_obj)
        
        p_sub = res_sub['probs'].get(winner_diagnosis, 0.0)
        p_obj = res_obj['probs'].get(winner_diagnosis, 0.0)
        
        # 4. Conflict Calculation
        conflict_score = abs(p_sub - p_obj)
        triage_type = "Pattern A: Consistent"
        latent_risk = None
        
        alerts = base_result['explanation'].get('alerts', [])
        
        # Pattern B: Over-Reporting (Sub >>> Obj)
        if conflict_score >= self.tau_over and p_sub > p_obj and p_obj < 0.5:
            triage_type = "Pattern B: Over-Reporting"
            alerts.append("⚠️ **Notice**: Objective findings are mild compared to subjective symptoms. (Confidence Gap: {:.2f})".format(conflict_score))
            
        # Pattern C: Silent Danger (Obj >>> Sub)
        elif conflict_score >= self.tau_silent and p_obj > p_sub:
            triage_type = "Pattern C: Silent Danger"
            if p_obj > self.silent_prob_limit: 
                alerts.insert(0, "⚠️ **WARNING**: Dangerous objective signs detected despite mild subjective symptoms. Risk of missed diagnosis. (Silent Danger)")
        
        # Logic 2: Shadow Check (Safety Guardrail)
        if triage_type != "Pattern C: Silent Danger":
            dangerous_pathogens = ['GAS', 'Fuso']
            for d in dangerous_pathogens:
                if d == winner_diagnosis: continue 
                
                p_obj_d = res_obj['probs'].get(d, 0.0)
                p_sub_d = res_sub['probs'].get(d, 0.0)
                
                # Use Calibrated Thresholds
                # Detect if Objective strongly indicates Danger while Subjective misses it
                if p_obj_d > self.silent_prob_limit and (p_obj_d - p_sub_d) > self.tau_silent:
                     triage_type = "Pattern C: Silent Danger (Latent)"
                     alerts.insert(0, f"⚠️ **WARNING**: Diagnosis is {winner_diagnosis}, but objective signs strongly suggest {d}. (Obj: {p_obj_d:.2f})")
                     latent_risk = d
                     break

        # Inject into result
        base_result['explanation']['alerts'] = alerts
        base_result['cognitive'] = {
            'conflict_score': round(conflict_score, 4),
            'prob_subjective': round(p_sub, 4),
            'prob_objective': round(p_obj, 4),
            'triage_type': triage_type
        }
        
        if latent_risk:
            base_result['cognitive']['latent_risk'] = latent_risk
            
        return base_result

    def diagnose(self, evidence: Dict[str, Any], enable_safety_net: bool = True) -> Dict[str, Any]:
        """
        Public API for diagnosis with Ablation Control.
        
        Args:
            evidence: Diagnostic evidence
            enable_safety_net: If True, runs full Cognitive Inference (Conflict/SafetyNet).
                               If False, runs Base Bayesian Inference (No SafetyNet).
        """
        if enable_safety_net:
            return self.infer_cognitive(evidence)
        else:
            return self.infer_and_explain(evidence)
