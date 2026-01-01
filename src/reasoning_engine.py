from typing import Dict, List, Any

class ReasoningEngine:
    def __init__(self, yaml_config: Dict):
        # Build optimized lookup for meta data
        # node_name -> {states -> {label, supports}}
        self.meta_data = {}
        if 'nodes' in yaml_config:
            for node in yaml_config['nodes']:
                if 'meta' in node:
                    self.meta_data[node['name']] = node['meta']
        
        # Load Thresholds
        thresh = yaml_config.get('thresholds', {})
        self.gas_alert = thresh.get('gas_alert', 0.70)
        self.fuso_alert = thresh.get('fuso_alert', 0.05)

    def generate(self, diagnosis_result: Dict[str, float], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates an explanation for the diagnosis.
        
        Args:
            diagnosis_result: Dict of pathogen probabilities {'GAS': 0.1, ...}
            evidence: Dict of input evidence {'C_lymph': 'Posterior', ...}
        
        Returns:
            Dict with 'summary', 'positive', 'negative', 'alerts' keys.
        """
        # Identify Winner
        winner = max(diagnosis_result, key=diagnosis_result.get)
        prob = diagnosis_result[winner]
        
        explanation = {
            "summary": f"The most likely cause is {winner} (Probability: {prob:.2%}).",
            "positive": [],
            "negative": [],
            "alerts": []
        }

        # Scan Evidence
        for node_name, state in evidence.items():
            # Skip if unknown node or state (or if state is not in meta)
            node_meta = self.meta_data.get(node_name)
            if not node_meta: continue
            
            # Evidence state might be mapped string. Assuming evidence is normalized to string names (e.g. "Posterior")
            # If evidence is int (raw API), we might have issues. CausalBrainV5 should ensure it calls this with mapped names.
            # But the loader's infer() method consumes mapped names internally but accepts raw. 
            # We need the String state here.
            
            state_meta = node_meta['states'].get(state)
            if not state_meta: continue
            
            label_jp = state_meta.get('label', state)
            supports = state_meta.get('supports', {})
            
            support_level = supports.get(winner)
            
            # A. Confirming Evidence (High)
            if support_level == 'High':
                explanation['positive'].append(
                    f"{state} is observed, which strongly matches the characteristics of {winner}."
                )
            
            # B. Consistent Evidence (Medium)
            elif support_level == 'Medium':
                 # Often too verbose to list all "Consistent" items? 
                 # Spec says: "Consistent... {state} match consistent"
                 # Let's include them for now but maybe limit them if too noisy.
                 # Spec example: "発熱... 整合します"
                 explanation['positive'].append(
                    f"{state} is consistent with {winner}."
                 )
            
            # C. Conflict / Ruling Out (Winner is Low, but others might be High?)
            elif support_level == 'Low':
                # Check if it supports others strongly?
                # Using specific phrasing from Spec
                # "While {state} is seen, derived {winner} is higher..."
                explanation['negative'].append(
                    f"Although {state} is observed, the overall probability suggests {winner} is more likely."
                )
        
        # Deduplicate list
        explanation['positive'] = list(dict.fromkeys(explanation['positive']))
        explanation['negative'] = list(dict.fromkeys(explanation['negative']))

        # D. Alerts & Recommendations (V6.5)
        
        # 1. Flu Recommendation
        if winner == 'Flu':
            explanation['alerts'].append(
                "💡 **Recommendation**: Influenza is strongly suspected. "
                "Anti-influenza drugs may be effective if taken within 48 hours of onset. "
                "Check for local outbreaks and consider visiting a clinic for testing."
            )

        # 2. GAS Recommendation
        if winner == 'GAS':
            gas_prob = diagnosis_result.get('GAS', 0.0)
            if gas_prob >= self.gas_alert:
                explanation['alerts'].append(
                    "💡 **Recommendation**: Group A Streptococcus (GAS) is strongly suspected (Exceeds Threshold). "
                    "Antibiotics are required to prevent complications like Rheumatic Fever. "
                    "Please get a rapid test at a clinic immediately."
                )
            else:
                explanation['alerts'].append(
                    "💡 **Recommendation**: Suspicion of Group A Streptococcus (GAS). "
                    "Medical consultation is recommended just in case."
                )

        # 3. EBV Recommendation
        if winner == 'EBV':
            explanation['alerts'].append(
                "ℹ️ **Note**: Possibility of Infectious Mononucleosis (EBV). "
                "Prescribing Penicillin (e.g., Amoxicillin) may cause a rash. "
                "Please inform your doctor of this result."
            )

        # 4. Fuso Risk (Unilateral Pain)
        if evidence.get('C_pain_lat') == 'Unilateral':
             explanation['alerts'].append("Unilateral pain detected. Consider risk of Peritonsillar Abscess or Lemierre's Syndrome.")
        
        # 5. Adeno Eye (Conjunctivitis)
        if evidence.get('C_eye') == 'Conjunctivitis':
             explanation['alerts'].append("Conjunctivitis detected. High likelihood of Adenovirus (Pharyngoconjunctival Fever).")
             
        # 6. Lemierre (Risk Check)
        # Even if not winner, if Fuso prob > fuso_alert, warn.
        fuso_prob = diagnosis_result.get('Fuso', 0.0)
        if winner == 'Fuso' or fuso_prob >= self.fuso_alert:
            explanation['alerts'].append(f"⚠️ **Warning**: Potential Fusobacterium infection (Prob: {fuso_prob:.1%}). Watch for Internal Jugular Vein Thrombosis (Lemierre's Syndrome).")

        return explanation
