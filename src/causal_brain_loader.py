import yaml
import logging
import itertools
import numpy as np
from typing import Dict, List, Any
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CausalBrainLoader:
    def __init__(self, config_path: str):
        self.config = self._load_yaml(config_path)
        self.model = self._build_model()
        
        if not self.model.check_model():
             # Basic check_model might not catch sum errors if pgmpy normalizes, but it catches structure issues
             raise ValueError("Generated model failed pgmpy validation check.")
        
        self.infer_engine = VariableElimination(self.model)
        logger.info(f"CausalBrainLoader: Model loaded from {config_path}")

    def _load_yaml(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_model(self) -> DiscreteBayesianNetwork:
        model = DiscreteBayesianNetwork()
        
        # 1. Add Nodes
        nodes_config = {n['name']: n for n in self.config['nodes']}
        for node_conf in self.config['nodes']:
            model.add_node(node_conf['name'])
        
        # 2. Add Edges
        for edge in self.config['edges']:
            model.add_edge(edge['parent'], edge['child'])
            
        # 3. Build CPTs
        for cpt_conf in self.config['cpts']:
            node_name = cpt_conf['node']
            node_states = nodes_config[node_name]['states']
            
            if cpt_conf['type'] == 'root':
                cpd = self._build_root_cpd(node_name, node_states, cpt_conf['probs'])
            elif cpt_conf['type'] == 'conditional':
                parents = cpt_conf['parents']
                parent_states = {p: nodes_config[p]['states'] for p in parents}
                cpd = self._build_conditional_cpd(node_name, node_states, parents, parent_states, cpt_conf['rules'])
            else:
                raise ValueError(f"Unknown CPT type: {cpt_conf['type']}")
            
            model.add_cpds(cpd)
            
        return model

    def _build_root_cpd(self, name: str, states: List[str], probs_map: Dict[str, float]) -> TabularCPD:
        values = []
        for s in states:
            if s not in probs_map:
                raise ValueError(f"Node {name} missing prob for state {s}")
            values.append(probs_map[s])
        
        # Normalize just in case (though YAML should be correct)
        total = sum(values)
        if abs(total - 1.0) > 0.01:
             raise ValueError(f"Probabilities for {name} do not sum to 1.0 ({total})")
        
        state_names = {name: states}
        return TabularCPD(variable=name, variable_card=len(states), values=[[v] for v in values], state_names=state_names)

    def _build_conditional_cpd(self, 
                               name: str, 
                               states: List[str], 
                               parents: List[str], 
                               parent_states_map: Dict[str, List[str]], 
                               rules: List[Dict]) -> TabularCPD:
        
        # 1. Generate Cartesian Product of Parent States
        # pgmpy expects evidence order to match list of parents
        parent_cards = [len(parent_states_map[p]) for p in parents]
        parent_state_lists = [parent_states_map[p] for p in parents]
        
        # Product creates combinations like (Child, True), (Child, False)...
        combinations = list(itertools.product(*parent_state_lists))
        
        # 2. For each combination, find the matching rule
        # Result matrix should be: [ [prob_state0_comb0, prob_state0_comb1...], [prob_state1_comb0...] ]
        # BUT pgmpy TabularCPD `values` arg expects: list of lists, where outer list is per STATE of CHILD node.
        # Inner list is probabilities for that state across all parent combinations.
        
        # Initialize result columns (one column per combination)
        columns_probs = [] # Will store a dict {State: Prob} for each combination

        for comb in combinations:
            # Create a context dict for checking rules: {'Age_Group': 'Child', 'GAS': 'True'}
            # Note: states in YAML are strings like 'True'/'False'. Python bools might need str conversion if loaded as such.
            context = {p: str(s) for p, s in zip(parents, comb)}
            
            matched_probs = None
            
            for rule in rules:
                # Check "if" condition
                if 'if' in rule:
                    condition = rule['if']
                    # Check if all conditions match context
                    match = True
                    for k, v in condition.items():
                        # Handle boolean conversion if YAML loaded as bool
                        val_str = str(v)
                        if context[k] != val_str:
                            match = False
                            break
                    if match:
                        matched_probs = rule['probs']
                        break # Stop at first match (ordered rules)
                elif 'default' in rule:
                    matched_probs = rule['probs']
                    # Default is explicitly checked last or as fallback if ordered correctly. 
                    # If this loop finds a default, it means no previous 'if' matched.
                    break
            
            if matched_probs is None:
                raise ValueError(f"No rule matched for node {name} with context {context}")
            
            columns_probs.append(matched_probs)

        # 3. Convert columns to pgmpy values format
        # values[i][j] = Probability of ChildState[i] given ParentCombination[j]
        cpt_values = []
        for state in states:
            row = []
            for col_prob in columns_probs:
                # Expect exact string match for keys "True"/"False" etc.
                # YAML parsed "True" might be boolean True key.
                # Let's handle string conversion safe access
                
                # Try raw, then string, then bool
                p = col_prob.get(state)
                if p is None and state == 'True': p = col_prob.get(True)
                if p is None and state == 'False': p = col_prob.get(False)
                
                if p is None:
                     raise ValueError(f"Probability for state '{state}' missing in rule for {name}")
                row.append(p)
            cpt_values.append(row)
            
        state_names = {name: states}
        state_names.update(parent_states_map)
        
        return TabularCPD(variable=name, variable_card=len(states), 
                          values=cpt_values,
                          evidence=parents, evidence_card=parent_cards,
                          state_names=state_names)

    def infer(self, evidence: Dict[str, Any]) -> Dict:
        """Mirroring the simplified infer method from V3 but using generic Node names internally."""
        
        # User input might use API keys (v_white) or Node names (V_white).
        # We need a mapper if we want to be robust, but for V4 verification let's assume keys match Node names 
        # OR we reuse the mapper from V3 logic.
        # Let's implement the mapping to be API compliant.
        
        key_map = {
            'age_group': 'Age_Group',
            'v_white': 'V_white',
            'v_vessel': 'V_vessel',
            'v_color': 'V_color',
            'c_temp': 'C_temp',
            'c_cough': 'C_cough',
            'c_eye': 'C_eye',
            'c_lymph': 'C_lymph',
            'c_pain_lat': 'C_pain_lat',
            'c_epidemic': 'C_epidemic',
            'c_joint': 'C_joint',
            'c_duration': 'C_duration',
            'c_pain_sev': 'C_pain_sev',
            'c_rash': 'C_rash',
            'c_onset': 'C_onset',
            'c_fatigue': 'C_fatigue'
        }
        
        # Reverse map for checking if input is already node name? No, just try map.
        cleaned_evidence = {}
        
        # Helper to map integer/enum input to State String
        # Since I am generic, I needs to know the States list from config to map Int -> String
        nodes_map = {n['name']: n for n in self.config['nodes']}
        
        for k, v in evidence.items():
            if v is None: continue
            
            node_name = key_map.get(k, k) # Use map or raw key
            
            if node_name not in nodes_map:
                continue # Unknown key
            
            states = nodes_map[node_name]['states']
            
            # If v is int and index is valid, map to string
            if isinstance(v, int) and 0 <= v < len(states):
                cleaned_evidence[node_name] = states[v]
            elif isinstance(v, str) and v in states:
                cleaned_evidence[node_name] = v
            # If boolean (True/False)
            elif isinstance(v, bool):
                 cleaned_evidence[node_name] = str(v) # mapped to 'True'/'False' string
            else:
                 # Try raw
                 cleaned_evidence[node_name] = v
                 
        target_pathogens = ['GAS', 'EBV', 'Fuso', 'Flu', 'Adeno', 'Other_Viral']
        results = {}
        alerts = []
        
        try:
            for pathogen in target_pathogens:
                posterior = self.infer_engine.query([pathogen], evidence=cleaned_evidence)
                # We need prob of "True"
                # Index of "True". Usually 1 if states are [False, True].
                # Let's find index of 'True'
                p_states = nodes_map[pathogen]['states']
                idx_true = p_states.index(True) if True in p_states else p_states.index('True')
                
                results[pathogen] = posterior.values[idx_true]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {'probs': {}, 'alert': [f"Error: {e}"]}

        # Alert Logic (Hardcoded for now as spec didn't move it to YAML yet, though it should be)
        if results.get('Fuso', 0) > 0.05:
            alerts.append("Lemierre Syndrome Risk")
        if results.get('GAS', 0) > 0.80:
             alerts.append("Highly Infectious (GAS)")

        return {'probs': results, 'alert': alerts}

if __name__ == "__main__":
    # Test load
    loader = CausalBrainLoader("c:/Users/leonm/Desktop/喉診断/inference_model/model_config.yaml")
    print("Model Loaded Successfully.")
    
    test_ev = {'age_group': 0, 'c_temp': 1}
    print("Infer:", loader.infer(test_ev))
