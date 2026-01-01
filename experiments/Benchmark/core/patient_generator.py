
import yaml
import random
import logging
from typing import Dict, List, Any, Tuple
from .data_schema import PatientRecord

logger = logging.getLogger(__name__)

class PatientGenerator:
    def __init__(self, config_path: str, alpha_leak: float = 0.15):
        self.config = self._load_yaml(config_path)
        self.nodes = {n['name']: n for n in self.config['nodes']}
        self.edges = self._build_edges()
        self.ALPHA_LEAK = alpha_leak
        
        # Overrides (Scenario Logic)
        self.cpt_overrides = {} # {NodeName: {Val: Prob, ...}}
        self.mutation_config = {} # {masking_rate: float, bias_miss_rate: float}

    def _load_yaml(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_edges(self) -> Dict[str, List[str]]:
        adj = {n: [] for n in self.nodes}
        for edge in self.config['edges']:
            p, c = edge['parent'], edge['child']
            if p in adj:
                adj[p].append(c)
        return adj
    
    def set_override(self, overrides: Dict[str, Dict[str, float]]):
        """
        Set CPT overrides.
        Example: {'GAS': {'True': 0.5, 'False': 0.5}} to force prevalence/prior.
        Note: This is a simplistic override that ignores parents for the overridden node.
        Effective for Root nodes or forcing a state.
        """
        self.cpt_overrides = overrides
        
    def set_mutation(self, config: Dict[str, Any]):
        """
        Set post-generation mutation config.
        Keys: 'nsaid_masking_rate', 'bias_miss_rate', etc.
        """
        self.mutation_config = config

    def generate(self, n_samples: int = 1000) -> List[PatientRecord]:
        records = []
        for i in range(n_samples):
            records.append(self._generate_single_patient(i))
        return records

    def _generate_single_patient(self, pid: int) -> PatientRecord:
        state = {}
        
        order = [
            'Age_Group', 'C_epidemic', # Roots
            'GAS', 'Fuso', 'EBV', 'Flu', 'Adeno', 'Other_Viral', # Pathogens
            'Inflam', 'Exudate_Gen', # Pathophysiology
            'C_temp', 'C_cough', 'C_lymph', 'C_fatigue', 'C_rash', 'V_vessel',
            'C_joint', 'C_duration', 'C_pain_sev', 'C_eye', 'C_onset', 'C_pain_lat',
            'V_color', 'V_white' # Generated from Inflam/Exudate
        ]
        
        sampling_order = [n for n in order if n in self.nodes]
        
        etiology = None
        is_carrier = False
        leakage_occurred = False
        
        for node in sampling_order:
            if node == 'Inflam' and etiology is None:
                etiology, is_carrier = self._classify_etiology(state)
            
            parents = self._get_parents(node)
            parent_states = {p: state[p] for p in parents}
            
            effective_parent_states = parent_states.copy()
            
            if is_carrier and 'GAS' in parents:
                effective_parent_states['GAS'] = False 
                if random.random() < self.ALPHA_LEAK:
                    effective_parent_states['GAS'] = True
                    leakage_occurred = True
            
            val = self._sample_node(node, effective_parent_states)
            state[node] = val
            
        # --- Mutations (Scenario B/C) ---
        state = self._apply_mutations(state)

        return PatientRecord(
            id=pid,
            ground_truth=state,
            etiology=etiology,
            is_leakage=leakage_occurred,
            q_env=0.0, 
            observation={} 
        )

    def _apply_mutations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mutated = state.copy()
        
        # 1. NSAID Masking
        masking_rate = self.mutation_config.get('nsaid_masking_rate', 0.0)
        if masking_rate > 0 and random.random() < masking_rate:
             # Mask C_temp and C_pain_sev
             if mutated.get('C_temp') == 'High':
                 mutated['C_temp'] = 'Mild' # or Normal
             if mutated.get('C_pain_sev') == 'Severe':
                 mutated['C_pain_sev'] = 'Mild'
                 
        # 2. Observation Bias (Missing White Coating)
        bias_rate = self.mutation_config.get('bias_miss_rate', 0.0)
        if bias_rate > 0 and mutated.get('V_white') == 'High':
             if random.random() < bias_rate:
                 mutated['V_white'] = 'None'
                 
        return mutated

    def _get_parents(self, node: str) -> List[str]:
        for cpt in self.config['cpts']:
            if cpt['node'] == node:
                if 'parents' in cpt:
                    return cpt['parents']
        return []

    def _classify_etiology(self, state: Dict[str, Any]) -> Tuple[str, bool]:
        gas = state.get('GAS')
        fuso = state.get('Fuso')
        viruses = ['EBV', 'Flu', 'Adeno', 'Other_Viral']
        has_viral = any(state.get(v) for v in viruses)
        
        if fuso:
            return "FUSO", False
        if gas:
            if has_viral:
                return "CARRIER", True
            else:
                return "PURE_GAS", False
        if has_viral:
            return "VIRAL", False
        return "NON_INFECTIOUS", False

    def _sample_node(self, node: str, parent_states: Dict[str, Any]) -> Any:
        # Check Override first
        if node in self.cpt_overrides:
            override_dist = self.cpt_overrides[node]
            # Simple weighted choice ignoring parents
            states = list(override_dist.keys())
            weights = list(override_dist.values())
            choice = random.choices(states, weights=weights, k=1)[0]
            if choice == 'True': return True
            if choice == 'False': return False
            return choice

        cpt = next((c for c in self.config['cpts'] if c['node'] == node), None)
        if not cpt:
            return None
            
        matched_probs = None
        
        if cpt['type'] == 'root':
            matched_probs = cpt['probs']
        else:
            for rule in cpt['rules']:
                if 'if' in rule:
                    condition = rule['if']
                    is_match = True
                    for k, v in condition.items():
                        p_val = parent_states.get(k)
                        if str(p_val) != str(v):
                            is_match = False
                            break
                    if is_match:
                        matched_probs = rule['probs']
                        break
            
            if matched_probs is None:
                for rule in cpt['rules']:
                    if 'default' in rule:
                        matched_probs = rule['probs']
                        break
        
        if matched_probs is None:
             # Fallback/Error
             # raise ValueError(f"No rule matched for {node}")
             # Return default state if possible? 
             # Let's assume binary False/Absent/Normal is usually safe fallback
             return None

        states = list(matched_probs.keys())
        weights = list(matched_probs.values())
        
        total = sum(weights)
        if total == 0: total=1
        weights = [w/total for w in weights]
        
        choice = random.choices(states, weights=weights, k=1)[0]
        
        if choice == 'True': return True
        if choice == 'False': return False
        
        return choice

