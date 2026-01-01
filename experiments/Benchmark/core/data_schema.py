from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class PatientRecord:
    id: int
    # 1. Ground Truth (Real Biological State)
    ground_truth: Dict[str, Any]  # e.g. {GAS: True, V_white: High, ...}
    etiology: str                 # "PURE_GAS", "CARRIER", "VIRAL", "FUSO", "NON_INFECTIOUS", "CO_INFECTION"
    is_leakage: bool              # If ANY leakage event occurred
    
    # 2. Environment
    q_env: float                  # 0.0 to 1.0
    
    # 3. Observation (Degraded / Visible to Agents)
    # Values can be None (missing)
    observation: Dict[str, Any]   # e.g. {V_white: None, C_temp: High, ...}

    def is_silent_danger(self) -> bool:
        """
        Definition from spec:
        GAS=True AND (Pain=Mild OR Temp=Low) AND Findings=High
        """
        gt = self.ground_truth
        
        # GAS Check
        if not gt.get('GAS') and not gt.get('Fuso'): 
             if not gt.get('GAS'): return False
        
        if not gt.get('GAS'): return False

        # Mild Symptoms Logic
        # Pain=Mild (C_pain_sev=Mild) OR Temp=Low (C_temp in [Normal, Mild])
        pain_mild = (gt.get('C_pain_sev') == 'Mild')
        temp_low = (gt.get('C_temp') in ['Normal', 'Mild'])
        
        if not (pain_mild or temp_low):
            return False
            
        # Findings High Logic
        # Findings=High (V_white=High OR V_color=DarkRed OR V_vessel=Prominent)
        # Assuming V_white='High', V_color='DarkRed'
        findings_high = (
            gt.get('V_white') == 'High' or 
            gt.get('V_color') == 'DarkRed' or
            gt.get('V_vessel') == 'Prominent'
        )
        
        return findings_high

    def is_fuso_case(self) -> bool:
        return self.ground_truth.get('Fuso') is True

@dataclass
class BenchmarkResult:
    scenario: str                 # "Pediatric", "Adult", "All"
    mcisaac_sensitivity: float
    ai_iso_sensitivity: float     # Should be equal to mcisaac
    mcisaac_specificity: float
    ai_specificity: float         # The Winning Metric
    p_value: float = 0.0          
    
    # Additional
    run_date: str = field(default="")
    break_point: Optional[float] = None

