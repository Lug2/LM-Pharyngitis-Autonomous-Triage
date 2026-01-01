import os
from .causal_brain_v6 import CausalBrainV6

def test_red_flags():
    print("Testing Red Flag Logic...")
    # Initialize with absolute path or relative from root
    config_path = os.path.join(os.path.dirname(__file__), 'model_config.yaml')
    ai = CausalBrainV6(config_path)
    
    # Test 1: Normal Case (Should proceed to inference)
    ev_normal = {'C_pain_lat': 'Bilateral', 'C_pain_sev': 'Mild'}
    res = ai.diagnose(ev_normal)
    assert res['diagnosis'] != 'Immediate Referral', f"Normal case failed: {res['diagnosis']}"
    print("✅ Normal Case Passed")
    
    # Test 2: Trismus (Epiglottitis)
    ev_risk1 = {'Trismus': 'Present'}
    res = ai.diagnose(ev_risk1)
    assert res['diagnosis'] == 'Immediate Referral', "Trismus failed to trigger referral"
    assert "Epiglottitis" in res['explanation']['summary'], f"Summary mismatch: {res['explanation']['summary']}"
    assert any("Trismus" in a for a in res['explanation']['alerts']), "Alerts missing Trismus"
    print("✅ Trismus Check Passed")
    
    # Test 3: Unilateral Severe Pain (Abscess)
    ev_risk2 = {'C_pain_lat': 'Unilateral', 'C_pain_sev': 'Severe'}
    res = ai.diagnose(ev_risk2)
    assert res['diagnosis'] == 'Immediate Referral', "Unilateral Severe Pain failed to trigger referral"
    assert "Peritonsillar Abscess" in res['explanation']['summary'], f"Summary mismatch: {res['explanation']['summary']}"
    assert any("Unilateral Severe" in a for a in res['explanation']['alerts']), "Alerts missing Unilateral Severe"
    print("✅ Abscess Check Passed")

    print("All Red Flag checks passed successfully.")

if __name__ == "__main__":
    test_red_flags()
