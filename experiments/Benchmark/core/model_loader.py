
import sys
import os

class ModelLoader:
    @staticmethod
    def load_causal_brain(config_path: str):
        """
        Loads CausalBrainV6 from the inference_model directory.
        Handles sys.path modification safely.
        """
        # Resolve absolute path to model config
        if not os.path.isabs(config_path):
            # If relative, it's usually relative to the caller or config file?
            # Let's assume it is absolute or correctly resolved by caller.
            # But if passed from our configs (../../...), we must resolve it relative to the config file location
            # OR the caller script location.
            # Best practice: Caller resolves to absolute path.
            config_path = os.path.abspath(config_path)

        # Infer inference_model directory from config path
        # Assuming config is in .../inference_model/model_config.yaml
        inference_dir = os.path.dirname(config_path)
        project_root = os.path.dirname(inference_dir)
        
        # Add project root to sys.path to allow `from src...`
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        try:
            from src.causal_brain_v6 import CausalBrainV6
            return CausalBrainV6(config_path)
        except ImportError as e:
            # Try appending inference_dir directly if module structure is different
            if inference_dir not in sys.path:
                sys.path.append(inference_dir)
            try:
                from causal_brain_v6 import CausalBrainV6
                return CausalBrainV6(config_path)
            except ImportError:
                raise ImportError(f"Could not import CausalBrainV6. Verified paths: {sys.path}") from e

