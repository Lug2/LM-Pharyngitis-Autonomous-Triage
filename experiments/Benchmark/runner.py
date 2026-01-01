
import os
import argparse
import pandas as pd
import logging
import time
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkRunner")

from tasks.standard_bench import StandardBenchmark
from tasks.stress_test import StressTest
from tasks.ablation_study import AblationStudy
from tasks.dca_analysis import DCAAnalysis
from tasks.breaking_point import BreakingPointAnalysis
from tasks.baseline_cnn import ComparativeExperiment

def get_next_run_dir(base_results_dir):
    os.makedirs(base_results_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_results_dir) if d.startswith('run_') and os.path.isdir(os.path.join(base_results_dir, d))]
    
    max_id = 0
    for d in existing:
        try:
            num = int(d.split('_')[1])
            if num > max_id:
                max_id = num
        except:
            pass
            
    next_id = max_id + 1
    new_dir = os.path.join(base_results_dir, f"run_{next_id:03d}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Suite Runner")
    parser.add_argument('--task', type=str, default='all', choices=['all', 'standard', 'stress', 'ablation', 'dca', 'breaking_point', 'comparative'], help='Which task to run')
    parser.add_argument('--n_samples', type=int, default=None, help='Override N samples (e.g. 500)')
    parser.add_argument('--noise', type=float, default=None, help='Override max noise level (e.g. 0.8)')
    parser.add_argument('--steps', type=int, default=None, help='Override robustness steps (e.g. 10)')
    parser.add_argument('--rob_samples', type=int, default=None, help='Override robustness samples per step (e.g. 200)')
    args = parser.parse_args()
    
    # Pack overrides
    overrides = {}
    if args.n_samples is not None: overrides['n_samples'] = args.n_samples
    if args.noise is not None: overrides['noise'] = args.noise
    if args.steps is not None: overrides['steps'] = args.steps
    if args.rob_samples is not None: overrides['rob_samples'] = args.rob_samples

    # 1. Setup Output Directory with Numbering
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, "results")
    run_dir = get_next_run_dir(results_root)
    
    logger.info(f"=== Starting Benchmark Suite [Run ID: {os.path.basename(run_dir)}] ===")
    logger.info(f"Results will be saved to: {run_dir}")
    if overrides:
        logger.info(f"Configuration Overrides: {overrides}")
    
    metrics_summary = {}
    
    # 2. Define Tasks
    tasks = []
    if args.task in ['all', 'standard']:
        tasks.append(StandardBenchmark(run_dir, config_overrides=overrides))
    if args.task in ['all', 'stress']:
        tasks.append(StressTest(run_dir, config_overrides=overrides))
    if args.task in ['all', 'ablation']:
        tasks.append(AblationStudy(run_dir, config_overrides=overrides))
    if args.task in ['all', 'dca']:
        # DCA depends on Stress output if running all, 
        # but the task handles fallback if file not found.
        # Order matters: inputs from Stress might be needed.
        tasks.append(DCAAnalysis(run_dir, config_overrides=overrides))
    if args.task in ['all', 'breaking_point']:
        tasks.append(BreakingPointAnalysis(run_dir, config_overrides=overrides))
    if args.task in ['all', 'comparative']:
        tasks.append(ComparativeExperiment(run_dir, config_overrides=overrides))
        
    # 3. Run Tasks
    for task in tasks:
        logger.info(f"--- Running Task: {task.name} ---")
        try:
            start_time = time.time()
            result = task.run()
            duration = time.time() - start_time
            
            # Prefix keys with task name for safety
            for k, v in result.items():
                metrics_summary[k] = v
                
            metrics_summary[f"{task.name.lower()}_duration"] = round(duration, 2)
            logger.info(f"Task {task.name} completed in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Task {task.name} Failed: {e}", exc_info=True)
            metrics_summary[f"{task.name}_error"] = str(e)
            
    # 4. Save Unified Report
    summary_path = os.path.join(run_dir, "unified_summary.csv")
    
    # Convert dict to simple 2-col CSV (Key, Val)
    df = pd.DataFrame(list(metrics_summary.items()), columns=['Metric', 'Value'])
    df.to_csv(summary_path, index=False)
    
    logger.info(f"=== Benchmark Completed. Summary: {summary_path} ===")
    
if __name__ == "__main__":
    main()

