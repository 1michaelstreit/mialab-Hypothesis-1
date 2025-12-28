"""
Experiment Runner.

This script executes the medical image analysis pipeline for:
1. Both modes: 'forest' and 'deep'
2. All normalization methods: 'z_score', 'min_max', 'percentile', 'histogram_matching', 'none'
"""

import os
import sys
import argparse
import time
import traceback
from types import SimpleNamespace

import pipeline

def run_all_experiments(
    base_result_dir: str,
    data_atlas_dir: str,
    data_train_dir: str,
    data_test_dir: str,
    run_deep: bool = True,
    run_forest: bool = False,
    prepro_only: bool = False
):
    # 1. Define the Test to iterate over
    normalization_methods = [
        'z_score', 
        'min_max', 
        'percentile', 
        'histogram_matching',
        'none'  
    ]

    modes = []
    if run_forest: modes.append('forest')
    if run_deep: modes.append('deep')

    total_experiments = len(modes) * len(normalization_methods)
    current_count = 0

    print("=" * 60)
    print(f"Starting Batch Experiment Runner")
    print(f"Total experiments to run: {total_experiments}")
    print("=" * 60)


    for mode in modes:
        for norm in normalization_methods:
            current_count += 1
            print(f"\n[Experiment {current_count}/{total_experiments}]")
            print(f"Mode:          {mode.upper()}")
            print(f"Normalization: {norm}")
            
            # 3. Create a specific sub-folder for this experiment to keep results organized
            # Structure: ./mia-result/deep/z_score/timestamp/...
            current_result_dir = os.path.join(base_result_dir, mode, norm)
            
            # Ensure the directory exists
            os.makedirs(current_result_dir, exist_ok=True)

            # 4. Construct the 'args' object expected by pipeline.main
            # pipeline.main expects an object with a .norm attribute
            args = SimpleNamespace()
            args.norm = norm
            args.result_dir = current_result_dir
            args.data_atlas_dir = data_atlas_dir
            args.data_train_dir = data_train_dir
            args.data_test_dir = data_test_dir
            args.mode = mode
            args.prepro_only = prepro_only

            # 5. Execute the pipeline
            start_time = time.time()
            try:
                pipeline.main(
                    result_dir=current_result_dir,
                    data_atlas_dir=data_atlas_dir,
                    data_train_dir=data_train_dir,
                    data_test_dir=data_test_dir,
                    mode=mode,
                    args=args
                )
                duration = time.time() - start_time
                print(f"✅ Success! Duration: {duration:.2f} seconds")
                print(f"   Results saved to: {current_result_dir}")

            except Exception as e:
                print(f"❌ Failed!")
                print("-" * 20)
                traceback.print_exc()
                print("-" * 20)
                print("Continuing to next experiment...")


            if mode == 'deep':
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

    print("\n" + "=" * 60)
    print("All experiments completed.")
    print("=" * 60)

if __name__ == "__main__":
    # You can change these paths manually here, or pass them via command line
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths
    default_result_dir = os.path.join(script_dir, "experiment_results")
    default_atlas_dir = os.path.join(script_dir, "mialab/data/atlas")
    default_train_dir = os.path.join(script_dir, "mialab/data/train")
    default_test_dir = os.path.join(script_dir, "mialab/data/test")

    parser = argparse.ArgumentParser(description="Run full suite of experiments.")
    parser.add_argument("--result_dir", type=str, default=default_result_dir)
    parser.add_argument("--data_atlas_dir", type=str, default=default_atlas_dir)
    parser.add_argument("--data_train_dir", type=str, default=default_train_dir)
    parser.add_argument("--data_test_dir", type=str, default=default_test_dir)
    
    # Flags to control what to run
    parser.add_argument("--skip_forest", action="store_true", help="Skip Random Forest experiments")
    parser.add_argument("--skip_deep", action="store_true", help="Skip Deep Learning experiments")
    parser.add_argument("--prepro_only", action="store_true", help="Only run preprocessing and save preprocessed images")

    args = parser.parse_args()

    run_all_experiments(
        base_result_dir=args.result_dir,
        data_atlas_dir=args.data_atlas_dir,
        data_train_dir=args.data_train_dir,
        data_test_dir=args.data_test_dir,
        run_deep=not args.skip_deep,
        run_forest=not args.skip_forest,
        prepro_only=args.prepro_only
    )