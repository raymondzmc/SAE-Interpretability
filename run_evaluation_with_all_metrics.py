#!/usr/bin/env python3
"""
Wrapper script to run evaluation with all metrics saving enabled.
This ensures complete data collection including:
- Activation data saving
- Explanation generation  
- Evaluation results upload
"""

import subprocess
import argparse
import sys

def run_evaluation_with_metrics(
    wandb_project="raymondl/tinystories-1m",
    filter_runs_by_name=None,
    window_size=64,
    num_neurons=300,
    num_features_to_explain=10,
    n_eval_samples=50000,
    explanation_model="gpt-4o-mini",
    simulator_model="gpt-4o-mini",
    generate_explanations=True,
    evaluate_explanations=False
):
    """Run evaluation with all metrics saving enabled."""
    
    cmd = [
        "python", "evaluation.py",
        "--window_size", str(window_size),
        "--num_neurons", str(num_neurons), 
        "--num_features_to_explain", str(num_features_to_explain),
        "--n_eval_samples", str(n_eval_samples),
        "--explanation_model", explanation_model,
        "--simulator_model", simulator_model,
        "--wandb_project", wandb_project,
        "--save_activation_data",  # Enable activation data saving
    ]
    
    if filter_runs_by_name:
        cmd.extend(["--filter_runs_by_name", filter_runs_by_name])
    
    if generate_explanations:
        cmd.append("--generate_explanations")
    
    if evaluate_explanations:
        cmd.append("--evaluate_explanations")
    
    print("Running evaluation with command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run evaluation with comprehensive metrics saving")
    
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project")
    parser.add_argument("--filter_runs_by_name", type=str, default=None,
                       help="Filter runs by name")
    parser.add_argument("--window_size", type=int, default=64,
                       help="Window size")
    parser.add_argument("--num_neurons", type=int, default=300,
                       help="Number of neurons to process")
    parser.add_argument("--num_features_to_explain", type=int, default=10,
                       help="Number of features to explain per neuron")
    parser.add_argument("--n_eval_samples", type=int, default=50000,
                       help="Number of evaluation samples")
    parser.add_argument("--explanation_model", type=str, default="gpt-4o-mini",
                       choices=["gpt-4o", "gpt-4o-mini"],
                       help="Model for generating explanations")
    parser.add_argument("--simulator_model", type=str, default="gpt-4o-mini", 
                       choices=["gpt-4o", "gpt-4o-mini"],
                       help="Model for simulation scoring")
    parser.add_argument("--no_explanations", action="store_true",
                       help="Skip explanation generation (faster)")
    parser.add_argument("--evaluate_explanations", action="store_true",
                       help="Also evaluate explanation quality (slower)")
    
    args = parser.parse_args()
    
    # Run the evaluation
    success = run_evaluation_with_metrics(
        wandb_project=args.wandb_project,
        filter_runs_by_name=args.filter_runs_by_name,
        window_size=args.window_size,
        num_neurons=args.num_neurons,
        num_features_to_explain=args.num_features_to_explain,
        n_eval_samples=args.n_eval_samples,
        explanation_model=args.explanation_model,
        simulator_model=args.simulator_model,
        generate_explanations=not args.no_explanations,
        evaluate_explanations=args.evaluate_explanations
    )
    
    if success:
        print("\nNext steps:")
        print("1. Run: python quick_verify.py --filter_runs_by_name YOUR_FILTER")
        print("2. Or run: python verify_metrics.py for full validation")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 