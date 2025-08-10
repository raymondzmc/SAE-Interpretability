#!/usr/bin/env python3
"""
Test script to verify the new saving functionality is working correctly.
This script will run evaluation on a single run and verify all files are saved properly.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def test_evaluation_saving(
    run_filter: str = "gated",
    wandb_project: str = "raymondl/tinystories-1m",
    test_explanations: bool = False
):
    """Test the evaluation saving functionality."""
    
    print("=" * 60)
    print("TESTING EVALUATION SAVING FUNCTIONALITY")
    print("=" * 60)
    
    # Step 1: Run evaluation with minimal settings for faster testing
    print("\n1. Running evaluation with new saving logic...")
    
    cmd = [
        "python", "evaluation.py",
        "--wandb_project", wandb_project,
        "--filter_runs_by_name", run_filter,
        "--window_size", "32",  # Smaller window for faster testing
        "--num_neurons", "50",  # Fewer neurons for faster testing
        "--n_eval_samples", "1000",  # Much smaller sample size
        "--save_activation_data",  # Test activation data saving
    ]
    
    if test_explanations:
        cmd.extend([
            "--generate_explanations",
            "--num_features_to_explain", "3",  # Minimal explanations for testing
            "--explanation_model", "gpt-4o-mini"  # Faster model
        ])
    
    print(f"Command: {' '.join(cmd)}")
    print("This may take a few minutes...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode != 0:
            print(f"❌ Evaluation failed!")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
        else:
            print("✅ Evaluation completed successfully!")
            print("Key output lines:")
            for line in result.stdout.split('\n'):
                if any(keyword in line.lower() for keyword in [
                    'successfully uploaded', 'saving', 'staged', 'found', 'runs matching'
                ]):
                    print(f"  {line}")
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False
    
    # Step 2: Verify files were saved correctly
    print("\n2. Verifying files were saved correctly...")
    
    # Wait a moment for files to sync
    print("Waiting 5 seconds for files to sync...")
    time.sleep(5)
    
    # Run verification
    verify_cmd = [
        "python", "quick_verify.py",
        "--wandb_project", wandb_project,
        "--filter_runs_by_name", run_filter,
        "--max_runs", "1"  # Just check the first run
    ]
    
    try:
        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        print("Verification results:")
        print(verify_result.stdout)
        
        # Check for success indicators
        if "✓" in verify_result.stdout:
            print("✅ Verification shows files were saved!")
        else:
            print("❌ Verification shows missing files")
            print(f"STDERR: {verify_result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running verification: {e}")
        return False
    
    # Step 3: Test comprehensive verification
    print("\n3. Running comprehensive verification...")
    
    comprehensive_cmd = [
        "python", "verify_metrics.py",
        "--wandb_project", wandb_project,
        "--filter_runs_by_name", run_filter,
        "--max_runs", "1"
    ]
    
    try:
        comp_result = subprocess.run(comprehensive_cmd, capture_output=True, text=True)
        print("Comprehensive verification results:")
        print(comp_result.stdout)
        
        # Check overall status
        if "COMPLETE" in comp_result.stdout:
            print("✅ Comprehensive verification passed!")
            return True
        else:
            print("❌ Comprehensive verification found issues")
            if comp_result.stderr:
                print(f"STDERR: {comp_result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running comprehensive verification: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the evaluation saving functionality")
    parser.add_argument("--run_filter", type=str, default="gated",
                       help="Filter string for run names to test")
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project to test")
    parser.add_argument("--test_explanations", action="store_true",
                       help="Also test explanation generation (slower)")
    parser.add_argument("--quick", action="store_true",
                       help="Skip evaluation, just test verification scripts")
    
    args = parser.parse_args()
    
    if args.quick:
        print("Quick mode: Testing verification scripts only...")
        
        # Just run the verification scripts
        print("\n1. Testing quick verification...")
        quick_cmd = [
            "python", "quick_verify.py",
            "--wandb_project", args.wandb_project,
            "--filter_runs_by_name", args.run_filter,
            "--max_runs", "2"
        ]
        
        result = subprocess.run(quick_cmd)
        
        print("\n2. Testing comprehensive verification...")
        comp_cmd = [
            "python", "verify_metrics.py",
            "--wandb_project", args.wandb_project,
            "--filter_runs_by_name", args.run_filter,
            "--max_runs", "1"
        ]
        
        result = subprocess.run(comp_cmd)
        
    else:
        success = test_evaluation_saving(
            run_filter=args.run_filter,
            wandb_project=args.wandb_project,
            test_explanations=args.test_explanations
        )
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 ALL TESTS PASSED! Saving functionality is working correctly.")
            print("\nNext steps:")
            print("1. You can now run full evaluations with confidence")
            print("2. Files will appear directly in your Wandb run directories")
            print("3. Use the verification scripts to check any run")
        else:
            print("❌ TESTS FAILED! Please check the output above for issues.")
            sys.exit(1)
        print("=" * 60)


if __name__ == "__main__":
    main() 