#!/usr/bin/env python3
"""
Script to verify that all evaluation metrics are properly saved to Wandb.

This script:
1. Connects to Wandb and fetches runs from a project
2. Checks for the presence of activation_data and evaluation_results artifacts
3. Validates that all expected metrics are present
4. Reports any missing or incomplete data
"""

import argparse
import json
import tempfile
import wandb
from pathlib import Path
from typing import Dict, Any, Set
from settings import settings

# Expected metrics for each SAE position
EXPECTED_METRICS = {
    'sparsity_l0',
    'mse', 
    'explained_variance',
    'alive_dict_components',
    'alive_dict_components_proportion'
}

# Expected explanation fields
EXPECTED_EXPLANATION_FIELDS = {
    'text',
    'score',
    'sae_position', 
    'neuron_index',
    'num_examples'
}


def check_activation_data_files(run_id: str, project: str) -> Dict[str, Any]:
    """Check if activation data files exist in the run."""
    api = wandb.Api()
    
    result = {
        'exists': False,
        'sae_positions': [],
        'num_files': 0,
        'has_token_ids': False,
        'errors': []
    }
    
    try:
        run = api.run(f"{project}/{run_id}")
        files = list(run.files())
        file_names = [f.name for f in files]
        
        # Check activation data files
        activation_files = [f for f in file_names if f.startswith("activation_data/") and f.endswith(".pt")]
        if activation_files:
            result['exists'] = True
            result['num_files'] = len(activation_files)
            
            # Extract SAE positions from filenames
            for filename in activation_files:
                if filename == "activation_data/all_token_ids.pt":
                    result['has_token_ids'] = True
                else:
                    # Convert filename back to SAE position
                    basename = Path(filename).name[:-3]  # Remove .pt
                    sae_pos = basename.replace("--", ".")
                    result['sae_positions'].append(sae_pos)
        else:
            result['errors'].append("No activation data files found")
                
    except wandb.errors.CommError as e:
        result['errors'].append(f"Run not found: {e}")
    except Exception as e:
        result['errors'].append(f"Error accessing run: {e}")
    
    return result


def check_metrics_and_explanations(run_id: str, project: str) -> Dict[str, Any]:
    """Check if metrics and explanations files exist in the run and validate their contents."""
    api = wandb.Api()
    
    result = {
        'has_metrics': False,
        'metrics_complete': False,
        'sae_positions': [],
        'missing_metrics': {},
        'has_explanations': False,
        'num_explanations': 0,
        'explanations_by_layer': {},
        'incomplete_explanations': [],
        'errors': []
    }
    
    try:
        run = api.run(f"{project}/{run_id}")
        files = list(run.files())
        file_names = [f.name for f in files]
        
        # Download files to temporary directory for validation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Check metrics
            if "metrics.json" in file_names:
                result['has_metrics'] = True
                
                # Download and validate metrics
                try:
                    for file in files:
                        if file.name == "metrics.json":
                            local_path = temp_path / "metrics.json"
                            file.download(root=temp_dir, replace=True)
                            
                            with open(local_path, 'r') as f:
                                metrics = json.load(f)
                            
                            result['sae_positions'] = list(metrics.keys())
                            result['missing_metrics'] = {}
                            
                            # Validate each SAE position has all expected metrics
                            all_complete = True
                            for sae_pos, sae_metrics in metrics.items():
                                missing = EXPECTED_METRICS - set(sae_metrics.keys())
                                if missing:
                                    result['missing_metrics'][sae_pos] = list(missing)
                                    all_complete = False
                            
                            result['metrics_complete'] = all_complete
                            break
                except Exception as e:
                    result['errors'].append(f"Error validating metrics: {e}")
            
            # Check explanations
            if "explanations.json" in file_names:
                result['has_explanations'] = True
                
                # Download and validate explanations
                try:
                    for file in files:
                        if file.name == "explanations.json":
                            local_path = temp_path / "explanations.json"
                            file.download(root=temp_dir, replace=True)
                            
                            with open(local_path, 'r') as f:
                                explanations = json.load(f)
                            
                            result['num_explanations'] = len(explanations)
                            
                            # Group explanations by layer
                            for key, explanation in explanations.items():
                                if "_neuron_" in key:
                                    layer_name = key.split("_neuron_")[0]
                                    if layer_name not in result['explanations_by_layer']:
                                        result['explanations_by_layer'][layer_name] = 0
                                    result['explanations_by_layer'][layer_name] += 1
                                    
                                    # Check if explanation has all expected fields
                                    missing_fields = EXPECTED_EXPLANATION_FIELDS - set(explanation.keys())
                                    if missing_fields:
                                        result['incomplete_explanations'].append({
                                            'key': key,
                                            'missing_fields': list(missing_fields)
                                        })
                            break
                except Exception as e:
                    result['errors'].append(f"Error validating explanations: {e}")
                
    except wandb.errors.CommError as e:
        result['errors'].append(f"Run not found: {e}")
    except Exception as e:
        result['errors'].append(f"Error accessing run: {e}")
    
    return result


def verify_run_metrics(run_id: str, run_name: str, project: str) -> Dict[str, Any]:
    """Verify all metrics for a single run."""
    print(f"\n=== Verifying Run: {run_name} ({run_id}) ===")
    
    # Check activation data files
    print("Checking activation data files...")
    activation_check = check_activation_data_files(run_id, project)
    
    # Check metrics and explanations files  
    print("Checking metrics and explanations files...")
    metrics_explanations_check = check_metrics_and_explanations(run_id, project)
    
    # Summary
    summary = {
        'run_id': run_id,
        'run_name': run_name,
        'activation_data': activation_check,
        'metrics_explanations': metrics_explanations_check,
        'overall_status': 'complete' if (
            activation_check['exists'] and 
            metrics_explanations_check['has_metrics'] and
            metrics_explanations_check['metrics_complete']
        ) else 'incomplete'
    }
    
    # Print summary
    print(f"  Activation Data: {'✓' if activation_check['exists'] else '✗'}")
    if activation_check['exists']:
        print(f"    SAE Positions: {len(activation_check['sae_positions'])}")
        print(f"    Number of Files: {activation_check['num_files']}")
        print(f"    Has Token IDs: {'✓' if activation_check['has_token_ids'] else '✗'}")
    
    print(f"  Metrics: {'✓' if metrics_explanations_check['has_metrics'] else '✗'}")
    if metrics_explanations_check['has_metrics']:
        print(f"    SAE Positions: {len(metrics_explanations_check['sae_positions'])}")
        print(f"    Metrics Complete: {'✓' if metrics_explanations_check['metrics_complete'] else '✗'}")
        if metrics_explanations_check['missing_metrics']:
            print(f"    Missing Metrics: {metrics_explanations_check['missing_metrics']}")
    
    print(f"  Explanations: {'✓' if metrics_explanations_check['has_explanations'] else '✗'}")
    if metrics_explanations_check['has_explanations']:
        print(f"    Total Explanations: {metrics_explanations_check['num_explanations']}")
        print(f"    Explanations by Layer: {metrics_explanations_check['explanations_by_layer']}")
        if metrics_explanations_check['incomplete_explanations']:
            print(f"    Incomplete Explanations: {len(metrics_explanations_check['incomplete_explanations'])}")
    
    # Print errors
    all_errors = activation_check['errors'] + metrics_explanations_check['errors']
    if all_errors:
        print(f"  Errors: {all_errors}")
    
    print(f"  Overall Status: {summary['overall_status'].upper()}")
    
    return summary


def main():
    """Main function to verify metrics for all runs in a project."""
    parser = argparse.ArgumentParser(description="Verify that all evaluation metrics are saved properly")
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project in format 'entity/project'")
    parser.add_argument("--filter_runs_by_name", type=str, default=None,
                       help="Filter runs by a specific string in their name")
    parser.add_argument("--max_runs", type=int, default=None,
                       help="Maximum number of runs to check (for testing)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save verification results to JSON file")
    
    args = parser.parse_args()
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get runs
    runs = api.runs(args.wandb_project)
    
    if args.filter_runs_by_name:
        old_len = len(runs)
        runs = [run for run in runs if args.filter_runs_by_name in run.name]
        print(f"Found {len(runs)}/{old_len} runs matching filter: {args.filter_runs_by_name}")
    
    if args.max_runs:
        runs = runs[:args.max_runs]
        print(f"Limiting verification to first {args.max_runs} runs")
    
    print(f"Verifying metrics for {len(runs)} runs...")
    
    # Verify each run
    all_results = []
    complete_runs = 0
    incomplete_runs = 0
    
    for run in runs:
        try:
            result = verify_run_metrics(run.id, run.name, args.wandb_project)
            all_results.append(result)
            
            if result['overall_status'] == 'complete':
                complete_runs += 1
            else:
                incomplete_runs += 1
                
        except Exception as e:
            print(f"Error verifying run {run.name}: {e}")
            incomplete_runs += 1
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total Runs Checked: {len(runs)}")
    print(f"Complete Runs: {complete_runs}")
    print(f"Incomplete Runs: {incomplete_runs}")
    print(f"Success Rate: {complete_runs/len(runs)*100:.1f}%" if runs else "N/A")
    
    # Save results to file if requested
    if args.output_file:
        output_data = {
            'summary': {
                'total_runs': len(runs),
                'complete_runs': complete_runs,
                'incomplete_runs': incomplete_runs,
                'success_rate': complete_runs/len(runs) if runs else 0
            },
            'detailed_results': all_results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Detailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main() 