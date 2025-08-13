"""
Analyze HardConcrete SAE evaluation results from wandb.
Separates runs by input_dependent_gates setting and extracts metrics.
"""

import json
import torch
import wandb
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm

from settings import settings
from models import SAETransformer
from utils.io import load_metrics_from_wandb


def extract_run_metrics(run_id: str, run_name: str, project: str, device: torch.device) -> Dict[str, Any]:
    """Extract metrics and configuration for a single run."""
    try:
        # Load the model to get the configuration
        print(f"  Loading model for run {run_id} ({run_name})...")
        model = SAETransformer.from_wandb(f"{project}/{run_id}")
        
        # Check if this is actually a HardConcrete SAE
        sae_type = model.sae_config.sae_type
        if str(sae_type) != "SAEType.HARD_CONCRETE":
            print(f"  Skipping run {run_id} - not a HardConcrete SAE (type: {sae_type})")
            return None
            
        # Get the input_dependent_gates setting
        input_dependent_gates = model.sae_config.input_dependent_gates
        
        # Get other relevant config settings
        sparsity_coeff = model.sae_config.sparsity_coeff
        dict_size_ratio = model.sae_config.dict_size_to_input_ratio
        
        # Get actual dictionary size from one of the SAE modules
        dict_size = None
        if len(model.saes) > 0:
            first_sae = next(iter(model.saes.values()))
            if hasattr(first_sae, 'n_dict_components'):
                dict_size = first_sae.n_dict_components
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()
        
        # Load metrics from wandb
        print(f"  Loading metrics for run {run_id}...")
        metrics = load_metrics_from_wandb(run_id, project=project)
        
        if metrics is None:
            print(f"  Warning: No metrics found for run {run_id}")
            return None
        
        return {
            'run_id': run_id,
            'run_name': run_name,
            'input_dependent_gates': input_dependent_gates,
            'sparsity_coeff': sparsity_coeff,
            'dict_size_ratio': dict_size_ratio,
            'dict_size': dict_size,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"  Error processing run {run_id}: {e}")
        return None


def main():
    """Main function to analyze HardConcrete runs."""
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get the project - adjust this if needed
    project = "raymondl/tinystories-1m"
    
    print(f"Fetching runs from project: {project}")
    runs = api.runs(project)
    
    # Filter for HardConcrete runs
    hc_runs = []
    for run in runs:
        name_lower = run.name.lower()
        if 'hard_concrete' in name_lower or 'hardconcrete' in name_lower:
            hc_runs.append(run)
    
    print(f"Found {len(hc_runs)} potential HardConcrete runs")
    
    # Process each run and separate by gate type
    input_dependent_runs = []
    input_independent_runs = []
    
    for run in tqdm(hc_runs, desc="Processing runs"):
        print(f"\nProcessing run: {run.name} (ID: {run.id})")
        
        run_data = extract_run_metrics(
            run_id=run.id,
            run_name=run.name,
            project=project,
            device=device
        )
        
        if run_data is not None:
            if run_data['input_dependent_gates']:
                input_dependent_runs.append(run_data)
            else:
                input_independent_runs.append(run_data)
    
    print(f"\n" + "="*60)
    print(f"Summary:")
    print(f"  Input-dependent gates: {len(input_dependent_runs)} runs")
    print(f"  Input-independent gates: {len(input_independent_runs)} runs")
    print("="*60)
    
    # Organize metrics by layer for each group
    def organize_by_layer(runs: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize run metrics by layer."""
        by_layer = defaultdict(list)
        
        for run_data in runs:
            metrics = run_data['metrics']
            for layer_name, layer_metrics in metrics.items():
                by_layer[layer_name].append({
                    'run_id': run_data['run_id'],
                    'run_name': run_data['run_name'],
                    'sparsity_coeff': run_data['sparsity_coeff'],
                    'dict_size_ratio': run_data['dict_size_ratio'],
                    'dict_size': run_data['dict_size'],
                    'metrics': layer_metrics
                })
        
        return dict(by_layer)
    
    # Organize results
    results = {
        'input_dependent_gates': {
            'num_runs': len(input_dependent_runs),
            'runs': input_dependent_runs,
            'by_layer': organize_by_layer(input_dependent_runs)
        },
        'input_independent_gates': {
            'num_runs': len(input_independent_runs),
            'runs': input_independent_runs,
            'by_layer': organize_by_layer(input_independent_runs)
        }
    }
    
    # Save results to JSON
    output_file = 'hc_metrics_by_gate_type.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Layer-wise Summary:")
    print("="*60)
    
    # Get all unique layers
    all_layers = set()
    all_layers.update(results['input_dependent_gates']['by_layer'].keys())
    all_layers.update(results['input_independent_gates']['by_layer'].keys())
    
    for layer in sorted(all_layers):
        print(f"\n{layer}:")
        
        # Input-dependent stats
        if layer in results['input_dependent_gates']['by_layer']:
            dep_runs = results['input_dependent_gates']['by_layer'][layer]
            avg_sparsity = sum(r['metrics']['sparsity_l0'] for r in dep_runs) / len(dep_runs)
            avg_mse = sum(r['metrics']['mse'] for r in dep_runs) / len(dep_runs)
            avg_exp_var = sum(r['metrics'].get('explained_variance', 0) for r in dep_runs) / len(dep_runs)
            print(f"  Input-dependent ({len(dep_runs)} runs):")
            print(f"    Avg L0: {avg_sparsity:.2f}")
            print(f"    Avg MSE: {avg_mse:.6f}")
            print(f"    Avg Explained Var: {avg_exp_var:.4f}")
        
        # Input-independent stats
        if layer in results['input_independent_gates']['by_layer']:
            indep_runs = results['input_independent_gates']['by_layer'][layer]
            avg_sparsity = sum(r['metrics']['sparsity_l0'] for r in indep_runs) / len(indep_runs)
            avg_mse = sum(r['metrics']['mse'] for r in indep_runs) / len(indep_runs)
            avg_exp_var = sum(r['metrics'].get('explained_variance', 0) for r in indep_runs) / len(indep_runs)
            print(f"  Input-independent ({len(indep_runs)} runs):")
            print(f"    Avg L0: {avg_sparsity:.2f}")
            print(f"    Avg MSE: {avg_mse:.6f}")
            print(f"    Avg Explained Var: {avg_exp_var:.4f}")
    
    return results


if __name__ == "__main__":
    results = main() 