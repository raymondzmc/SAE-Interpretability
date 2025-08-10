#!/usr/bin/env python3
"""
Improved test script to verify file saving paths in Wandb.
This addresses the warnings and ensures proper file upload.
"""

import json
import tempfile
import wandb
import torch
import time
from pathlib import Path
from settings import settings

def test_file_saving_fixed():
    """Test saving files to Wandb with proper paths."""
    
    print("Testing improved Wandb file saving...")
    
    # Login to Wandb
    wandb.login(key=settings.wandb_api_key)
    
    # Initialize a test run
    run = wandb.init(
        project="tinystories-1m",
        entity="raymondl", 
        name="test_file_saving_fixed",
        tags=["test", "file_saving", "fixed"]
    )
    
    print(f"Started test run: {run.id}")
    print(f"Run URL: {run.url}")
    print(f"Files will appear at: {run.url}/files")
    
    # Create test data
    test_metrics = {
        "layer1.sae": {
            "sparsity_l0": 0.12,
            "mse": 0.005,
            "explained_variance": 0.98,
            "alive_dict_components": 150,
            "alive_dict_components_proportion": 0.75
        },
        "layer2.sae": {
            "sparsity_l0": 0.15,
            "mse": 0.007,
            "explained_variance": 0.96,
            "alive_dict_components": 180,
            "alive_dict_components_proportion": 0.80
        }
    }
    
    test_explanations = {
        "layer1.sae_neuron_42": {
            "text": "This neuron activates on positive sentiment words",
            "score": 0.85,
            "sae_position": "layer1.sae",
            "neuron_index": 42,
            "num_examples": 10
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Working in temp directory: {temp_path}")
        
        # Test 1: Save metrics.json (matching our actual function)
        print("\n1. Saving metrics.json...")
        metrics_path = temp_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"Created metrics file: {metrics_path}")
        wandb.save(str(metrics_path), policy="now")
        print("✅ Saved metrics.json")
        
        # Test 2: Save activation data directory (matching our actual function) 
        print("\n2. Saving activation_data directory...")
        activation_data_dir = temp_path / "activation_data"
        activation_data_dir.mkdir()
        
        # Create test files
        layer1_file = activation_data_dir / "layer1--sae.pt"
        layer2_file = activation_data_dir / "layer2--sae.pt"
        token_ids_file = activation_data_dir / "all_token_ids.pt"
        
        # Save actual torch tensors (like real data)
        torch.save({
            'nonzero_activations': torch.randn(100, 32),
            'data_indices': torch.randint(0, 1000, (100,)),
            'neuron_indices': torch.randint(0, 200, (100,))
        }, layer1_file)
        
        torch.save({
            'nonzero_activations': torch.randn(150, 32),
            'data_indices': torch.randint(0, 1000, (150,)),
            'neuron_indices': torch.randint(0, 200, (150,))
        }, layer2_file)
        
        torch.save([["token1", "token2"], ["token3", "token4"]], token_ids_file)
        
        print(f"Created activation files in: {activation_data_dir}")
        
        # Use the exact same pattern as our save function
        wandb.save(str(activation_data_dir / "*"), base_path=str(temp_path), policy="now")
        print("✅ Saved activation_data directory")
        
        # Test 3: Save explanations.json
        print("\n3. Saving explanations.json...")
        explanations_path = temp_path / "explanations.json"
        with open(explanations_path, "w") as f:
            json.dump(test_explanations, f, indent=2)
        
        print(f"Created explanations file: {explanations_path}")
        wandb.save(str(explanations_path), policy="now")
        print("✅ Saved explanations.json")
        
        # Test 4: Save explanation summary
        print("\n4. Saving explanation_summary.json...")
        summary_path = temp_path / "explanation_summary.json"
        summary_data = {
            "num_explanations": len(test_explanations),
            "explained_neurons_per_layer": {"layer1.sae": 1}
        }
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Created summary file: {summary_path}")
        wandb.save(str(summary_path), policy="now")
        print("✅ Saved explanation_summary.json")
    
    print(f"\n🎉 All files saved! Check them at: {run.url}/files")
    
    # Wait for sync
    print("Waiting 10 seconds for files to sync...")
    time.sleep(10)
    
    # Try to verify immediately
    print("\nTesting immediate verification...")
    run_files = run.files()
    file_names = [f.name for f in run_files]
    print(f"Files visible via run.files(): {file_names}")
    
    wandb.finish()
    
    return run.id

if __name__ == "__main__":
    run_id = test_file_saving_fixed()
    print(f"\n✅ Test complete! Run ID: {run_id}")
    print(f"Now test verification with: python debug_files.py {run_id}")
    print(f"Or: python quick_verify.py --run_ids {run_id}") 