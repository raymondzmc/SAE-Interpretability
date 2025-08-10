#!/usr/bin/env python3
"""
Simple test script to verify file saving paths in Wandb.
This will save a test file and show you exactly where it appears.
"""

import json
import tempfile
import wandb
from pathlib import Path
from settings import settings

def test_file_saving():
    """Test saving files to Wandb and show the paths."""
    
    print("Testing Wandb file saving paths...")
    
    # Login to Wandb
    wandb.login(key=settings.wandb_api_key)
    
    # Initialize a test run
    run = wandb.init(
        project="tinystories-1m",
        entity="raymondl", 
        name="test_file_saving",
        tags=["test", "file_saving"]
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
    
    # Test 1: Save a simple JSON file to root
    print("\n1. Testing root-level file saving...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save metrics file
        metrics_path = temp_path / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"Staged test_metrics.json at {metrics_path}")
        wandb.save(str(metrics_path), policy="now")
        print("✅ Uploaded test_metrics.json to run root")
    
    # Test 2: Save files in a subdirectory
    print("\n2. Testing subdirectory file saving...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create activation_data subdirectory
        activation_data_dir = temp_path / "activation_data"
        activation_data_dir.mkdir()
        
        # Save some test activation files
        test_file1 = activation_data_dir / "layer1--sae.pt"
        test_file2 = activation_data_dir / "layer2--sae.pt"
        
        # Create dummy data (just save some tensors)
        import torch
        torch.save({"test": "data1"}, test_file1)
        torch.save({"test": "data2"}, test_file2)
        
        print(f"Staged activation files in {activation_data_dir}")
        
        # Upload the whole directory
        wandb.save(str(activation_data_dir / "*"), base_path=str(temp_path), policy="now")
        print("✅ Uploaded activation_data directory")
    
    # Test 3: Save another root file
    print("\n3. Testing another root-level file...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save explanations file
        explanations_path = temp_path / "test_explanations.json"
        with open(explanations_path, "w") as f:
            json.dump(test_explanations, f, indent=2)
        
        print(f"Staged test_explanations.json at {explanations_path}")
        wandb.save(str(explanations_path), policy="now")
        print("✅ Uploaded test_explanations.json to run root")
    
    print(f"\n🎉 Test completed! Check the files at:")
    print(f"   {run.url}/files")
    print(f"\nExpected structure:")
    print(f"   📄 test_metrics.json")
    print(f"   📄 test_explanations.json") 
    print(f"   📁 activation_data/")
    print(f"     📄 layer1--sae.pt")
    print(f"     📄 layer2--sae.pt")
    
    # Keep the run active for a moment to ensure files sync
    print(f"\nWaiting a few seconds for files to sync...")
    import time
    time.sleep(3)
    
    # Finish the run
    wandb.finish()
    
    return run.url

if __name__ == "__main__":
    run_url = test_file_saving()
    print(f"\n✅ Test complete! View files at: {run_url}/files") 