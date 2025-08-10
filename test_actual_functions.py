#!/usr/bin/env python3
"""
Test script that uses the actual saving functions from utils/io.py
to verify they work correctly.
"""

import json
import torch
import wandb
from pathlib import Path
from settings import settings
from utils.io import save_metrics_to_wandb, save_activation_data_to_wandb, save_explanations_to_wandb

def test_actual_functions():
    """Test the actual saving functions from our utils.io module."""
    
    print("Testing actual saving functions from utils.io...")
    
    # Login to Wandb
    wandb.login(key=settings.wandb_api_key)
    
    # Initialize a test run
    wandb.init(
        project="tinystories-1m",
        entity="raymondl", 
        name="test_actual_functions",
        tags=["test", "actual_functions"]
    )
    
    current_run = wandb.run
    print(f"Started test run: {current_run.id}")
    print(f"Run URL: {current_run.url}")
    
    # Test 1: Test save_metrics_to_wandb
    print("\n1. Testing save_metrics_to_wandb...")
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
    
    try:
        save_metrics_to_wandb(metrics=test_metrics)
        print("✅ save_metrics_to_wandb completed")
    except Exception as e:
        print(f"❌ save_metrics_to_wandb failed: {e}")
    
    # Test 2: Test save_activation_data_to_wandb  
    print("\n2. Testing save_activation_data_to_wandb...")
    test_activation_data = {
        "layer1.sae": {
            'nonzero_activations': torch.randn(100, 32, dtype=torch.float16),
            'data_indices': torch.randint(0, 1000, (100,), dtype=torch.long),
            'neuron_indices': torch.randint(0, 200, (100,), dtype=torch.long)
        },
        "layer2.sae": {
            'nonzero_activations': torch.randn(150, 32, dtype=torch.float16),
            'data_indices': torch.randint(0, 1000, (150,), dtype=torch.long),
            'neuron_indices': torch.randint(0, 200, (150,), dtype=torch.long)
        }
    }
    
    test_token_ids = [["token1", "token2"], ["token3", "token4"]]
    
    try:
        save_activation_data_to_wandb(
            accumulated_data=test_activation_data,
            all_token_ids=test_token_ids
        )
        print("✅ save_activation_data_to_wandb completed")
    except Exception as e:
        print(f"❌ save_activation_data_to_wandb failed: {e}")
    
    # Test 3: Test save_explanations_to_wandb
    print("\n3. Testing save_explanations_to_wandb...")
    test_explanations = {
        "layer1.sae_neuron_42": {
            "text": "This neuron activates on positive sentiment words",
            "score": 0.85,
            "sae_position": "layer1.sae",
            "neuron_index": 42,
            "num_examples": 10
        },
        "layer2.sae_neuron_123": {
            "text": "This neuron activates on numerical tokens",
            "score": 0.72,
            "sae_position": "layer2.sae",
            "neuron_index": 123,
            "num_examples": 8
        }
    }
    
    try:
        save_explanations_to_wandb(explanations=test_explanations)
        print("✅ save_explanations_to_wandb completed")
    except Exception as e:
        print(f"❌ save_explanations_to_wandb failed: {e}")
    
    print(f"\n🎉 All tests completed! Check files at: {current_run.url}/files")
    
    # Finish the run
    wandb.finish()
    
    return current_run.id

if __name__ == "__main__":
    run_id = test_actual_functions()
    print(f"\n✅ Test complete! Run ID: {run_id}")
    print(f"Check files with: python debug_files.py {run_id}")
    print(f"Or verify with: python quick_verify.py --run_ids {run_id}")
    
    # Also provide the direct URL
    print(f"Direct URL: https://wandb.ai/raymondl/tinystories-1m/runs/{run_id}/files") 