#!/usr/bin/env python3
"""
Demo: When does explained variance become negative?

This shows that negative explained variance occurs when your model performs worse 
than a naive baseline that just predicts the mean of the targets.
"""

import torch
import numpy as np
from utils.metrics import explained_variance

def demo_negative_explained_variance():
    """Demonstrate cases where explained variance becomes negative."""
    
    print("🧪 EXPLAINED VARIANCE: When does it go negative?")
    print("=" * 60)
    
    # Create some target data
    torch.manual_seed(42)
    batch_size, seq_len, dim = 4, 32, 128
    
    # Target has some structure (not just noise)
    target = torch.randn(batch_size, seq_len, dim) * 0.5
    target_mean = target.mean(dim=-1, keepdim=True)
    
    print("Test Cases:")
    print("-" * 30)
    
    # Case 1: Perfect reconstruction
    print("1️⃣ Perfect Reconstruction:")
    perfect_pred = target.clone()
    ev_perfect = explained_variance(perfect_pred, target).mean().item()
    print(f"   Explained Variance: {ev_perfect:.6f}")
    print("   → Should be ~1.0 (perfect)")
    
    # Case 2: Good reconstruction
    print("\n2️⃣ Good Reconstruction (small error):")
    good_pred = target + 0.1 * torch.randn_like(target)
    ev_good = explained_variance(good_pred, target).mean().item() 
    print(f"   Explained Variance: {ev_good:.6f}")
    print("   → Should be positive and high")
    
    # Case 3: Mean baseline (should be ~0)
    print("\n3️⃣ Naive Baseline (predict mean):")
    mean_pred = target_mean.expand_as(target)
    ev_mean = explained_variance(mean_pred, target).mean().item()
    print(f"   Explained Variance: {ev_mean:.6f}")
    print("   → Should be ~0.0 (as good as predicting mean)")
    
    # Case 4: Bad reconstruction (NEGATIVE!)
    print("\n4️⃣ Bad Reconstruction (worse than mean):")
    bad_pred = torch.randn_like(target) * 2.0  # Random predictions with high variance
    ev_bad = explained_variance(bad_pred, target).mean().item()
    print(f"   Explained Variance: {ev_bad:.6f}")
    print("   → NEGATIVE! Worse than predicting the mean")
    
    # Case 5: Very bad reconstruction (MORE NEGATIVE!)
    print("\n5️⃣ Very Bad Reconstruction:")
    very_bad_pred = torch.randn_like(target) * 5.0  # Even more random
    ev_very_bad = explained_variance(very_bad_pred, target).mean().item()
    print(f"   Explained Variance: {ev_very_bad:.6f}")
    print("   → VERY NEGATIVE! Much worse than mean baseline")
    
    print(f"\n{'='*60}")
    print("🔍 INTERPRETATION:")
    print("  • EV ≈ 1.0   → Perfect reconstruction") 
    print("  • EV > 0.0   → Better than predicting mean (good!)")
    print("  • EV ≈ 0.0   → As good as predicting mean (baseline)")
    print("  • EV < 0.0   → Worse than predicting mean (bad!)")
    print("  • EV << 0.0  → Much worse than mean (very bad!)")
    
    print(f"\n🚨 NEGATIVE EXPLAINED VARIANCE IN SAE TRAINING:")
    print("  This typically happens when:")
    print("  • SAE is undertrained/poorly initialized") 
    print("  • Learning rate is too high (unstable training)")
    print("  • SAE architecture doesn't fit the data well")
    print("  • Early in training before the SAE has learned anything useful")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo_negative_explained_variance() 