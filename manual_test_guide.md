# Manual Testing Guide for New Saving Functionality

## Quick Testing Steps

### 1. Test Verification Scripts First (No evaluation needed)
```bash
# Test if the verification scripts work with existing runs
python test_saving.py --quick --run_filter "gated"
```

### 2. Test Just Metrics Saving (Fast - 2-3 minutes)
```bash
# Run evaluation with minimal settings to test metrics saving
python evaluation.py \
  --wandb_project "raymondl/tinystories-1m" \
  --filter_runs_by_name "gated" \
  --n_eval_samples 1000 \
  --window_size 32 \
  --num_neurons 50 \
  --save_activation_data
```

### 3. Verify Files Were Saved
```bash
# Quick check
python quick_verify.py --filter_runs_by_name "gated" --max_runs 1

# Detailed check  
python verify_metrics.py --filter_runs_by_name "gated" --max_runs 1
```

### 4. Test With Explanations (Slower - 5-10 minutes)
```bash
# Test full functionality including explanations
python evaluation.py \
  --wandb_project "raymondl/tinystories-1m" \
  --filter_runs_by_name "gated" \
  --n_eval_samples 1000 \
  --window_size 32 \
  --num_neurons 20 \
  --num_features_to_explain 3 \
  --save_activation_data \
  --generate_explanations \
  --explanation_model "gpt-4o-mini"
```

### 5. Full Automated Test
```bash
# Run the comprehensive test script
python test_saving.py --run_filter "gated" --test_explanations
```

## What to Look For

### In Terminal Output:
- ✅ "Staged activation data for [layer] at [path]"
- ✅ "Staged token IDs at [path]"
- ✅ "Staged metrics at [path]"
- ✅ "Successfully uploaded activation data to Wandb run files"
- ✅ "Successfully uploaded metrics to Wandb run files"
- ✅ "Successfully uploaded explanations to Wandb run files"

### In Wandb Web Interface:
Navigate to: `https://wandb.ai/raymondl/tinystories-1m/runs/[RUN_ID]/files`

**Expected files:**
```
📁 activation_data/
  📄 layer1--sae.pt
  📄 layer2--sae.pt  
  📄 all_token_ids.pt
📄 metrics.json
📄 explanations.json (if explanations generated)
📄 explanation_summary.json (if explanations generated)
```

### In Verification Scripts:
- ✅ "Activation data files found: X files"
- ✅ "Metrics file exists"
- ✅ "Explanations file exists" (if generated)
- ✅ "Overall Status: COMPLETE"

## Expected File Sizes

### Activation Data Files:
- Each layer file: ~1-100MB (depends on layer size and samples)
- Token IDs: ~1-10MB
- Total activation_data folder: ~10-500MB

### Other Files:
- metrics.json: ~1-10KB
- explanations.json: ~1-50KB (depends on number of explanations)
- explanation_summary.json: ~1KB

## Troubleshooting

### If files don't appear in Wandb:
1. Check terminal output for error messages
2. Wait 30-60 seconds for sync
3. Refresh the Wandb page
4. Check if the run is still active (files save when run finishes)

### If verification scripts fail:
1. Make sure you're using the correct run filter
2. Check that the run actually exists
3. Verify your Wandb credentials are working

### Common Error Messages:
- "Run not found" → Check project name and run ID
- "No activation data files found" → Evaluation may not have completed
- "Failed to upload" → Check internet connection and Wandb credentials

## Success Criteria

**✅ Test passes if:**
1. Evaluation completes without errors
2. Files appear in Wandb run directory (not artifacts)
3. Verification scripts show "✓" for all file types
4. File sizes are reasonable (not 0 bytes)
5. JSON files contain valid data

**❌ Test fails if:**
1. Files appear in artifacts instead of run files
2. metrics.json is missing even when save_activation_data=False
3. Files are 0 bytes or corrupted
4. Verification scripts show "✗" for expected files 