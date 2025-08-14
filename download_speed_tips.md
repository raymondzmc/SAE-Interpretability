# Speed Up Activation Data Downloads

## Problem
Downloading 42GB of activation data from Wandb can be slow (~10-30 minutes depending on connection).

## Solutions Implemented

### 1. **Local Caching** (Instant after first download)
The data is now cached locally at `{output_path}/cached_{run_id}/`
- First run: Downloads and caches
- Subsequent runs: Loads from cache instantly
- Cache location: `/scratch/raymondl/cached_gcscazk1/`

### 2. **Parallel Downloads** (2-4x faster)
Downloads multiple chunks simultaneously:
```python
# In evaluation.py, data is downloaded with parallel_download=True by default
# Uses 4 parallel workers to download chunks
```

### 3. **Skip Re-downloads**
If you already have the files locally from a previous run:
```bash
# The files are automatically cached and reused
python evaluation.py --generate_explanations --output_path /scratch/raymondl
```

## Manual Options

### Option 1: Keep Files After First Download
After your first successful download, the files are cached at:
```
/scratch/raymondl/cached_gcscazk1/
├── blocks--2--hook_resid_pre.pt  (24.7 GB)
├── blocks--4--hook_resid_pre.pt  (11.8 GB)
├── blocks--6--hook_resid_pre.pt  (5.8 GB)
└── all_token_ids.pt               (0.7 GB)
```

### Option 2: Copy From Previous Download
If you downloaded before but deleted the cache:
```bash
# If you still have the activation_data directory from --skip_upload
cp -r /scratch/raymondl/activation_data_gcscazk1/* /scratch/raymondl/cached_gcscazk1/
```

### Option 3: Download Once, Use Many Times
```bash
# First time - download and cache
python evaluation.py --generate_explanations --output_path /scratch/raymondl

# All subsequent runs use cache (no download needed)
python evaluation.py --generate_explanations --output_path /scratch/raymondl
```

## Performance Comparison

| Method | Time | Notes |
|--------|------|-------|
| Sequential download | ~30 min | Original method |
| Parallel download (4 workers) | ~10-15 min | 2-3x faster |
| Local cache (2nd+ run) | <1 min | Just loads from disk |

## Troubleshooting

### If download is interrupted:
- Delete partial cache: `rm -rf /scratch/raymondl/cached_gcscazk1/`
- Restart the download

### To force re-download:
- Delete cache: `rm -rf /scratch/raymondl/cached_gcscazk1/`
- Or temporarily rename it: `mv /scratch/raymondl/cached_gcscazk1 /scratch/raymondl/cached_gcscazk1_backup`

### To check cache size:
```bash
du -sh /scratch/raymondl/cached_*
```

## Summary
The system now:
1. Downloads chunks in parallel (2-4x faster)
2. Caches locally (instant on subsequent runs)
3. Automatically reuses cached data

You only need to download once per run_id! 