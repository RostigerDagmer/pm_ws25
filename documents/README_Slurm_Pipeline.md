**Branch:** `train_evaluate_ML`

# ML Classifier Training Pipeline

## Overview

```
Step 0: Profile (one-time)  →  Step 1: Generate Labels  →  Step 2: Train Classifier
   ~5-10 minutes                   ~2-4 hours (parallel)        ~1 hour
   Optimize resources               16 datasets → CSV files      CSV → trained model
```

## Optimized Workflow: Profile → Adjust → Run

### Step 0: Profile Resources (First Time / New Data)

**Automatically determine optimal Slurm resources:**

```bash
cd ~/pm_ws25
bash scripts/profile_job.sh
```

**What it does:**
- Runs label generation on 1 dataset
- Measures CPU, memory, and time usage
- Outputs recommendations for Slurm config

**Example output:**
```
RECOMMENDATIONS FOR SLURM CONFIG
============================================
Edit lrz-cluster/run_create_labels.slurm:
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=24G
  #SBATCH --time=03:00:00

Check cluster: sinfo -o "%P %a %T %c"
Then adjust throttle:
  If idle CPUs > 128: --array=0-15%16 (FAST)
  If idle CPUs > 64:  --array=0-15%8  (BALANCED)
```

**How to apply recommendations:**

Profile measures **what your task needs**:
- **Memory peak**: How much RAM your dataset requires
- **CPU usage**: How efficiently it uses parallel workers
- **Runtime**: How long it takes to process one dataset

**Edit [lrz-cluster/run_create_labels.slurm](../lrz-cluster/run_create_labels.slurm) lines 22-26:**

```bash
#SBATCH --array=0-15%8          # Line 22: Manually set (not from profile)
#SBATCH --time=04:00:00         # Line 23: From profile
#SBATCH --cpus-per-task=8       # Line 25: From profile
#SBATCH --mem=32G               # Line 26: From profile
```

**How to apply:**
1. **Line 23** (`--time`): Copy from profile output
2. **Line 25** (`--cpus-per-task`): Copy from profile output
3. **Line 26** (`--mem`): Copy from profile output
4. **Line 22** (`--array` throttle): Manually adjust based on cluster availability

**Example:**
- Profile outputs: `--cpus-per-task=12, --mem=40G, --time=03:30:00`
- Apply: Line 25 = 12, Line 26 = 40G, Line 23 = 03:30:00
- Check cluster: `sinfo` → adjust Line 22 throttle (e.g., %8 or %16)
- Save and submit: `sbatch lrz-cluster/run_create_labels.slurm`

### Step 1: Generate Training Labels

**Cache mechanism:** Script automatically skips datasets with existing CSV files.

```bash
cd ~/pm_ws25

# Optional: Check cluster availability
sinfo -o "%P %a %T %c"

# Submit job array
sbatch lrz-cluster/run_create_labels.slurm

# Monitor progress
squeue -u $USER
tail -f logs/create_labels_*.out

# Check completion (should reach 16)
find data/runs -name "*.train.csv" | wc -l
```

**Cache behavior:**
- ✓ **Exists**: Skips processing, saves time
- ✗ **Missing**: Generates CSV files
- 🔄 **Force regenerate**: Add `--force` flag to `create_labels.py`

### Step 2: Train & Evaluate Classifier

```bash
# After Step 1 completes (all CSV files generated)
sbatch lrz-cluster/run_evaluate_classifier.slurm

# View results
cat outputs/evaluate_classifier/summary.txt
```

## Slurm Configuration Guide

### Job Array Syntax

```bash
#SBATCH --array=0-15%8
```
- `0-15`: 16 datasets (tasks 0 through 15)
- `%8`: Max 8 concurrent jobs (throttle)

**Throttle determines speed:**
- `%16` = All 16 run at once = ~2 hours (FAST)
- `%8` = 8 at a time = ~4 hours (BALANCED)
- `%4` = 4 at a time = ~8 hours (CONSERVATIVE)

### Resource Settings (Per Job)

```bash
#SBATCH --cpus-per-task=8    # CPUs for this job
#SBATCH --mem=32G            # RAM for this job
#SBATCH --time=04:00:00      # Time limit per job
```

**Adjust based on profiling results.**

## Scaling for Larger/Synthetic Datasets

### When to Re-Profile

**Re-run profiling if:**
- Dataset size changes significantly (e.g., 2x larger)
- Switching to synthetic data
- Adding complex features

```bash
# Profile your new dataset type
bash scripts/profile_job.sh data/path/to/synthetic_large.xes
```

### Scaling Examples

**100 datasets (2x larger data):**
```bash
# Step 1: Profile new dataset type
bash scripts/profile_job.sh data/synthetic/large_dataset.xes

# Step 2: Apply recommendations to run_create_labels.slurm
# Example if profiling shows: 45GB memory, 3 hours runtime

#SBATCH --array=0-99%20        # 100 datasets, 20 concurrent
#SBATCH --cpus-per-task=8      # From profiling
#SBATCH --mem=48G              # From profiling (45GB + buffer)
#SBATCH --time=04:00:00        # From profiling (3hrs + buffer)

# Step 3: Add datasets to DATASETS array (lines 63-80)
DATASETS+=(
    "data/synthetic/dataset_01.xes"
    "data/synthetic/dataset_02.xes"
    # ... add all 84 new datasets ...
)
```

**Key principle:** Profile → Adjust → Run. Simple and effective.

## Quick Reference

### Add New Datasets

Edit [lrz-cluster/run_create_labels.slurm](../lrz-cluster/run_create_labels.slurm):

```bash
# Line 63-80: Add to DATASETS array
DATASETS+=(
    "data/your-uuid/your-dataset.xes"
)

# Line 22: Update array size (16 → 18 for 2 new datasets)
#SBATCH --array=0-17%8
```

### Check Cluster Status

```bash
sinfo -o "%P %a %T %c"              # Partition availability
squeue -u $USER                     # Your running jobs
```

### Common Commands

```bash
# Monitor job progress
tail -f logs/create_labels_*.out

# Count completed datasets
find data/runs -name "*.train.csv" | wc -l

# Cancel all your jobs
scancel -u $USER

# View job details
scontrol show job <jobid>
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "No module named 'util'" | PYTHONPATH not set (auto-configured in Slurm script) |
| "No valid training samples found" | CSV files missing - run Step 1 first |
| Job timeout | Increase `--time` based on profiling |
| Out of memory | Increase `--mem` based on profiling |

**Check failed jobs:**
```bash
grep "Exit Code: 1" logs/create_labels_*.out
cat logs/create_labels_<jobid>_<taskid>.err
```

**Re-run single dataset:**
```bash
# Normal (uses cache if CSV exists)
python scripts/create_labels.py \
    --config configs/default.yaml \
    --path data/<uuid>/<file>.xes \
    --workers 8 --seed 1

# Force regenerate (ignores cache)
python scripts/create_labels.py \
    --config configs/default.yaml \
    --path data/<uuid>/<file>.xes \
    --workers 8 --seed 1 --force
```
