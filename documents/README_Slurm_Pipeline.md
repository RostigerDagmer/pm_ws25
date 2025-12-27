# Slurm Pipeline: Data Generation → Evaluation

## Overview

This pipeline generates training data from real and synthetic sources, then trains an ML classifier to recommend the fastest alignment algorithm.

### Three Optimized Slurm Scripts

| Script | Purpose | Partition | CPUs | Memory | Runs | Expected Runtime |
|--------|---------|-----------|------|--------|------|------------------|
| **run_create_labels_parallel.slurm** | Real data (21 XES datasets) | cm4_inter | 224<br>(7 datasets × 32 CPUs per batch) | 480GB | 5 | **~6.6 hours** |
| **run_create_labels_synthetic.slurm** | Synthetic data | cm4_inter | 96 | 256GB | 10 | **30-40 min** |
| **run_evaluate_classifier.slurm** | Train & test classifier | serial_std | 32 | 128GB | - | **5 min** |

---

## Quick Start (Recommended)

**Note:** Due to MaxJobs=1 and MaxSubmit=2 limits, run jobs sequentially:

```bash
cd ~/pm_ws25

# Step 1: Generate synthetic data
sbatch lrz-cluster/run_create_labels_synthetic.slurm

# Step 2: Generate real data
sbatch lrz-cluster/run_create_labels_parallel.slurm

# Step 3: Train ML Model & Run evaluation
sbatch lrz-cluster/run_evaluate_classifier.slurm

# Step 4: Check results
cat outputs/evaluate_classifier/summary.txt
```

---

## Step 1: Generate Training Data

### 1.1 Real Dataset Processing

**Script:** `run_create_labels_parallel.slurm`

**What it does:**
- Processes 21 XES event log files using bash background jobs
- Uses 1 Slurm job to work around MaxSubmit=2 limit
- Each dataset: discovery → deduplication → alignment → feature extraction
- Outputs to: `cache/.runs/`

**Configuration:**
- **Single Slurm job** with **224 CPUs** total (cm4_inter partition)
- **21 datasets processed in 3 sequential batches** (32 CPUs each, 7+7+7 per batch)
- **Uses /tmp (1.5TB) for temporary cache** to avoid home directory quota limits
- **480GB RAM** total
- **5 alignment runs** per model-trace pair (`--runs 5`)
- **5 alignment algorithms** tested
- **3 batches** to process all 21 datasets (batch size = 7)

**Submit:**
```bash
sbatch lrz-cluster/run_create_labels_parallel.slurm
```

**Expected output (per dataset):**
```
cache/.runs/<hash>.train.csv   # Training samples (70%)
cache/.runs/<hash>.test.csv    # Testing samples (20%)
cache/.runs/<hash>.eval.csv    # Evaluation samples (10%)
cache/.runs/<hash>.runs.csv    # All alignment runs
cache/.runs/<hash>.labels.csv  # Best aligners per combination
```

---

### 1.2 Synthetic Data Generation

**Script:** `run_create_labels_synthetic.slurm`

**What it does:**
- Generates 100 synthetic process models
- Simulates 40 traces per model
- Runs 5 alignment algorithms × 10 runs per combination
- Outputs to: `cache/.runs_synthetic/`

**Configuration:**
- **1 job** with **96 CPUs**
- **256GB RAM**
- **200,000 alignments** total (optimized workload)

**Submit:**
```bash
sbatch lrz-cluster/run_create_labels_synthetic.slurm
```

**Expected output:**
```
cache/.runs_synthetic/<hash>.train.csv  # Training samples
cache/.runs_synthetic/<hash>.test.csv   # Testing samples
cache/.runs_synthetic/<hash>.eval.csv   # Evaluation samples
```

---

### 1.3 Monitor Data Generation (optional)

```bash
# Check running jobs
squeue -u $USER

# Watch real-time logs
tail -f logs/create_labels_synthetic_*.out
tail -f logs/create_labels_all_*.out  # For parallel processing job

# Check generated files
ls -lh cache/.runs_synthetic/*.train.csv
```

---

## Step 2: Train & Evaluate Classifier

### 2.1 Run Evaluation

**Script:** `run_evaluate_classifier.slurm`

**Prerequisites:** At least one of the data generation scripts must complete successfully.

**What it does:**
1. **Automatically searches** for CSV files in:
   - `cache/.runs/` (real data)
   - `cache/.runs_synthetic/` (synthetic data)
   - `data/runs/` (legacy location) (optional)

2. **Combines all data** for training (hybrid training)

3. **Trains XGBoost classifier** to predict best aligner

4. **Evaluates** on test datasets

5. **Compares** with baseline methods

**Configuration:**
- **32 CPUs** for parallel XGBoost training (respects QOS limit)
- **128GB RAM** for large dataset handling

**Submit:**
```bash
sbatch lrz-cluster/run_evaluate_classifier.slurm
```
---

### 2.2 Check Results

**Log output shows:**
```
INFO: Loading pre-computed CSV tables...
INFO:   Searching in: cache/.runs
INFO:     Found: 21 train, 21 test, 21 eval tables
INFO:   Searching in: cache/.runs_synthetic
INFO:     Found: 1 train, 1 test, 1 eval tables
INFO: Training XGBoostClassifier...
INFO: Total: ~X training samples
```

**Output files:**
```
outputs/evaluate_classifier/
├── summary.txt              # Main results summary
├── comparison.csv           # Classifier vs baselines
└── metrics.json             # Detailed metrics
```

**View results:**
```bash
cat outputs/evaluate_classifier/summary.txt
```

---

## Data Output Locations

```
pm_ws25/
├── cache/
│   ├── .runs/                    # Real data (from run_create_labels.slurm)
│   │   ├── <hash>.train.csv      # 21 datasets × 5 files each
│   │   ├── <hash>.test.csv
│   │   ├── <hash>.eval.csv
│   │   ├── <hash>.runs.csv
│   │   └── <hash>.labels.csv
│   │
│   └── .runs_synthetic/          # Synthetic data (from run_create_labels_synthetic.slurm)
│       ├── <hash>.train.csv      # 1 combined synthetic dataset
│       ├── <hash>.test.csv
│       ├── <hash>.eval.csv
│       ├── <hash>.runs.csv
│       └── <hash>.labels.csv
│
└── outputs/
    └── evaluate_classifier/      # Evaluation results
        ├── summary.txt
        ├── comparison.csv
        └── metrics.json
```

---

## CPU Resource Requirements

### Available Resources:
- **serial_std partition:** 1,616 CPUs available
- **cm4_inter partition:** ~932 idle CPUs available (1,344 total)

### Account Limits:
- **MaxJobs:** 1 (only 1 job can run at a time)
- **MaxSubmit:** 2 (only 2 jobs can be submitted/queued at once)

### Resource Usage:

**Real Data Processing (bash background jobs approach):**
- Needs: **1 Slurm job** with **224 CPUs** (cm4_inter)
- Runs: 21 datasets in **3 sequential batches** (32 CPUs each, batches of 7+7+7)
- Processes 7 datasets at a time per batch
- Uses /tmp (1.5TB) for cache to avoid disk quota limits
- Fits in cm4_inter: ✅ (uses ~24% of partition CPUs)
- **Workaround for MaxSubmit=2 limit** ✅

**Synthetic Data Generation:**
- Needs: **96 CPUs** (cm4_inter)
- Fits in cm4_inter: ✅ (836 CPUs headroom)

**Evaluation:**
- Needs: **32 CPUs** (serial_std)
- Fits in serial_std: ✅ (1,584 CPUs headroom)

---

## Caching Behavior

All scripts use intelligent caching to avoid recomputing alignments:
- **First run:** Requires `--force-recompute` flag (uncomment in script) to generate alignment data from scratch
- **Subsequent runs:** Reuses cached alignment results unless you uncomment `--force-recompute` to regenerate everything

### Disk Quota Optimization

**run_create_labels_parallel.slurm** uses a smart caching strategy:
- **During execution:** Writes all cache directories to `/tmp` (1.5TB available, no quota limits)
- **After completion:** Automatically moves cache back to `~/pm_ws25/cache/` (`.runs`, `.cache_unique_models`, `.cache_process_models`)
- **Benefit:** Avoids home directory disk quota issues during large parallel writes

To regenerate data (e.g., after changing parameters), uncomment the `--force-recompute` line in the respective Slurm script.

---

## Troubleshooting

### Check job status
```bash
squeue -u $USER
```

### View logs
```bash
# Synthetic data
tail -f logs/create_labels_synthetic_*.out
tail -f logs/create_labels_synthetic_*.err

# Real data (parallel processing)
tail -f logs/create_labels_all_*.out
tail -f logs/dataset_*_1.out  # Watch first dataset
ls logs/dataset_*.out  # List all per-dataset logs

# Evaluation
tail -f logs/eval_classifier_*.out
```

### Check generated data
```bash
# Count CSV files
ls cache/.runs/*.train.csv | wc -l           # Should be 21
ls cache/.runs_synthetic/*.train.csv | wc -l # Should be 1

# Check file sizes
ls -lh cache/.runs/*.train.csv
ls -lh cache/.runs_synthetic/*.train.csv
```

### Common issues

**Job not starting:**
- Check partition availability: `sinfo`
- Check resource limits: ensure you're not exceeding quotas

**Out of memory:**
- Real data: Job has 480GB RAM (sufficient for 7 datasets in parallel)
- Synthetic: Job has 256GB RAM (sufficient for 96 workers)

**Slow execution:**
- Verify CPU allocation: `squeue -u $USER -o "%.18i %.9P %.8T %C"`
- Check logs for errors or warnings

**Pickle data truncation error:**

**Symptom:**
```
_pickle.UnpicklingError: pickle data was truncated
```

**Root cause:**
- Occurs during parallel batch processing when multiple datasets compete for shared cache files
- Process model cache files (`.cache_process_models/*.pkl`) can be corrupted due to concurrent write conflicts
- Truncated files typically have suspicious sizes (e.g., exactly 4MB = 2^22 bytes)

**How to fix:**

1. **Identify failed datasets:**
```bash
# Check job summary for failures
tail -50 logs/create_labels_all_*.out | grep -A5 "SUMMARY"

# Find datasets with errors
grep -l "pickle data was truncated" logs/dataset_*.out
```

2. **Delete corrupted cache files:**

**Option A: If only process model cache is corrupted (pkl can be reused):**
```bash
# Example: If dataset hash is db35afac-2133-40f3-a565-2dc77a9329a3
rm -fv cache/.cache_process_models/db35afac-2133-40f3-a565-2dc77a9329a3.pkl
```

**Option B: If pkl backup is also corrupted (must reprocess from scratch):**
```bash
# Example: If dataset hash is fb84cf2d-166f-4de2-87be-62ee317077e5
# Delete ALL corrupted files including pkl backups
rm -v ~/pm_ws25/cache/.runs/fb84cf2d-166f-4de2-87be-62ee317077e5.*
rm -v ~/pm_ws25/cache/.cache_process_models/fb84cf2d-166f-4de2-87be-62ee317077e5.pkl
rm -v ~/pkl_backup/fb84cf2d-166f-4de2-87be-62ee317077e5.pkl
```

3. **Modify and run single-dataset script:**

Edit `lrz-cluster/run_permitlog.slurm` and change the `DATASET` variable (around line 89):

```bash
DATASET="data/<HASH>/<DATASET_NAME>.xes"
```

**Examples:**

PermitLog (Option A - reuses cache):
```bash
DATASET="data/db35afac-2133-40f3-a565-2dc77a9329a3/PermitLog.xes"
```

PrepaidTravelCost (Option B - processes from scratch):
```bash
DATASET="data/fb84cf2d-166f-4de2-87be-62ee317077e5/PrepaidTravelCost.xes"
```

Then submit:
```bash
sbatch lrz-cluster/run_permitlog.slurm
```

This script uses all 224 CPUs for the single dataset (avoiding parallel competition). For Option A, it reuses existing cache. For Option B, it processes from scratch without corrupted pkl files.

