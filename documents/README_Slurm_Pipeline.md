# Slurm Pipeline: Data Generation → Evaluation

## Overview

This pipeline generates training data from real and synthetic sources, then trains an ML classifier to recommend the fastest alignment algorithm.

### Three Optimized Slurm Scripts

| Script | Purpose | Partition | CPUs | Memory | Runs | Expected Runtime |
|--------|---------|-----------|------|--------|------|------------------|
| **run_create_labels_parallel.slurm** | Real data (21 XES datasets) | cm4_inter | 1029<br>(21 datasets × 49 CPUs) | 512GB | 10 | **~6.1 hours** |
| **run_create_labels_synthetic.slurm** | Synthetic data | cm4_inter | 96 | 256GB | 10 | **30-40 min** |
| **run_evaluate_classifier.slurm** | Train & test classifier | serial_std | 32 | 128GB | - | **30-45 min** |

**Total pipeline time: ~7-8 hours** (sequential execution due to MaxJobs=1 and MaxSubmit=2 limits)

---

## Quick Start (Recommended)

**Note:** Due to MaxJobs=1 and MaxSubmit=2 limits, run jobs sequentially:

```bash
cd ~/pm_ws25

# Step 1: Generate synthetic data first (faster, ~30-40 min)
sbatch lrz-cluster/run_create_labels_synthetic.slurm

# Step 2: After synthetic job completes, generate real data (~6.1 hours)
sbatch lrz-cluster/run_create_labels_parallel.slurm

# Step 3: After both complete, run evaluation (~30-45 min)
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
- **Single Slurm job** with **1029 CPUs** total (cm4_inter partition)
- **ALL 21 datasets processed in parallel** (49 CPUs each)
- **512GB RAM** total
- **10 alignment runs** per model-trace pair (`--runs 10`)
- **5 alignment algorithms** tested
- **1 batch** to process all 21 datasets simultaneously

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
tail -f logs/create_labels_*_*.out  # For array jobs

# Check generated files
∂
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
- Needs: **1 Slurm job** with **1029 CPUs** (cm4_inter)
- Runs: ALL 21 datasets in parallel (49 CPUs each) within the single job
- Processes 21 datasets in 1 batch
- Fits in cm4_inter: ✅ (uses ~99% of available CPUs)
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
- Real data: Each job has 80GB (sufficient for all datasets)
- Synthetic: Job has 350GB (sufficient for 192 workers)

**Slow execution:**
- Verify CPU allocation: `squeue -u $USER -o "%.18i %.9P %.8T %C"`
- Check logs for errors or warnings

---

## Summary

✅ **Optimized for maximum parallelism:** Uses cm4_inter partition with 1029 CPUs
✅ **Sequential job execution:** Jobs run one at a time due to MaxJobs=1 limit
✅ **Parallel dataset processing:** Bash background jobs process ALL 21 datasets simultaneously within single job (MaxSubmit=2 workaround)
✅ **Completion time:** ~6.1 hours for all 21 datasets with 10 alignment runs per combination
✅ **High quality:** 10 alignment runs (`--runs 10`) for robust statistical results
✅ **Intelligent caching:** Reuses alignment results across runs (use `--force-recompute` for first run)
✅ **Automatic hybrid training:** Combines all CSV files automatically

