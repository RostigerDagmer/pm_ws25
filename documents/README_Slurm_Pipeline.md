**Branch:** `train_evaluate_ML`

# ML Classifier Training Pipeline

## Overview

```
Step 0: Profile (one-time)  →  Step 1: Generate Labels  →  Step 2: Train Classifier
   ~5-10 minutes                   ~2-4 hours (parallel)        ~1 hour
   Optimize resources               16 datasets → CSV files      CSV → trained model
```

---

## Complete Step-by-Step Guide

### Prerequisites

Ensure you're on the correct branch:

```bash
cd ~/pm_ws25
git branch  # Should show: * train_evaluate_ML
```

---

### Step 0: Profile Resources (~5-10 minutes)

Before running the full pipeline on datasets, profile a single dataset to determine optimal Slurm resource allocation.

#### 0.1: Run the Profiling Script

```bash
cd ~/pm_ws25
bash scripts/profile_job.sh
```

**What happens:**
- Activates Python virtual environment
- Runs `create_labels.py` on one dataset
- Uses `/usr/bin/time -v` to track:
  - Maximum memory usage (RSS)
  - Total runtime (wall clock)
  - CPU utilization percentage
- Saves complete output to `results/profile_output.txt`

---

#### 0.2: View Profiling Results

```bash
cat results/profile_output.txt
```

**What to look for:**
The script outputs recommendations at the end:

```
============================================
RECOMMENDATIONS FOR SLURM CONFIG
============================================

Edit lrz-cluster/run_create_labels.slurm:

  #SBATCH --cpus-per-task=8
  #SBATCH --mem=32G
  #SBATCH --time=03:00:00
```

**What these mean:**
- `--cpus-per-task`: Number of CPU cores needed per job
- `--mem`: Memory required per job
- `--time`: Maximum runtime before job is killed

---

#### 0.3: Apply Recommendations to Slurm Config

Edit the Slurm configuration file:

```bash
nano lrz-cluster/run_create_labels.slurm
```

**Update these lines with your profiling results:**

| Line | Parameter | What to Change | Example Value |
|------|-----------|----------------|---------------|
| 22 | `--array` | Job array and throttle | `#SBATCH --array=0-15%8` |
| 23 | `--time` | Maximum runtime | `#SBATCH --time=03:00:00` |
| 25 | `--cpus-per-task` | CPU cores per job | `#SBATCH --cpus-per-task=8` |
| 26 | `--mem` | Memory per job | `#SBATCH --mem=32G` |

**Setting the throttle (line 22):**

First, check cluster availability:
```bash
sinfo -o "%P %a %T %c"
```

Then choose throttle based on idle CPUs:
- **Idle CPUs > 128**: `#SBATCH --array=0-15%16` → Run 16 jobs in parallel (FAST, ~2 hours)
- **Idle CPUs > 64**: `#SBATCH --array=0-15%8` → Run 8 jobs in parallel (BALANCED, ~4 hours)
- **Idle CPUs > 32**: `#SBATCH --array=0-15%4` → Run 4 jobs in parallel (CONSERVATIVE, ~8 hours)

**What the throttle does:**
The `%N` suffix limits concurrent jobs. For example, `--array=0-15%8` means "run 16 tasks total (0-15), but only 8 at a time." This prevents overwhelming the cluster.

**Save and exit:** Press `Ctrl+X`, then `Y`, then `Enter`

---

### Step 1: Generate Training Labels (~2-4 hours)

After profiling and configuring resources, submit the job array to generate training data from all 16 datasets.

#### 1.1: Submit Job Array

```bash
cd ~/pm_ws25
sbatch lrz-cluster/run_create_labels.slurm
```

**What happens:**
1. **Job Submission**: Slurm creates 16 separate jobs (array indices 0-15)
2. **Parallel Processing**: Jobs run in parallel based on your throttle setting
3. **Per-Job Workflow**:
   - Loads one XES event log
   - Discovers process models using inductive miner with noise thresholds [0.0, 0.1, 0.2, 0.3]
   - Samples trace variants using configured distributions
   - Generates ~100 process models per dataset
   - Runs alignment benchmarks (20 traces × 5 runs each)
   - Saves results to CSV files in `data/runs/<dataset_hash>/`
4. **Deduplication**: Removes duplicate models (typically reduces 800 → 584 unique)
5. **Caching**: Automatically skips datasets with existing CSV files

**Expected output:** Each job creates 3 CSV files (train/test/eval) containing features and timing labels for ML training.

---

#### 1.2: Monitor Progress

**Check job status:**
```bash
squeue -u $USER  # Shows running/pending jobs
```

Expected output:
```
JOBID   PARTITION  NAME              USER     ST  TIME  NODES  NODELIST
123456  serial     create_labels     user     R   1:23  1      node001
```

**Watch real-time logs:**
```bash
tail -f logs/create_labels_*.out  # Live output from all jobs
```

**Count completed datasets:**
```bash
find data/runs -name "*.train.csv" | wc -l  # Should reach 16
```

**Check for errors:**
```bash
grep -l "Exit Code: 1" logs/create_labels_*.out  # Lists failed jobs
cat logs/create_labels_<jobid>_<taskid>.err      # View specific error
```

---

#### 1.3: Verify Completion

**Before proceeding to Step 2, ensure all datasets completed successfully:**

```bash
# Should show 16 training files
ls -lh data/runs/*/run_*.train.csv

# Should show 16 test files
ls -lh data/runs/*/run_*.test.csv

# Should show 16 evaluation files
ls -lh data/runs/*/run_*.eval.csv
```

**If any files are missing:** Check the error logs for that specific job and re-run if necessary (see Troubleshooting section).

**⚠️ Do not proceed to Step 2 until all 16 CSV file sets exist!**

---

### Step 2: Train & Evaluate Classifier (~1 hour)

After all 16 CSV datasets are generated, train the ML classifier to predict algorithm performance.

#### 2.1: Submit Classifier Training Job

```bash
cd ~/pm_ws25
sbatch lrz-cluster/run_evaluate_classifier.slurm
```

**What happens:**
1. **Data Loading**: Scans `data/runs/` for all `.train.csv` and `.test.csv` files
2. **Feature Engineering**: Combines process model features with historical timing data
3. **Model Training**:
   - Trains gradient boosting classifier on aggregated training data
   - Uses cross-validation for hyperparameter tuning
   - Saves trained model to disk
4. **Evaluation**: Tests prediction accuracy on held-out evaluation sets
5. **Reporting**: Generates summary statistics and comparison to baseline methods

**Expected output:** Performance metrics showing how accurately the classifier predicts which algorithm will be fastest for a given process model.

---

#### 2.2: Monitor Training

**Check job status:**
```bash
squeue -u $USER  # Should show eval_classifier job running
```

**Watch training progress:**
```bash
tail -f logs/eval_classifier_*.out
```

Expected log messages:
- "Loading training data from N datasets..."
- "Training classifier..."
- "Evaluating on test set..."
- "Writing results..."

---

#### 2.3: View Results

**Once job completes, view the summary:**

```bash
cat outputs/evaluate_classifier/summary.txt
```

**Expected contents:**
- Training dataset statistics (number of samples, features)
- Model performance metrics (accuracy, F1-score, precision/recall)
- Comparison to baseline predictors
- Feature importance rankings

**View detailed metrics:**
```bash
cat outputs/evaluate_classifier/metrics.json
```

---

## Output Locations

### Complete File Structure

| Step | Directory | Files | Full Path Example | Description |
|------|-----------|-------|-------------------|-------------|
| **Step 0** | `results/` | `profile_output.txt` | `~/pm_ws25/results/profile_output.txt` | Resource usage metrics and Slurm recommendations |
| **Step 1** | `data/runs/<hash>/` | `run_<id>.train.csv` | `~/pm_ws25/data/runs/a0addfda-2044.../run_123.train.csv` | Training features and labels (70% split) |
| | | `run_<id>.test.csv` | `~/pm_ws25/data/runs/a0addfda-2044.../run_123.test.csv` | Test features and labels (20% split) |
| | | `run_<id>.eval.csv` | `~/pm_ws25/data/runs/a0addfda-2044.../run_123.eval.csv` | Evaluation features and labels (10% split) |
| | | `process_models.pkl` | `~/pm_ws25/data/runs/a0addfda-2044.../process_models.pkl` | Cached discovered process models |
| | | `dedup_mapping.json` | `~/pm_ws25/data/runs/a0addfda-2044.../dedup_mapping.json` | Model deduplication metadata |
| **Step 2** | `outputs/evaluate_classifier/` | `summary.txt` | `~/pm_ws25/outputs/evaluate_classifier/summary.txt` | Human-readable performance report |
| | | `metrics.json` | `~/pm_ws25/outputs/evaluate_classifier/metrics.json` | Detailed metrics in JSON format |
| | | `model.pkl` | `~/pm_ws25/outputs/evaluate_classifier/model.pkl` | Trained classifier (serialized) |
| **Logs** | `logs/` | `create_labels_<job>_<task>.out` | `~/pm_ws25/logs/create_labels_123456_0.out` | Stdout from Step 1 job array |
| | | `create_labels_<job>_<task>.err` | `~/pm_ws25/logs/create_labels_123456_0.err` | Stderr from Step 1 job array |
| | | `eval_classifier_<job>.out` | `~/pm_ws25/logs/eval_classifier_789012.out` | Stdout from Step 2 training |
| | | `eval_classifier_<job>.err` | `~/pm_ws25/logs/eval_classifier_789012.err` | Stderr from Step 2 training |

### Quick Access Commands

```bash
# View profiling results
cat results/profile_output.txt

# Count generated CSV files (should be 16 each)
find data/runs -name "*.train.csv" | wc -l
find data/runs -name "*.test.csv" | wc -l
find data/runs -name "*.eval.csv" | wc -l

# List all training datasets with sizes
ls -lh data/runs/*/run_*.train.csv

# View classifier results
cat outputs/evaluate_classifier/summary.txt
cat outputs/evaluate_classifier/metrics.json

# Check latest logs
ls -lt logs/ | head -10
```

---

## Essential Commands

**Check jobs:**
```bash
squeue -u $USER                  # Your running jobs
scontrol show job <jobid>        # Job details
scancel -u $USER                 # Cancel all jobs
```

**Monitor progress:**
```bash
tail -f logs/create_labels_*.out              # Watch logs
find data/runs -name "*.train.csv" | wc -l    # Count completed datasets
```

**Check for errors:**
```bash
grep "Exit Code: 1" logs/create_labels_*.out  # Find failed jobs
cat logs/create_labels_<jobid>_<taskid>.err   # View error details
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Job timeout | Increase `--time` in Slurm config based on profiling |
| Out of memory | Increase `--mem` in Slurm config based on profiling |
| "No valid training samples" | Run Step 1 first - CSV files missing |
| Job fails | Check error log: `cat logs/create_labels_*_*.err` |

**Re-run single dataset:**
```bash
python scripts/create_labels.py \
    --config configs/default.yaml \
    --path data/<uuid>/<file>.xes \
    --workers 8 --seed 1
```

---

## Quick Start Summary

```bash
# 1. Profile (once)
bash scripts/profile_job.sh
cat results/profile_output.txt

# 2. Apply recommendations
nano lrz-cluster/run_create_labels.slurm
# Update lines 22-26 with profile values

# 3. Generate labels
sbatch lrz-cluster/run_create_labels.slurm
find data/runs -name "*.train.csv" | wc -l  # Wait for 16

# 4. Train classifier
sbatch lrz-cluster/run_evaluate_classifier.slurm

# 5. View results
cat outputs/evaluate_classifier/summary.txt
```

---

## Scaling to More Datasets

**Add new datasets:**

1. Edit `lrz-cluster/run_create_labels.slurm`
2. Add to DATASETS array (lines 63-80):
```bash
DATASETS+=(
    "data/your-uuid/your-dataset.xes"
)
```
3. Update array size (line 22): `#SBATCH --array=0-17%8` (for 18 total)

**Re-profile if:**
- Dataset size changes significantly
- Switching to synthetic data
- Resource requirements different

```bash
bash scripts/profile_job.sh data/path/to/new_dataset.xes
```

---

## Data Flow

```
XES Event Logs (16 files)
    ↓
Process Discovery (800 models → 584 unique)
    ↓
Alignment Benchmarks (20 traces × 5 runs)
    ↓
CSV Datasets (train/test/eval)
    ↓
ML Classifier Training
    ↓
Performance Evaluation
```