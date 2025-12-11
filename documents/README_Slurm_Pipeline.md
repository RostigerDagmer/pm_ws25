**Branch:** `train_evaluate_ML`

# ML Classifier Training Pipeline

## Overview

```
Step 0: Profile (one-time)  →  Step 1: Generate Labels  →  Step 2: Train & Evaluate Classifier
   ~5-10 minutes                   ~2-4 hours (parallel)        ~1 hour
   Optimize resources               N datasets → CSV files       CSV → trained model + metrics
                                    ⚡ Re-runs: instant (cache)  ⚡ Uses cached data
```

**🆕 Intelligent Caching:**
- **First run:** Full pipeline execution (~2-4 hours for label generation)
- **Re-runs:** Near-instant completion if cache exists (~1-2 minutes)
- **Control:** Use `--force-recompute` flag to regenerate data

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

Before running the full pipeline on datasets, profile a representative dataset to determine optimal Slurm resource allocation.

**⚠️ Important:** If your datasets vary significantly in size, profile the **largest** dataset to avoid resource allocation failures. The Slurm configuration will apply the same resources to all jobs, so you need to allocate for the worst case.

#### 0.1: Run the Profiling Script

##### If you don't know which is largest, check first:
```bash
for dataset in data/*/*.xes; do
    echo "$dataset: $(grep -c "trace" $dataset) traces"
done | sort -t: -k2 -n
```

##### Profile the LARGEST dataset to get upper-bound resource requirements

```bash
cd ~/pm_ws25
source .venv/bin/activate

# Basic profiling with default settings (8 workers, 100 runs)
bash scripts/profile_job.sh data/<uuid>/your-largest-dataset.xes

# OR: Quick profiling with fewer runs (faster, for estimates)
bash scripts/profile_job.sh data/<uuid>/your-largest-dataset.xes 8 50

# OR: Profile with specific worker count
bash scripts/profile_job.sh data/<uuid>/your-largest-dataset.xes 4 100
```

**What happens:**
- Activates Python virtual environment
- Runs `create_labels.py` on one dataset with `--force-recompute` (ensures fresh profiling)
- Uses `/usr/bin/time -v` to track:
  - Maximum memory usage (RSS)
  - Total runtime (wall clock)
  - CPU utilization percentage
- Saves complete output to `results/profile_output.txt`

**Note:** The script uses `--force-recompute` to bypass caching and get accurate resource measurements.

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

After profiling and configuring resources, submit the job array to generate training data from all datasets.

#### 1.1: Submit Job Array

```bash
cd ~/pm_ws25
sbatch lrz-cluster/run_create_labels.slurm
```

**What happens:**
1. **Job Submission**: Slurm creates N separate jobs (one per dataset in DATASETS array)
2. **Parallel Processing**: Jobs run in parallel based on your throttle setting
3. **Smart Caching** (NEW):
   - **Alignment cache**: Uses cached alignment runs (`.pkl` files) if available
   - **Output cache**: Skips entire job if CSV files already exist
   - **Cache location**: `data/runs/<hash>.pkl` and `data/runs/<hash>.{train,test,eval}.csv`
   - ⚡ **Re-runs complete instantly** if cache exists!
4. **Per-Job Workflow** (only if not cached):
   - Loads one XES event log
   - Discovers process models using inductive miner with configured noise thresholds
   - Samples trace variants using configured distributions
   - Generates models: `num_thresholds × num_samplers × n_subsets` per dataset
     - Current config: 4 thresholds × 2 samplers × 100 subsets = 800 models
   - Runs alignment benchmarks with configured traces and runs
   - Saves results to CSV files in `data/runs/<dataset_hash>/`
5. **Deduplication**: Removes duplicate models (typically ~70% unique models remain)

**Expected output:** Each job creates:
- 1 pickle file: `<hash>.pkl` (cached alignment runs, ~100MB)
- 3 CSV files: `<hash>.{train,test,eval}.csv` (features + labels, ~10MB each)
- 2 additional CSV files: `<hash>.{runs,labels}.csv` (detailed data)

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
find data/runs -name "*.train.csv" | wc -l  # Should match number of datasets in DATASETS array
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
# Count training files (should match number of datasets)
find data/runs -name "*.train.csv" | wc -l

# List all generated files
ls -lh data/runs/*/run_*.train.csv
ls -lh data/runs/*/run_*.test.csv
ls -lh data/runs/*/run_*.eval.csv
```

**If any files are missing:** Check the error logs for that specific job and re-run if necessary (see Troubleshooting section).

**⚠️ Do not proceed to Step 2 until all CSV file sets exist (one per dataset)!**

---

### Step 2: Train & Evaluate Classifier (~1 hour)

After all 16 CSV datasets are generated, train the ML classifier to predict algorithm performance.

#### 2.1: Submit Classifier Training Job

```bash
cd ~/pm_ws25
sbatch lrz-cluster/run_evaluate_classifier.slurm
```

**What happens:**
1. **Pre-flight Check**: Verifies that required `.train.csv` and `.test.csv` files exist
2. **Data Loading**:
   - Loads all `.train.csv` files from `data/runs/` for training
   - **Uses cached RunDatasets** from `.pkl` files for test datasets (very fast!)
3. **Feature Engineering**: Combines process model features with historical timing data
4. **Model Training**:
   - Trains XGBoost classifier on aggregated training data
   - Also trains baseline models (SingleBest, Random) for comparison
   - Saves trained models to `cache/models/`
5. **Evaluation**: Tests prediction accuracy on held-out test datasets
6. **Reporting**: Generates summary statistics and comparison to baseline methods

**Expected output:** Performance metrics showing how accurately the classifier predicts which algorithm will be fastest for a given process model.

**Performance:** Thanks to caching, test dataset loading is near-instant (~10 seconds vs ~2 hours without cache).

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

### Understanding the Hash Files

**What is a hash?** Each `.pkl` file in `data/runs/` represents one dataset, identified by a SHA-1 hash of the dataset path.

**Example mapping:**
- Dataset: `data/a0addfda-2044-4541-a450-fdcc9fe16d17/BPIC15_1.xes`
- Hash file: `data/runs/28893bc1b19233185e6a2910b54781d9756f091c.pkl`

**What's inside each `.pkl` file?**
```python
{
    'process_models': [...],      # ~584 unique process models
    'alignments': [...],          # Alignment results (20 traces × 5 runs per model)
    'train_csv': 'path/to/train', # Training data (70%)
    'test_csv': 'path/to/test',   # Test data (20%)
    'eval_csv': 'path/to/eval'    # Evaluation data (10%)
}
```

### What Each Step Produces

**Step 0: Profile**
```
Input:  data/<uuid>/<dataset>.xes (one test dataset)
Output: results/profile_output.txt (resource recommendations)
```

**Step 1: Generate Labels** (runs on all datasets in parallel)
```
Input:  data/<uuid>/<dataset>.xes (N datasets from DATASETS array)
Output: data/runs/<hash>.pkl (cached alignment runs, ~100 MB each)
        data/runs/<hash>.train.csv (training split, ~10 MB)
        data/runs/<hash>.test.csv (test split, ~3 MB)
        data/runs/<hash>.eval.csv (eval split, ~2 MB)
        data/runs/<hash>.runs.csv (all runs, ~50 MB)
        data/runs/<hash>.labels.csv (best aligners, ~15 MB)
        logs/create_labels_<job>_<task>.out (job output)
        logs/create_labels_<job>_<task>.err (job errors)
```

**Step 2: Train & Evaluate Classifier**
```
Input:  data/runs/*.train.csv (training data from Step 1)
        data/runs/*.pkl (cached test datasets, reused from Step 1)
Output: outputs/evaluate_classifier/
        ├── summary.txt (human-readable results)
        ├── metrics.json (detailed metrics)
        └── baseline_comparison.csv (comparison with baselines)
        cache/models/*.pkl (trained classifier models, ~20 MB)
        logs/eval_classifier_<job>.out (training output)
        logs/eval_classifier_<job>.err (training errors)
```

### Quick Access Commands

**Check profiling results:**
```bash
cat results/profile_output.txt
```

**Count generated datasets (Step 1):**
```bash
# Count training CSV files (should match DATASETS array length)
find data/runs -name "*.train.csv" | wc -l

# Show all generated files for a dataset
ls -lh data/runs/*.{pkl,train.csv,test.csv,eval.csv,runs.csv,labels.csv} 2>/dev/null | head -20

# Show file sizes summary
du -sh data/runs/
```

**Inspect a specific dataset:**
```bash
# List what's inside a .pkl file (requires Python)
python3 -c "
import pickle
with open('data/runs/<hash>.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f'Models: {len(data.get(\"process_models\", []))}')
    print(f'Alignments: {len(data.get(\"alignments\", []))}')
"
```

**View classifier results (Step 2):**
```bash
cat outputs/evaluate_classifier/summary.txt
cat outputs/evaluate_classifier/metrics.json
```

**Check job logs:**
```bash
# List all log files
ls -lth logs/

# View specific job output
cat logs/create_labels_<jobid>_<taskid>.out

# Check for errors
cat logs/create_labels_<jobid>_<taskid>.err

# Search for failures across all jobs
grep -l "Exit Code: 1" logs/create_labels_*.out
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
| Need to regenerate data | Use `--force-recompute` flag (see below) |
| Cache seems corrupted | Delete `.pkl` file and re-run with `--force-recompute` |

**Re-run single dataset (uses cache if available):**
```bash
python scripts/create_labels.py \
    --config configs/default.yaml \
    --path data/<uuid>/<file>.xes \
    --seed 1 \
    --train 0.7 \
    --test 0.2 \
    --eval 0.1 \
    --runs 100 \
    --workers 8
```

**Force regenerate (bypass all caches):**
```bash
python scripts/create_labels.py \
    --config configs/default.yaml \
    --path data/<uuid>/<file>.xes \
    --seed 1 \
    --train 0.7 \
    --test 0.2 \
    --eval 0.1 \
    --runs 100 \
    --workers 8 \
    --force-recompute
```

---

## Quick Start Summary

```bash
# 1. Profile (once) - Use --force-recompute for accurate measurements
bash scripts/profile_job.sh data/<uuid>/largest-dataset.xes
cat results/profile_output.txt

# 2. Apply recommendations
nano lrz-cluster/run_create_labels.slurm
# Update lines 22-26 with profile values

# 3. Generate labels (uses cache by default for fast re-runs)
sbatch lrz-cluster/run_create_labels.slurm
find data/runs -name "*.train.csv" | wc -l  # Wait until count matches DATASETS array

# 4. Train classifier (uses cached RunDatasets automatically)
sbatch lrz-cluster/run_evaluate_classifier.slurm

# 5. View results
cat outputs/evaluate_classifier/summary.txt

# Optional: Force regeneration after config changes
# Edit run_create_labels.slurm and add --force-recompute flag
```

**⚡ Pro Tip:** After the first successful run, re-running the same datasets completes in seconds thanks to intelligent caching!

---

## Configuration

### Change Which Datasets to Process

**For Label Generation (Step 1):**

**File:** `lrz-cluster/run_create_labels.slurm` (lines 63-80, 22)

```bash
# 1. Edit DATASETS array (add/remove paths)
DATASETS=(
    "data/uuid1/dataset1.xes"
    "data/uuid2/dataset2.xes"
    # ... add more datasets
)

# 2. Update array size (line 22): --array=0-(N-1)%throttle
# Formula: If you have N datasets, use 0-(N-1)
# Examples:
#   10 datasets: --array=0-9%8
#   16 datasets: --array=0-15%16
#   20 datasets: --array=0-19%8
```

**Re-profile if new dataset is significantly different:**
```bash
# Profile default dataset (BPIC15_1.xes)
bash scripts/profile_job.sh

# OR profile a specific new dataset
bash scripts/profile_job.sh data/your-uuid/your-large-dataset.xes
```

**For Classifier Training/Evaluation (Step 2):**

**File:** `scripts/evaluate_classifier_e2e.py` (lines 51-86)

```python
# Edit TRAIN_DATASETS dictionary to add/remove training datasets
TRAIN_DATASETS = {
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
    # ... add more training datasets
}

# Edit TEST_DATASETS dictionary to add/remove test datasets
TEST_DATASETS = {
    'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
    # Uncomment to enable more test datasets:
    # 'b32c6fe5-f212-4286-9774-58dd53511cf8': ['BPIC15_5.xes'],
}
```

**Important:** Keep these in sync with `run_create_labels.slurm` to ensure all datasets have generated labels before training!

### Change Train/Test/Eval Split

**File:** `lrz-cluster/run_create_labels.slurm` (around lines 123-131)

```bash
python scripts/create_labels.py \
    --train 0.6 \  # Training: 60% (default: 0.7)
    --test 0.3 \   # Test: 30% (default: 0.2)
    --eval 0.1 \   # Eval: 10% (default: 0.1)
    # Note: train + test + eval must equal 1.0
```

### Control Caching Behavior

**Default:** Uses cache (fast, ~1-2 minutes)
**Force regenerate:** Uncomment line 132 in `lrz-cluster/run_create_labels.slurm`

```bash
# Change this line from:
    # --force-recompute  # Uncomment to regenerate all data (bypasses cache)

# To this:
    --force-recompute  # Uncomment to regenerate all data (bypasses cache)
```

**Use `--force-recompute` when:**
- Changed `configs/default.yaml` settings
- Suspect corrupted cache
- Otherwise: Leave commented (saves hours!)

### Change Model Generation

**File:** `configs/default.yaml`

```yaml
discovery:
  params:
    noise_threshold: [0.0, 0.2, 0.4]  # More/fewer thresholds → more/fewer models

  samplers:
    - name: variant1
      n_subsets: 150  # Increase for more models (default: 100)

alignment:
  runs: 10         # More runs → more stable timing (default: 5)
  sampler:
    slice:
      to: 50     # More traces → better features (default: 20)
```

**Effect on training data:**
- `noise_threshold: [0.0, 0.2, 0.4]` → 3 thresholds × 2 samplers × 150 = 900 models
- `n_subsets: 150` → 50% more models per sampler
- `runs: 10` → 2× more alignment measurements
- `to: 50` → 2.5× more traces aligned per model

---

## Data Flow

```
XES Event Logs (N files from DATASETS array)
    ↓
Process Discovery (num_thresholds × num_samplers × n_subsets models)
    ↓
Deduplication (~70% unique models remain)
    ↓
Alignment Benchmarks (configurable traces × runs per model)
    ↓
CSV Datasets (train/test/eval split per dataset)
    ↓
ML Classifier Training (aggregate all datasets)
    ↓
Performance Evaluation
```
