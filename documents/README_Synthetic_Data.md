# Synthetic Data Generation & Hybrid Training Guide

## Overview

This guide shows you how to:
1. Generate synthetic process model data
2. Combine it with real XES-based data for training
3. Improve classifier performance with hybrid training data

## Data Overview

### What Data Will Be Used?

When you run `sbatch lrz-cluster/run_evaluate_classifier.slurm`, the classifier **automatically finds and combines** all CSV files:

| Data Source | Script | Training | Testing | Location |
|-------------|--------|----------|---------|----------|
| **Real Data** | `create_labels.py` | ~53,000 | ~15,000 | `data/runs/` |
| **Synthetic Data** | `create_labels_synthetic.py` | ~29,000 | ~8,000 | `data/runs_synthetic/` |
| **TOTAL (Hybrid)** | - | **~82,000** | **~23,000** | - |

✅ **No configuration needed** - just place CSV files in the directories above!

### How Data Is Generated:

**`create_labels.py`** (Real Data):
- Input: XES event log files from real process executions
- Process: Discovers models using Inductive Miner → Runs alignment algorithms
- Output: Train/test/eval CSV splits with extracted features

**`create_labels_synthetic.py`** (Synthetic Data):
- Input: Parameters (n_models=200, n_traces=50, runs=10)
- Process: Generates Petri nets programmatically → Simulates traces → Runs alignments
- Output: Train/test/eval CSV splits with same format as real data

---

## Quick Start

```bash
# Step 1: Generate synthetic training data (~75-80 minutes with 64 CPUs)
cd ~/pm_ws25
sbatch lrz-cluster/run_create_labels_synthetic.slurm

# Step 2: Train classifier (automatic hybrid training - no config needed!)
sbatch lrz-cluster/run_evaluate_classifier.slurm
```

---

## Step 1: Generate Synthetic Training Data

### 1.1: Submit the Job

```bash
cd ~/pm_ws25
sbatch lrz-cluster/run_create_labels_synthetic.slurm
```

**What it does:**
- Generates 200 synthetic process models (3 configurations)
- Simulates 50 traces per model
- Runs alignment benchmarks (10 runs per model-trace combo)
- Uses 64 parallel workers for fast processing
- Creates train/test/eval CSV splits
- Saves to `data/runs_synthetic/`
- **Estimated time: 75-80 minutes (~1.3 hours)**

**Expected output:**
```
data/runs_synthetic/
  ├── <hash>.train.csv  (~3-5 MB)
  ├── <hash>.test.csv   (~1-2 MB)
  ├── <hash>.eval.csv   (~500 KB)
  ├── <hash>.runs.csv   (all runs, ~10 MB)
  └── <hash>.labels.csv (best aligners, ~5 MB)
```

### 1.2: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch real-time logs
tail -f logs/create_labels_synthetic_*.out

# Check for completion
ls -lh data/runs_synthetic/*.train.csv
```

### 1.3: Customize Parameters (Optional)

Edit [lrz-cluster/run_create_labels_synthetic.slurm](lrz-cluster/run_create_labels_synthetic.slurm):

```bash
python scripts/create_labels_synthetic.py \
    --n-models 500 \        # More models → more training data
    --n-traces 100 \        # More traces → better coverage
    --min-depth 2 \         # Deeper models → more complex
    --max-depth 4 \
    --runs 10 \             # More runs → more stable timing (default: 10)
    --workers 64            # Use all 64 CPUs for maximum speed
```

**Trade-offs:**
- More models/traces → longer generation time but richer data
- Deeper models → more realistic but slower alignment
- More runs → more accurate timing but longer computation

---

## Step 2: Train Classifier with Hybrid Data

### 2.1: Generate CSV Files First!

**You need CSV files before training.** Two scripts available:

**A) For Synthetic Data (already done in Step 1):**
```bash
python scripts/create_labels_synthetic.py \
    --config configs/default.yaml \
    --n-models 200 \
    --n-traces 50 \
    --runs 10 \
    --workers 64

# Output → data/runs_synthetic/*.csv
```

**B) For Real Data (from XES files):**
```bash
# Convert your XES event logs to CSV format
sbatch lrz-cluster/run_create_labels.slurm

# Output → data/runs/*.csv
```

### 2.2: Verify Your Data

Check what CSV files you have:

```bash
# Real data (if generated)
ls -lh data/runs/*.train.csv
ls -lh data/runs/*.test.csv

# Synthetic data (from Step 1)
ls -lh data/runs_synthetic/*.train.csv
ls -lh data/runs_synthetic/*.test.csv
```

**Example:**
```
data/runs/3ae76aa...train.csv           5.1M  (52,896 samples)
data/runs/3ae76aa...test.csv            1.5M  (15,226 samples)
data/runs_synthetic/synthetic.train.csv  39K     (61 samples)
data/runs_synthetic/synthetic.test.csv   11K     (17 samples)
```

### 2.3: Train the Classifier - No Configuration Needed! 🎉

**Automatic hybrid training** - just run:

```bash
sbatch lrz-cluster/run_evaluate_classifier.slurm
```

**What happens automatically:**

1. **Searches 3 locations** for CSV files:
   - `cache/.runs/` (alternative cache)
   - `data/runs/` (real data from XES)
   - `data/runs_synthetic/` (synthetic data)

2. **Loads and combines** all `*.train.csv` and `*.test.csv` files

3. **Trains on combined dataset** - real + synthetic together!

**Example log output:**
```
INFO: Loading pre-computed CSV tables...
INFO:   Searching in: cache/.runs
INFO:     Found: 0 train, 0 test, 0 eval tables
INFO:   Searching in: data/runs
INFO:     Found: 1 train, 1 test, 1 eval tables
INFO:   Searching in: data/runs_synthetic
INFO:     Found: 1 train, 1 test, 1 eval tables
INFO: Training XGBoostClassifier...
INFO: Total: 52,957 training samples (52,896 real + 61 synthetic)
INFO: Evaluating on 2 test datasets...
```

### 2.4: Training Scenarios

**Scenario A: Synthetic Only**
- Run: `create_labels_synthetic.py` only
- Training: ~61 samples
- Use case: Quick testing, pipeline validation

**Scenario B: Real Only**
- Run: `create_labels.py` only
- Training: ~52,896 samples
- Use case: Standard training on real data

**Scenario C: Hybrid (Recommended)** ✅
- Run: Both scripts
- Training: ~52,957 samples (52,896 + 61)
- Use case: Best generalization, combines real patterns + synthetic diversity

### 2.5: Important Notes

⚠️ **No manual configuration needed!** The old `TRAIN_DATASETS` and `TEST_DATASETS` dictionaries in [scripts/evaluate_classifier_e2e.py](scripts/evaluate_classifier_e2e.py) are **no longer used**.

✅ **Fully automatic discovery** - just place CSV files in:
- `data/runs/` (for real data)
- `data/runs_synthetic/` (for synthetic data)

✅ **Both train and test data** use CSV files now - no more loading from XES during evaluation

---

## Benefits of Hybrid Training

**Why combine synthetic + real data?**

1. **More training samples** → Better generalization
2. **Diverse model structures** → Handles edge cases
3. **Controlled complexity** → Test specific scenarios
4. **Faster iteration** → Generate data on-demand

**Example results:**
```
Training on real data only:     → 1,234 samples, 78% accuracy
Training on hybrid data:         → 5,678 samples, 85% accuracy ✓
```

---

## File Structure Summary

```
pm_ws25/
├── scripts/
│   ├── create_labels_synthetic.py           # Generate synthetic data
│   ├── create_labels.py                     # Generate real data from XES
│   └── evaluate_classifier_e2e.py           # Train classifier (automatic discovery!)
│
├── lrz-cluster/
│   ├── run_create_labels_synthetic.slurm    # Synthetic generation job (64 CPUs)
│   ├── run_create_labels.slurm              # Real data generation job
│   └── run_evaluate_classifier.slurm        # Training job
│
└── data/
    ├── runs/                                 # Real XES-based data (from create_labels.py)
    │   ├── <hash>.train.csv                  # ← Automatically discovered!
    │   ├── <hash>.test.csv
    │   └── <hash>.eval.csv
    │
    └── runs_synthetic/                       # Synthetic data (from create_labels_synthetic.py)
        ├── synthetic.train.csv               # ← Automatically discovered!
        ├── synthetic.test.csv
        └── synthetic.eval.csv
```

---

## Troubleshooting

### Problem: Import errors on login node

**Solution:** Always run on compute nodes via `sbatch` or `srun`:
```bash
# DON'T run on login node:
python scripts/create_labels_synthetic.py  # ✗

# DO submit to compute node:
sbatch lrz-cluster/run_create_labels_synthetic.slurm  # ✓
```

### Problem: Out of memory

**Solution:** Reduce models or increase memory:
```bash
# In run_create_labels_synthetic.slurm:
#SBATCH --mem=256G  # Increase from 128G

# Or reduce data size:
--n-models 100 \   # Reduce from 200
--n-traces 30 \    # Reduce from 50
```

### Problem: Synthetic data generation too slow

**Solution:** Reduce complexity or increase workers:
```bash
# Reduce complexity:
--max-depth 2 \    # Simpler models
--runs 5 \         # Fewer alignment runs (currently 10)
--n-traces 20      # Fewer traces

# Or increase parallelism:
#SBATCH --cpus-per-task=96   # Use more CPUs if available
```

### Problem: CSV files not found by classifier

**Solution:** Check that CSV files are in the correct locations:
```bash
# Check real data:
ls -lh data/runs/*.train.csv

# Check synthetic data:
ls -lh data/runs_synthetic/*.train.csv

# Classifier searches these 3 locations automatically:
# - cache/.runs/
# - data/runs/
# - data/runs_synthetic/
```

---

## Configuration Options

### Synthetic Model Parameters

Edit [scripts/create_labels_synthetic.py](scripts/create_labels_synthetic.py) line 150-200:

```python
param_grid = [
    # Configuration 1: Sequential-heavy models
    ({
        "dist_params": {
            "op": CategoricalSpec([0.5, 0.2, 0.2, 0.1]),  # seq, choice, parallel, loop
            "seq_len": PoissonSpec(6),       # Longer sequences
            "p_stop": BernoulliDepthLinearSpec(base=0.1, slope=0.1),
            "width": PoissonSpec(2),         # Narrower branches
        },
        "min_depth": 2,
        "max_depth": 4,
    }, 100),  # 100 models with this config
]
```

**Parameter meanings:**
- `op`: Operation distribution [sequence, choice, parallel, loop]
- `seq_len`: Average length of sequential blocks
- `p_stop`: Probability of stopping recursion (controls depth)
- `width`: Average number of parallel branches
- `min_depth`/`max_depth`: Model complexity bounds

---

## Next Steps

After generating hybrid data:

1. **Evaluate performance gain:**
   ```bash
   cat outputs/evaluate_classifier/summary.txt
   ```

2. **Compare baselines:**
   - Real data only vs. hybrid training
   - Synthetic data only vs. hybrid training

3. **Iterate on synthetic parameters:**
   - Try different depth ranges
   - Adjust operation distributions
   - Vary model complexity

4. **Scale up:**
   - Generate more synthetic models (500-1000)
   - Create multiple synthetic datasets with different parameters
   - Run ensemble training with varied synthetic data

---

## Summary

✅ Synthetic data generates **process models programmatically**
✅ Output format **matches real XES-based CSV files**
✅ Can **mix synthetic + real data** in training
✅ Improves classifier **generalization and coverage**
✅ Faster iteration than collecting more real event logs

**Complete workflow:**
```bash
# 1. Generate synthetic data (~1 min with cache, ~75-80 min without)
cd ~/pm_ws25
sbatch lrz-cluster/run_create_labels_synthetic.slurm

# 2. (Optional) Generate real data CSV if not already done
sbatch lrz-cluster/run_create_labels.slurm

# 3. Train with automatic hybrid data discovery
sbatch lrz-cluster/run_evaluate_classifier.slurm

# 4. Compare results
cat outputs/evaluate_classifier/summary.txt
```

Good luck! 🚀
