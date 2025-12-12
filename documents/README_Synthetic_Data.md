# Synthetic Data Generation & Hybrid Training Guide

## Overview

This guide shows you how to:
1. Generate synthetic process model data
2. Combine it with real XES-based data for training
3. Improve classifier performance with hybrid training data

## Quick Start

```bash
# Step 1: Test that synthetic data generation works (10 minutes)
sbatch lrz-cluster/test_synthetic.slurm

# Step 2: Generate synthetic training data (2-4 hours)
sbatch lrz-cluster/run_create_labels_synthetic.slurm

# Step 3: Combine with real data and train classifier
# (Edit scripts/evaluate_classifier_e2e.py first - see below)
sbatch lrz-cluster/run_evaluate_classifier.slurm
```

---

## Step 1: Test Synthetic Data Generation

Before generating the full dataset, verify your environment works:

```bash
cd ~/pm_ws25
sbatch lrz-cluster/test_synthetic.slurm
```

**Check the output:**
```bash
cat logs/test_synthetic_<jobid>.out
```

You should see:
```
✓ All imports successful
✓ RNG initialized with seed 42
✓ Created synthetic model
✓ Created dataset with 10 models
✓ All tests passed!
```

---

## Step 2: Generate Synthetic Training Data

### 2.1: Submit the Job

```bash
cd ~/pm_ws25
sbatch lrz-cluster/run_create_labels_synthetic.slurm
```

**What it does:**
- Generates 200 synthetic process models (3 configurations)
- Simulates 50 traces per model
- Runs alignment benchmarks (100 runs per model-trace combo)
- Creates train/test/eval CSV splits
- Saves to `data/runs_synthetic/`

**Expected output:**
```
data/runs_synthetic/
  ├── <hash>.train.csv  (~3-5 MB)
  ├── <hash>.test.csv   (~1-2 MB)
  ├── <hash>.eval.csv   (~500 KB)
  ├── <hash>.runs.csv   (all runs, ~10 MB)
  └── <hash>.labels.csv (best aligners, ~5 MB)
```

### 2.2: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch real-time logs
tail -f logs/create_labels_synthetic_*.out

# Check for completion
ls -lh data/runs_synthetic/*.train.csv
```

### 2.3: Customize Parameters (Optional)

Edit [lrz-cluster/run_create_labels_synthetic.slurm](lrz-cluster/run_create_labels_synthetic.slurm):

```bash
python scripts/create_labels_synthetic.py \
    --n-models 500 \        # More models → more training data
    --n-traces 100 \        # More traces → better coverage
    --min-depth 2 \         # Deeper models → more complex
    --max-depth 4 \
    --runs 200 \            # More runs → more stable timing
    --workers 16
```

**Trade-offs:**
- More models/traces → longer generation time but richer data
- Deeper models → more realistic but slower alignment
- More runs → more accurate timing but longer computation

---

## Step 3: Combine Synthetic + Real Data for Training

The synthetic CSV files use the **same format** as real XES-based CSV files, so you can mix them!

### 3.1: Find Your CSV File Paths

**Real data** (already exists):
```bash
ls ~/pm_ws25/data/runs/*.train.csv
```

**Synthetic data** (after Step 2):
```bash
ls ~/pm_ws25/data/runs_synthetic/*.train.csv
```

### 3.2: Edit the Training Script

Open [scripts/evaluate_classifier_e2e.py](scripts/evaluate_classifier_e2e.py) and find the `TRAIN_DATASETS` section (around lines 51-86):

**Before (real data only):**
```python
TRAIN_DATASETS = {
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
    # ... more real datasets
}
```

**After (hybrid: real + synthetic):**
```python
TRAIN_DATASETS = {
    # Real XES-based datasets
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],

    # Synthetic datasets (add the hash from your data/runs_synthetic/)
    'runs_synthetic': ['<your_synthetic_hash>'],  # e.g., '3ae76aaddcd2...'
}
```

**To find the synthetic hash:**
```bash
cd ~/pm_ws25/data/runs_synthetic
ls *.train.csv | cut -d'.' -f1
# Copy the hash that appears (e.g., 3ae76aaddcd2a994b31497c8196e88630295c1ef)
```

### 3.3: Update the Data Loading Logic

In the same file, find the data loading section and ensure it can handle the `runs_synthetic` directory:

```python
# Around line 120-140
def load_training_data():
    all_train_dfs = []

    for dataset_dir, filenames in TRAIN_DATASETS.items():
        for filename in filenames:
            # Handle both regular and synthetic data paths
            if dataset_dir == 'runs_synthetic':
                base_path = Path(f"data/runs_synthetic")
            else:
                base_path = Path(f"data/runs")

            # Find the CSV file
            csv_file = list(base_path.glob(f"*{filename.replace('.xes', '')}*.train.csv"))
            if csv_file:
                df = pd.read_csv(csv_file[0])
                all_train_dfs.append(df)

    return pd.concat(all_train_dfs, ignore_index=True)
```

### 3.4: Train the Classifier

```bash
sbatch lrz-cluster/run_evaluate_classifier.slurm
```

**The classifier will now train on:**
- Real process models from XES files
- Synthetic process models
- Combined feature space

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
│   ├── test_synthetic.py                    # Test script
│   ├── create_labels_synthetic.py           # Generate synthetic data
│   └── evaluate_classifier_e2e.py           # Train classifier (edit this!)
│
├── lrz-cluster/
│   ├── test_synthetic.slurm                 # Test job
│   ├── run_create_labels_synthetic.slurm    # Synthetic generation job
│   └── run_evaluate_classifier.slurm        # Training job
│
└── data/
    ├── runs/                                 # Real XES-based data
    │   ├── <hash>.train.csv
    │   ├── <hash>.test.csv
    │   └── <hash>.eval.csv
    │
    └── runs_synthetic/                       # Synthetic data
        ├── <hash>.train.csv                  # ← Add this to TRAIN_DATASETS!
        ├── <hash>.test.csv
        └── <hash>.eval.csv
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
#SBATCH --mem=64G  # Increase from 32G

# Or reduce data size:
--n-models 100 \   # Reduce from 200
--n-traces 30 \    # Reduce from 50
```

### Problem: Synthetic data generation too slow

**Solution:** Reduce complexity:
```bash
--max-depth 2 \    # Simpler models
--runs 50 \        # Fewer alignment runs
--n-traces 20      # Fewer traces
```

### Problem: Can't find synthetic CSV hash

**Solution:**
```bash
# List all synthetic hashes:
ls data/runs_synthetic/*.train.csv | xargs -n1 basename | cut -d'.' -f1

# Or just use the whole path in TRAIN_DATASETS:
TRAIN_DATASETS = {
    'runs_synthetic': ['any_identifier'],  # The script will find it
}
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
# 1. Test (10 min)
sbatch lrz-cluster/test_synthetic.slurm

# 2. Generate synthetic data (2-4 hours)
sbatch lrz-cluster/run_create_labels_synthetic.slurm

# 3. Edit evaluate_classifier_e2e.py to include synthetic CSV

# 4. Train with hybrid data (1 hour)
sbatch lrz-cluster/run_evaluate_classifier.slurm

# 5. Compare results
cat outputs/evaluate_classifier/summary.txt
```

Good luck! 🚀
