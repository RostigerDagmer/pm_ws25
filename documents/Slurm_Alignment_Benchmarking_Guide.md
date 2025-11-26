# Alignment Heuristics Benchmarking Guide

> **Branch:** `train_evaluate_ML`
> **Files:**
> - `pm_ws25/lrz-cluster/run_heuristics_parallel.py`
> - `pm_ws25/lrz-cluster/run_heuristics_parallel.slurm`
>
> **Output Directories:**
> - `pm_ws25/results/` - Experiment data and ML models
> - `pm_ws25/logs/` - SLURM job execution logs

---

## Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Usage Guide](#usage-guide)
4. [Complete Workflow](#complete-workflow)
5. [Output Files](#output-files)
6. [Known Issues & Workarounds](#known-issues--workarounds)

---

## Overview

This system is a **Process Mining benchmarking tool** designed to compare different alignment algorithms on SLURM cluster systems.

**Main Purpose:** Evaluates the performance of 4 different alignment algorithms (used to match event logs with process models) across multiple datasets and traces, then trains ML models to predict which algorithm will be fastest for a given case.

### Experiment Design

**4 Alignment Variants:**
- `dijkstra` - Basic approach without heuristics
- `lp_heuristic` - A* with linear programming heuristic
- `ilp_heuristic` - A* with integer linear programming
- `incremental_astar` - Incremental A* search

**2 Datasets:**
- BPI Challenge 2013
- BPI Challenge 2017

**10 trace indices** per dataset to test on

**Total:** 2 datasets × 10 traces × 4 variants = **80 experiments**

### Simple Explanation

Think of it like testing cars on race tracks:

- **2 Datasets** = 2 different race tracks (BPI 2013, BPI 2017)
- **10 Traces** = 10 different routes on each track
- **4 Variants** = 4 different navigation algorithms

The script tests **every combination** to find out which algorithm is fastest for different situations!

---

## System Components

### 1. Python Script (run_heuristics_parallel.py)

**Configuration (Lines 39-77)**
- Defines alignment variants, datasets, and experiment grid
- Maps experiment IDs to specific configurations

**Core Functions:**

- **`profile_alignment()`** (Line 123)
  - Runs alignment with profiling enabled
  - Measures: total runtime, search time, LP/ILP solving time
  - Returns detailed performance metrics

- **`run_single_experiment()`** (Line 169)
  - Loads event log and extracts specific trace
  - Discovers process model using Inductive Miner
  - Runs alignment with one variant
  - Extracts features for ML training (optional)
  - Saves results as JSON

- **`aggregate_results()`** (Line 324)
  - Aggregates individual experiment results
  - Creates summary CSV with all metrics
  - Identifies the fastest aligner for each trace
  - Trains 3 classifiers to predict best aligner:
    - Gradient Boosting
    - Random Forest
    - XGBoost

### 2. SLURM Batch Script (run_heuristics_parallel.slurm)

Automates the parallel execution of all 80 experiments on the LRZ cluster.

**SLURM Configuration:**

Job Settings:
- `--job-name=pm_heuristics` - Job identifier
- `--clusters=serial` - Use serial cluster
- `--partition=serial_std` - Standard serial partition
- `--array=0-79` - Create 80 parallel tasks (one per experiment)

Resources per Task:
- `--time=01:00:00` - 1 hour maximum runtime
- `--ntasks=1` - 1 CPU core per task
- `--mem=4G` - 4GB RAM per task

Logging:
- `logs/job_%A_%a.out` - Standard output (%A = array job ID, %a = task ID)
- `logs/job_%A_%a.err` - Error output

**Setup Steps:**
1. Create directories (`logs/` and `results/`)
2. Load modules (`slurm_setup`, `python/3.10.12-extended`)
3. Activate virtual environment (`~/venv/.venv/bin/activate`)
4. Set Python path (adds `~/pm_ws25` to `PYTHONPATH`)
5. Change directory to `~/pm_ws25`

**Execution Flow:**

Each of the 80 tasks:
1. Prints job information (task ID, node, resources, timestamp)
2. Runs: `python lrz-cluster/run_heuristics_parallel.py --run_id $SLURM_ARRAY_TASK_ID`
3. Saves individual experiment results to `results/exp_XXXX.json`

**Special behavior for last task (ID 79):**
- Waits 10 seconds for all other tasks to finish writing
- Automatically runs aggregation: `python ... --aggregate`
- Trains all ML models and creates summary CSV

---

## Usage Guide

### SLURM Cluster Usage

```bash
# Submit the SLURM job array (starts all 80 experiments)
sbatch lrz-cluster/run_heuristics_parallel.slurm

# Check job status
squeue -u $USER

# Cancel all jobs
scancel <job_array_id>

# View logs for specific task
cat logs/job_<array_id>_<task_id>.out
cat logs/job_<array_id>_<task_id>.err
```

### Manual Python Usage (without SLURM)

```bash
# List all 80 experiment configurations
python run_heuristics_parallel.py --list-experiments

# Run experiment #0 manually
python run_heuristics_parallel.py --run_id 0

# After all jobs finish, aggregate results and train ML models
python run_heuristics_parallel.py --aggregate --output-dir results
```

### Manual Aggregation (if needed)

If some experiments fail and automatic aggregation doesn't run, you can manually aggregate:

```bash
# Navigate to project directory
cd ~/pm_ws25

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"

# Activate virtual environment
source .venv/bin/activate

# Run aggregation on available experiments
python lrz-cluster/run_heuristics_parallel.py --aggregate --output-dir results
```

This command was used to generate `summary.csv` and `best_aligner_labels.csv` after the `incremental_astar` failures.

---

## Complete Workflow

1. **Submit SLURM array job** - `sbatch lrz-cluster/run_heuristics_parallel.slurm`
2. **80 tasks run in parallel** - Each saves results to `results/exp_XXXX.json`
3. **Task 79 automatically aggregates** - Creates `results/summary.csv` and trains ML models
4. **ML models saved** - 3 trained models as `.pkl` files (if training succeeds)

---

## Output Files

### Experiment Data (`pm_ws25/results/`)

**Individual Experiment Results:**
- `exp_0000.json` to `exp_0079.json` - Individual experiment results (80 files expected)
  - Contains: timing metrics, visited states, costs, feature vectors

**Aggregated Results:**
- `summary.csv` - Aggregated performance statistics across all successful experiments
  - Contains: all metrics from individual experiments in tabular format

- `best_aligner_labels.csv` - Training labels showing best aligner per trace
  - Used to identify which variant was fastest for each trace

**ML Models (if training succeeds):**
- `aligner_predictor_gb.pkl` - Gradient Boosting classifier
- `aligner_predictor_rf.pkl` - Random Forest classifier
- `aligner_predictor_xgb.pkl` - XGBoost classifier

**Note:** ML model training requires at least 2 different variants to be optimal across different traces. If one variant is always fastest, models cannot be trained.

### SLURM Job Logs (`pm_ws25/logs/`)

- `job_<array_id>_0.out` to `job_<array_id>_79.out` - Standard output for each task (80 files)
- `job_<array_id>_0.err` to `job_<array_id>_79.err` - Error output for each task (80 files)

---

## Known Issues & Workarounds

### Incremental A* Failures

**Problem:** 11 out of 80 experiments failed with `incremental_astar` variant returning `None`.

**Failed Experiments:**
- exp_0011, exp_0043, exp_0047, exp_0051, exp_0055, exp_0059, exp_0063, exp_0067, exp_0071, exp_0075, exp_0079

**Pattern:** All failed experiments have `experiment_id % 4 = 3`, meaning all use the `incremental_astar` variant.

**Root Cause Analysis:**

The `incremental_astar` implementation in pm4py has multiple issues:

1. **Returns None** when Extended Marking Equation (EME) initial solve is infeasible (line 746 in `incremental_a_star.py`)
2. **Search exhausts without solution** (line 866 in `incremental_a_star.py`)
3. **Incorrect costs** observed in `sandbox-heuristics.ipynb`:
   - BPI 2017 trace 761: returned cost 80073 vs correct 70074
4. **Extremely slow** compared to other variants:
   - 11.1 seconds vs 0.4-0.7 seconds for entire BPI 2013 log
   - 1772 seconds (29.5 minutes) for single BPI 2017 trace

**Evidence from notebook (`sandbox-heuristics.ipynb`):**
- BPI 2013 trace 823: ✓ Succeeded but slow (123ms)
- BPI 2013 full log: ✓ All 1487 traces succeeded but 27× slower
- BPI 2017 trace 761: ✓ Succeeded but returned **wrong cost** and 2387× slower

**Impact:**
- 69 out of 80 experiments succeeded (86% success rate)
- Only affects `incremental_astar` variant
- Other 3 variants (`dijkstra`, `lp_heuristic`, `ilp_heuristic`) work correctly

**Workaround:**

Due to this error, manual aggregation was performed on the 69 successful experiments using:

```bash
cd ~/pm_ws25
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"
source .venv/bin/activate
python lrz-cluster/run_heuristics_parallel.py --aggregate --output-dir results
```

This successfully generated:
- ✅ `results/summary.csv` - Statistics for 69 successful experiments
- ✅ `results/best_aligner_labels.csv` - Best aligner per trace

The other 3 variants are reliable and sufficient for algorithm comparison.

**Note on ML Training:** ML model training failed because all 20 tested traces showed `dijkstra` as the fastest variant. Machine learning requires at least 2 different classes, but this result is valuable - it shows that for these particular traces, **dijkstra consistently performs best**.

**Recommendation:**
Consider excluding `incremental_astar` from future experiments until pm4py fixes are available.

---

## Design Benefits

This design allows **parallel execution** on HPC clusters while collecting comprehensive benchmarking data for algorithm selection optimization.
