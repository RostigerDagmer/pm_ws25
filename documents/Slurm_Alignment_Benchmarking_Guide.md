# Alignment Heuristics Benchmarking Guide

Complete documentation for the parallel alignment algorithm benchmarking system.

## Overview

This system is a **Process Mining benchmarking tool** designed to compare different alignment algorithms on SLURM cluster systems using:
- `lrz-cluster/run_heuristics_parallel.py` - Python experiment runner
- `lrz-cluster/run_heuristics_parallel.slurm` - SLURM batch script

**Main Purpose:** Evaluates the performance of 4 different alignment algorithms (used to match event logs with process models) across multiple datasets and traces, then trains ML models to predict which algorithm will be fastest for a given case.

---

## Python Script (run_heuristics_parallel.py)

### 1. Configuration (Lines 30-83)

**4 Alignment Variants:**
- `dijkstra` - Basic approach without heuristics
- `lp_heuristic` - A* with linear programming heuristic
- `ilp_heuristic` - A* with integer linear programming
- `incremental_astar` - Incremental A* search

**2 Datasets:** BPI Challenge 2013 and 2017

**10 trace indices** per dataset to test on

### 2. Experiment Grid (Lines 88-127)

Creates all combinations: 2 datasets × 10 traces × 4 variants = **80 total experiments**

### 3. Core Functionality

**`profile_alignment()`** (Lines 133-178)
- Runs alignment with profiling enabled
- Measures: total runtime, search time, LP/ILP solving time
- Returns detailed performance metrics

**`run_single_experiment()`** (Lines 181-308)
- Loads event log and extracts specific trace
- Discovers process model using Inductive Miner
- Runs alignment with one variant
- Extracts features for ML training (optional)
- Saves results as JSON

### 4. ML Model Training (Lines 315-481)

**`aggregate_results()`** aggregates individual experiment results and:
1. Creates summary CSV with all metrics
2. Identifies the **fastest aligner** for each trace
3. Trains 3 classifiers to predict best aligner:
   - Gradient Boosting
   - Random Forest
   - XGBoost

The trained models can then predict which alignment algorithm to use for new traces.

---

## SLURM Batch Script (run_heuristics_parallel.slurm)

The SLURM script automates the parallel execution of all 80 experiments on the LRZ cluster.

### SLURM Configuration

**Job Settings:**
- `--job-name=pm_heuristics` - Job identifier
- `--clusters=serial` - Use serial cluster
- `--partition=serial_std` - Standard serial partition
- `--array=0-79` - Create 80 parallel tasks (one per experiment)

**Resources per Task:**
- `--time=01:00:00` - 1 hour maximum runtime
- `--ntasks=1` - 1 CPU core per task
- `--mem=4G` - 4GB RAM per task

**Logging:**
- `logs/job_%A_%a.out` - Standard output (%A = array job ID, %a = task ID)
- `logs/job_%A_%a.err` - Error output

### Setup Steps

1. **Create directories** - `logs/` and `results/`
2. **Load modules:**
   - `slurm_setup`
   - `python/3.10.12-extended`
3. **Activate virtual environment** - `~/venv/.venv/bin/activate`
4. **Set Python path** - Adds `~/pm_ws25` to `PYTHONPATH` so features module is found
5. **Change directory** - Move to `~/pm_ws25`

### Execution Flow

Each of the 80 tasks:
1. Prints job information (task ID, node, resources, timestamp)
2. Runs: `python lrz-cluster/run_heuristics_parallel.py --run_id $SLURM_ARRAY_TASK_ID`
3. Saves individual experiment results to `results/exp_XXXX.json`

**Special behavior for last task (ID 79):**
- Waits 10 seconds for all other tasks to finish writing
- Automatically runs aggregation: `python ... --aggregate`
- Trains all ML models and creates summary CSV

### Usage Examples

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
python run_heuristics_parallel.py --aggregate
```

---

## Complete Workflow

1. **Submit SLURM array job** - `sbatch lrz-cluster/run_heuristics_parallel.slurm`
2. **80 tasks run in parallel** - Each saves results to `results/exp_XXXX.json`
3. **Task 79 automatically aggregates** - Creates `results/summary.csv`
4. **ML models trained** - Saves 3 trained models as `.pkl` files

## Output Files

- `exp_XXXX.json` - Individual experiment results with timing, states visited, costs
- `summary.csv` - Aggregated performance statistics
- `aligner_predictor_*.pkl` - Trained ML models (GB, RF, XGBoost)
- `best_aligner_labels.csv` - Training labels showing best aligner per trace

## Simple Explanation: The Testing Plan

Think of it like testing cars on race tracks:

**2 Datasets** = 2 different race tracks
- Track 1: BPI 2013 dataset
- Track 2: BPI 2017 dataset

**10 Traces** = 10 different routes on each track
- Each trace is like one specific path through the event log
- For example: trace #100, trace #200, trace #300, etc.

**4 Variants** = 4 different navigation algorithms
- dijkstra
- lp_heuristic
- ilp_heuristic
- incremental_astar

The script tests **every combination**:
- Take Track 1 → Test each of the 10 routes with each of the 4 algorithms
- Take Track 2 → Test each of the 10 routes with each of the 4 algorithms

**Total:** 2 tracks × 10 routes × 4 algorithms = **80 experiments**

**The goal:** Find out which algorithm is fastest for different situations!

## Design Benefits

This design allows **parallel execution** on HPC clusters while collecting comprehensive benchmarking data for algorithm selection optimization.
