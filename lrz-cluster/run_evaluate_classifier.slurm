#!/bin/bash
#SBATCH --job-name=eval_classifier
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --time=04:00:00                 # 4 hours for full evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32              # 32 CPUs for XGBoost training
#SBATCH --mem=128G                       # 128GB RAM for large datasets and model training
#SBATCH -o logs/eval_classifier_%j.out  # %j = Job ID
#SBATCH -e logs/eval_classifier_%j.err  # Error logs
# SBATCH --mail-type=END,FAIL           # Email notification (optional)
# SBATCH --mail-user=your@email.de      # Your email (optional, commented out)

# ====================================
# Setup
# ====================================

# Create directories
mkdir -p logs
mkdir -p outputs/evaluate_classifier
mkdir -p cache/models

# Load modules
module load slurm_setup
module load python/3.13.7

# Activate virtual environment
source ~/pm_ws25/.venv/bin/activate

# Set Python module path
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"

# Change to project directory
cd ${HOME}/pm_ws25

# Set number of threads for numpy/scipy (prevent oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ====================================
# Job Information
# ====================================

echo "======================================"
echo "ML Classifier Evaluation (End-to-End)"
echo "======================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "======================================"
echo ""

# ====================================
# Pre-flight Checks
# ====================================

echo "Checking for required data files..."
TRAIN_FILES=$(find data/runs -name "*.train.csv" 2>/dev/null | wc -l)
TEST_FILES=$(find data/runs -name "*.test.csv" 2>/dev/null | wc -l)
EVAL_FILES=$(find data/runs -name "*.eval.csv" 2>/dev/null | wc -l)

echo "Found:"
echo "  - Train files: $TRAIN_FILES"
echo "  - Test files:  $TEST_FILES"
echo "  - Eval files:  $EVAL_FILES"
echo ""

if [ "$TRAIN_FILES" -eq 0 ] || [ "$TEST_FILES" -eq 0 ]; then
    echo "ERROR: No training/test data found in data/runs/"
    echo "Please run create_labels job array first to generate labeled datasets."
    echo "Expected files: data/runs/**/*.{train,test,eval}.csv"
    exit 1
fi

echo "Data files check passed!"
echo ""

# ====================================
# Run Evaluation
# ====================================
#
# CACHING BEHAVIOR:
# - Automatically uses cached alignment runs from data/runs/
# - Only regenerates alignments if cache is missing
# - Train data: Loaded from pre-computed *.train.csv files
# - Test data: Loaded from cached RunDataset pickle files
#
# This makes evaluation MUCH faster (hours → minutes)
#
# ====================================

echo "Starting classifier evaluation..."
python scripts/evaluate_classifier_e2e.py

EXIT_CODE=$?

# ====================================
# Cleanup & Summary
# ====================================

echo ""
echo "======================================"
echo "Job Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: outputs/evaluate_classifier/"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
    echo "Check error log: logs/eval_classifier_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
