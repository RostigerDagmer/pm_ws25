#!/bin/bash
#SBATCH --job-name=eval_gbrf
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --time=01:30:00                 # 1.5 hours (sufficient)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # 16 CPUs (half of QOS limit)
#SBATCH --mem=64G                        # 64GB RAM
#SBATCH -o logs/eval_gbrf_%j.out        # %j = Job ID
#SBATCH -e logs/eval_gbrf_%j.err        # Error logs
# SBATCH --mail-type=END,FAIL           # Email notification (optional)
# SBATCH --mail-user=your@email.de      # Your email (optional, commented out)

# ====================================
# Setup
# ====================================

# Create directories
mkdir -p logs
mkdir -p outputs/evaluate_gbrf_classifier
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
echo "Gradient Boosting & Random Forest Evaluation"
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
# Search in all 3 locations that the Python script uses
TRAIN_FILES=$(find cache/.runs data/runs cache/.runs_synthetic -name "*.train.csv" 2>/dev/null | wc -l)
TEST_FILES=$(find cache/.runs data/runs cache/.runs_synthetic -name "*.test.csv" 2>/dev/null | wc -l)
EVAL_FILES=$(find cache/.runs data/runs cache/.runs_synthetic -name "*.eval.csv" 2>/dev/null | wc -l)

# Count by location for detailed reporting
REAL_TRAIN=$(find cache/.runs -name "*.train.csv" 2>/dev/null | wc -l)
SYNTHETIC_TRAIN=$(find cache/.runs_synthetic -name "*.train.csv" 2>/dev/null | wc -l)

echo "Found (across cache/.runs, data/runs, cache/.runs_synthetic):"
echo "  - Train files: $TRAIN_FILES ($REAL_TRAIN real + $SYNTHETIC_TRAIN synthetic)"
echo "  - Test files:  $TEST_FILES"
echo "  - Eval files:  $EVAL_FILES"
echo ""

if [ "$TRAIN_FILES" -eq 0 ] || [ "$TEST_FILES" -eq 0 ]; then
    echo "ERROR: No training/test data found!"
    echo "Please run one or both:"
    echo "  - sbatch lrz-cluster/run_create_labels.slurm (for real data)"
    echo "  - sbatch lrz-cluster/run_create_labels_synthetic.slurm (for synthetic data)"
    exit 1
fi

echo "Data files check passed!"
echo ""

# ====================================
# Run Evaluation
# ====================================
#
# This script evaluates:
# - Gradient Boosting Classifier
# - Random Forest Classifier
#
# And compares them with baselines:
# - XGBoost
# - SingleBestSolver
# - RandomClassifier
#
# Results are saved to:
# - outputs/evaluate_gbrf_classifier/gradient_boosting/
# - outputs/evaluate_gbrf_classifier/random_forest/
# - outputs/evaluate_gbrf_classifier/combined_summary.csv
#
# ====================================

echo "Starting Gradient Boosting & Random Forest evaluation..."
python scripts/evaluate_gbrf_classifier.py

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
    echo ""
    echo "Results saved to:"
    echo "  - Gradient Boosting: outputs/evaluate_gbrf_classifier/gradient_boosting/"
    echo "  - Random Forest:     outputs/evaluate_gbrf_classifier/random_forest/"
    echo "  - Combined Summary:  outputs/evaluate_gbrf_classifier/combined_summary.csv"
    echo ""
    echo "View combined results:"
    echo "  cat outputs/evaluate_gbrf_classifier/combined_summary.csv"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
    echo "Check error log: logs/eval_gbrf_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
