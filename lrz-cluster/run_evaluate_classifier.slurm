#!/bin/bash
#SBATCH --job-name=eval_classifier
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --time=04:00:00                 # 4 hours for full evaluation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # 16 CPUs for parallel alignment workers
#SBATCH --mem=64G                       # 64GB RAM for large datasets and model training
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
module load python/3.10.12-extended

# Activate virtual environment
source ~/venv/.venv/bin/activate

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
# Run Evaluation
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
