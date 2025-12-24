#!/bin/bash
#SBATCH --partition=cm4_inter
#SBATCH --time=01:30:00  # 1.5 hours (sufficient with 96 CPUs)
#SBATCH --cpus-per-task=96  # 96 CPUs (safe limit for cm4_inter)
#SBATCH --mem=256G  # 256GB RAM (sufficient for 96 workers)
#SBATCH --output=logs/create_labels_synthetic_%j.out
#SBATCH --error=logs/create_labels_synthetic_%j.err
#SBATCH --job-name=synthetic_labels

echo "============================================"
echo "Synthetic Data Generation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo ""

# Load Python module
module load python/3.13.7

# Activate virtual environment
source ~/pm_ws25/.venv/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"

# Change to project directory
cd ~/pm_ws25

# Ensure logs directory exists
mkdir -p logs

# Run synthetic data generation
echo "Generating synthetic process models and alignment data..."
echo ""

python scripts/create_labels_synthetic.py \
    --config configs/default.yaml \
    --seed 42 \
    --n-models 100 \
    --n-traces 40 \
    --max-trace-length 50 \
    --min-depth 1 \
    --max-depth 3 \
    --runs 10 \
    --workers $SLURM_CPUS_PER_TASK \
    --train 0.7 \
    --test 0.2 \
    --eval 0.1 \
    --force-recompute  # Required for first run or when changing parameters

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Synthetic data generated"
    echo "Check cache/.runs_synthetic/ for output CSV files"
else
    echo "✗ FAILED: Check logs/create_labels_synthetic_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
