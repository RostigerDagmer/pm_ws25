#!/bin/bash
#SBATCH --partition=cm4_inter
#SBATCH --time=04:00:00  # Should finish much faster with 64 CPUs
#SBATCH --cpus-per-task=64  # Using 64 CPUs for maximum speed
#SBATCH --mem=128G  # More memory for 64 workers
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
    --n-models 200 \
    --n-traces 50 \
    --max-trace-length 50 \
    --min-depth 1 \
    --max-depth 3 \
    --runs 10 \
    --workers $SLURM_CPUS_PER_TASK \
    --train 0.7 \
    --test 0.2 \
    --eval 0.1 \
    --force-recompute  # Generate alignment runs (required for first run)

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Synthetic data generated"
    echo "Check data/runs_synthetic/ for output CSV files"
else
    echo "✗ FAILED: Check logs/create_labels_synthetic_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
