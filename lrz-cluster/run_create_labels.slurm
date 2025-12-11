#!/bin/bash
#
# Parallel Label Generation using Slurm Job Arrays
#
# HOW IT WORKS:
# - Creates 16 independent jobs (one per dataset)
# - Jobs run in parallel based on throttle setting
# - Each job gets dedicated CPUs and RAM
#
# RESOURCE PROFILES (choose one below):
# ┌──────────────┬────────┬──────┬────────┬──────────────┬──────────────┐
# │ Profile      │ Jobs   │ CPUs │ Memory │ Total CPUs   │ Est. Time    │
# ├──────────────┼────────┼──────┼────────┼──────────────┼──────────────┤
# │ fast         │ 16     │ 8    │ 32GB   │ 128 CPUs     │ ~2 hours     │
# │ balanced     │ 8      │ 8    │ 32GB   │ 64 CPUs      │ ~4 hours     │
# │ conservative │ 4      │ 8    │ 32GB   │ 32 CPUs      │ ~8 hours     │
# └──────────────┴────────┴──────┴────────┴──────────────┴──────────────┘

#SBATCH --job-name=create_labels
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --array=0-15%16                  # 16 datasets, 8 concurrent (BALANCED PROFILE)
#SBATCH --time=02:00:00                 # 4 hours per dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # 8 CPUs per job
#SBATCH --mem=24G                       # 32GB per job
#SBATCH -o logs/create_labels_%A_%a.out
#SBATCH -e logs/create_labels_%A_%a.err
# SBATCH --mail-type=END,FAIL
# SBATCH --mail-user=your@email.de

# ====================================
# TO CHANGE SPEED: Modify line above marked "BALANCED PROFILE"
# ====================================
# FAST:         --array=0-15%16  (all datasets run simultaneously)
# BALANCED:     --array=0-15%8   (8 datasets run simultaneously) ← CURRENT
# CONSERVATIVE: --array=0-15%4   (4 datasets run simultaneously)
#
# Total concurrent CPUs = throttle × 8
# Total concurrent RAM  = throttle × 32GB

# ====================================
# Setup
# ====================================

mkdir -p logs data/runs

module load slurm_setup
module load python/3.10.12-extended

source ~/pm_ws25/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"
cd ${HOME}/pm_ws25

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ====================================
# Dataset Manifest
# ====================================

DATASETS=(
    "data/d9769f3d-0ab0-4fb8-803b-0d1120ffcf54/Hospital_log.xes"
    "data/63a8435a-077d-4ece-97cd-2c76d394d99c/BPIC15_2.xes"
    "data/ed445cdd-27d5-4d77-a1f7-59fe7360cfbe/BPIC15_3.xes"
    "data/679b11cf-47cd-459e-a6de-9ca614e25985/BPIC15_4.xes"
    "data/3301445f-95e8-4ff0-98a4-901f1f204972/BPI%20Challenge%202018.xes"
    "data/3926db30-f712-4394-aebc-75976070e91f/BPI_Challenge_2012.xes"
    "data/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/RequestForPayment.xes"
    "data/6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd/Hospital%20Billing%20-%20Event%20Log.xes"
    "data/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/Sepsis%20Cases%20-%20Event%20Log.xes"
    "data/d06aff4b-79f0-45e6-8ec8-e19730c248f1/BPI_Challenge_2019.xes"
    "data/3537c19d-6c64-4b1d-815d-915ab0e479da/BPI_Challenge_2013_open_problems.xes"
    "data/500573e6-accc-4b0c-9576-aa5468b10cee/BPI_Challenge_2013_incidents.xes"
    "data/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/InternationalDeclarations.xes"
    "data/fb84cf2d-166f-4de2-87be-62ee317077e5/PrepaidTravelCost.xes"
    "data/12683249/Road_Traffic_Fine_Management_Process.xes"
    "data/a0addfda-2044-4541-a450-fdcc9fe16d17/BPIC15_1.xes"
)

# FOR FUTURE DATASETS: Just add lines to this array!
# Example:
# DATASETS+=(
#     "data/your-new-uuid/your-new-dataset.xes"
# )

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# ====================================
# Job Info
# ====================================

echo "======================================"
echo "Parallel Label Generation"
echo "======================================"
echo "Array Job ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID / 15"
echo "Dataset:       $DATASET"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo ""
echo "Resources (THIS job):"
echo "  CPUs:        $SLURM_CPUS_PER_TASK"
echo "  Memory:      32GB"
echo "======================================"
echo ""

# ====================================
# Run Label Generation
# ====================================
#
# CACHING BEHAVIOR:
# - By default, uses cached alignment runs if available (FAST)
# - Skips feature extraction if output .csv files already exist
# - To regenerate everything, add: --force-recompute
#
# Example for force recompute:
#   python scripts/create_labels.py ... --force-recompute
#
# ====================================

python scripts/create_labels.py \
    --config configs/default.yaml \
    --path "$DATASET" \
    --seed 1 \
    --train 0.7 \
    --test 0.2 \
    --eval 0.1 \
    --runs 100 \
    --workers $SLURM_CPUS_PER_TASK \
    # --force-recompute  # Uncomment to regenerate all data (bypasses cache)

EXIT_CODE=$?

# ====================================
# Summary
# ====================================

echo ""
echo "======================================"
echo "Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Success: $DATASET"
else
    echo "✗ Failed: $DATASET"
    echo "Check: logs/create_labels_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
fi

exit $EXIT_CODE
