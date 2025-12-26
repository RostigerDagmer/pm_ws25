#!/bin/bash
#
# Single Dataset: PermitLog (Sequential Processing)
#
# Process the failed PermitLog dataset with full resources
# Uses /tmp for cache to avoid disk quota limits
#
#SBATCH --job-name=permitlog
#SBATCH --partition=cm4_inter
#SBATCH --time=02:00:00                 # 2 hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224             # All 224 CPUs
#SBATCH --mem=480G                      # 480GB RAM
#SBATCH -o logs/permitlog_%j.out
#SBATCH -e logs/permitlog_%j.err

echo "============================================"
echo "PermitLog Dataset (Single Processing)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "============================================"
echo ""

# Setup
mkdir -p logs

module load slurm_setup
module load python/3.13.7

source ~/pm_ws25/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"
cd ${HOME}/pm_ws25

# Use /tmp for ALL cache directories to avoid home directory quota issues
TEMP_CACHE_BASE="/tmp/pm_cache_permitlog_${SLURM_JOB_ID}"
mkdir -p "$TEMP_CACHE_BASE"/{runs,unique_models,process_models}
mkdir -p cache

# Backup and symlink cache/.runs
if [ -d "cache/.runs" ] && [ ! -L "cache/.runs" ]; then
    mv cache/.runs "cache/.runs.backup_${SLURM_JOB_ID}"
    echo "✓ Backed up cache/.runs"
    # Copy existing cache to /tmp for reuse
    if [ "$(ls -A cache/.runs.backup_${SLURM_JOB_ID})" ]; then
        cp -r cache/.runs.backup_${SLURM_JOB_ID}/* "$TEMP_CACHE_BASE/runs/"
        echo "✓ Copied $(ls cache/.runs.backup_${SLURM_JOB_ID}/*.pkl 2>/dev/null | wc -l) cached datasets to /tmp"
    fi
fi
ln -sf "$TEMP_CACHE_BASE/runs" cache/.runs

# Backup and symlink cache/.cache_unique_models
if [ -d "cache/.cache_unique_models" ] && [ ! -L "cache/.cache_unique_models" ]; then
    mv cache/.cache_unique_models "cache/.cache_unique_models.backup_${SLURM_JOB_ID}"
    echo "✓ Backed up cache/.cache_unique_models"
    # Copy existing cache to /tmp for reuse
    if [ "$(ls -A cache/.cache_unique_models.backup_${SLURM_JOB_ID})" ]; then
        cp -r cache/.cache_unique_models.backup_${SLURM_JOB_ID}/* "$TEMP_CACHE_BASE/unique_models/"
        echo "✓ Copied cached unique models to /tmp"
    fi
fi
ln -sf "$TEMP_CACHE_BASE/unique_models" cache/.cache_unique_models

# Backup and symlink cache/.cache_process_models
if [ -d "cache/.cache_process_models" ] && [ ! -L "cache/.cache_process_models" ]; then
    mv cache/.cache_process_models "cache/.cache_process_models.backup_${SLURM_JOB_ID}"
    echo "✓ Backed up cache/.cache_process_models"
    # Copy existing cache to /tmp for reuse
    if [ "$(ls -A cache/.cache_process_models.backup_${SLURM_JOB_ID})" ]; then
        cp -r cache/.cache_process_models.backup_${SLURM_JOB_ID}/* "$TEMP_CACHE_BASE/process_models/"
        echo "✓ Copied cached process models to /tmp"
    fi
fi
ln -sf "$TEMP_CACHE_BASE/process_models" cache/.cache_process_models

echo "✓ All cache directories using /tmp:"
echo "  cache/.runs -> $TEMP_CACHE_BASE/runs"
echo "  cache/.cache_unique_models -> $TEMP_CACHE_BASE/unique_models"
echo "  cache/.cache_process_models -> $TEMP_CACHE_BASE/process_models"
echo "✓ /tmp space: $(df -h /tmp | tail -1 | awk '{print $4}') available"
echo ""

export OMP_NUM_THREADS=224  # Full resources
export MKL_NUM_THREADS=224
export NUMEXPR_NUM_THREADS=224

DATASET="data/db35afac-2133-40f3-a565-2dc77a9329a3/PermitLog.xes"

echo "Processing: $DATASET"
echo "Started: $(date)"
echo ""

python3 scripts/create_labels.py \
    --config configs/default.yaml \
    --path "$DATASET" \
    --seed 1 \
    --train 0.7 \
    --test 0.2 \
    --eval 0.1 \
    --runs 5 \
    --workers 224

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo ""

# Move cache from /tmp to final location
echo "============================================"
echo "Moving cache from /tmp to final location..."
echo "============================================"

# Move cache/.runs
if [ -L "cache/.runs" ]; then
    if [ -d "cache/.runs.backup_${SLURM_JOB_ID}" ]; then
        cp -v "$TEMP_CACHE_BASE/runs"/* "cache/.runs.backup_${SLURM_JOB_ID}/" 2>/dev/null || true
        rm cache/.runs
        mv "cache/.runs.backup_${SLURM_JOB_ID}" cache/.runs
        echo "✓ Merged runs cache"
    else
        rm cache/.runs
        mv "$TEMP_CACHE_BASE/runs" cache/.runs
        echo "✓ Moved runs cache"
    fi
fi

# Move cache/.cache_unique_models
if [ -L "cache/.cache_unique_models" ]; then
    if [ -d "cache/.cache_unique_models.backup_${SLURM_JOB_ID}" ]; then
        cp -v "$TEMP_CACHE_BASE/unique_models"/* "cache/.cache_unique_models.backup_${SLURM_JOB_ID}/" 2>/dev/null || true
        rm cache/.cache_unique_models
        mv "cache/.cache_unique_models.backup_${SLURM_JOB_ID}" cache/.cache_unique_models
        echo "✓ Merged unique_models cache"
    else
        rm cache/.cache_unique_models
        mv "$TEMP_CACHE_BASE/unique_models" cache/.cache_unique_models
        echo "✓ Moved unique_models cache"
    fi
fi

# Move cache/.cache_process_models
if [ -L "cache/.cache_process_models" ]; then
    if [ -d "cache/.cache_process_models.backup_${SLURM_JOB_ID}" ]; then
        cp -v "$TEMP_CACHE_BASE/process_models"/* "cache/.cache_process_models.backup_${SLURM_JOB_ID}/" 2>/dev/null || true
        rm cache/.cache_process_models
        mv "cache/.cache_process_models.backup_${SLURM_JOB_ID}" cache/.cache_process_models
        echo "✓ Merged process_models cache"
    else
        rm cache/.cache_process_models
        mv "$TEMP_CACHE_BASE/process_models" cache/.cache_process_models
        echo "✓ Moved process_models cache"
    fi
fi

# Clean up temp directory
rm -rf "$TEMP_CACHE_BASE"
FINAL_SIZE=$(du -sh cache/ 2>/dev/null | cut -f1)
echo "✓ All caches moved successfully (total: $FINAL_SIZE)"
echo "============================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ PERMITLOG PROCESSED SUCCESSFULLY"
    exit 0
else
    echo "✗ PERMITLOG FAILED"
    exit 1
fi
