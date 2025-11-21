#!/bin/bash
#SBATCH --job-name=pm_heuristics
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --array=0-79                    # 80 Experimente (2 Datasets x 10 Traces x 4 Variants)
#SBATCH --time=01:00:00                 # 1 Stunde pro Job
#SBATCH --ntasks=1                      # 1 CPU pro Job
#SBATCH --mem=4G                        # 4GB RAM
#SBATCH -o logs/job_%A_%a.out           # %A = Array-ID, %a = Task-ID
#SBATCH -e logs/job_%A_%a.err           # Error logs
# SBATCH --mail-type=END,FAIL            # Email bei Ende/Fehler (optional)
# SBATCH --mail-user=deine@email.de    # Deine Email (optional, auskommentiert)

# ====================================
# Setup
# ====================================

# Verzeichnisse erstellen
mkdir -p logs
mkdir -p results

# Module laden
module load slurm_setup
module load python/3.10.12-extended

# Virtual Environment aktivieren
source ~/venv/.venv/bin/activate

# ====================================
# Job Information
# ====================================

echo "======================================"
echo "Process Mining Heuristics Comparison"
echo "======================================"
echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "======================================"

# ====================================
# Run Experiment
# ====================================

python run_heuristics_parallel.py --run_id $SLURM_ARRAY_TASK_ID --output-dir results

EXIT_CODE=$?

# ====================================
# Cleanup & Summary
# ====================================

echo ""
echo "======================================"
echo "Job Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "======================================"

# Wenn der letzte Job (ID 79), aggregiere Ergebnisse und trainiere ML Modelle
if [ $SLURM_ARRAY_TASK_ID -eq 79 ]; then
    echo ""
    echo "Last job completed - aggregating results and training ML models..."
    sleep 10  # Warte kurz, damit alle Jobs ihre Dateien schreiben können
    python run_heuristics_parallel.py --aggregate --output-dir results
fi

exit $EXIT_CODE