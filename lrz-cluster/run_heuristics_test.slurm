#!/bin/bash
#SBATCH --clusters=serial              # Wichtig: Cluster angeben!
#SBATCH --partition=serial_std         # Partition angeben
#SBATCH --array=0-99                   # 100 parallele Jobs
#SBATCH --time=02:00:00                # 2 Stunden pro Job
#SBATCH --ntasks=1                     # 1 Task pro Job
#SBATCH -o logs/job_%A_%a.out          # Output-Datei
#SBATCH -e logs/job_%A_%a.err          # Error-Datei (wichtig!)

# Logs-Verzeichnis erstellen (falls nicht vorhanden)
mkdir -p logs

# Module laden
module load slurm_setup
module load python/3.10.12   # -extended ist wichtig!

# Virtual Environment aktivieren
source ~/venv/.venv/bin/activate

# Skript ausführen mit Array Task ID
python sandbox_heuristics.py --run_id $SLURM_ARRAY_TASK_ID