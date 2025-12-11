#!/bin/bash
# Profile a single dataset to determine optimal Slurm resources
# Run this BEFORE submitting the full job array
#
# USAGE:
#   bash scripts/profile_job.sh [DATASET] [WORKERS] [RUNS]
#
# EXAMPLES:
#   # Profile with defaults (8 workers, 100 runs)
#   bash scripts/profile_job.sh
#
#   # Profile specific dataset with 4 workers
#   bash scripts/profile_job.sh data/d9769f3d-0ab0-4fb8-803b-0d1120ffcf54/Hospital_log.xes 4
#
#   # Profile with 8 workers and 50 runs (faster profiling)
#   bash scripts/profile_job.sh data/d9769f3d-0ab0-4fb8-803b-0d1120ffcf54/Hospital_log.xes 8 50
#
# OUTPUT:
#   - Real-time output to console
#   - Detailed report in results/profile_output.txt
#   - SLURM configuration recommendations
#

set -e

DATASET="${1:-data/a0addfda-2044-4541-a450-fdcc9fe16d17/BPIC15_1.xes}"
CONFIG="configs/default.yaml"
WORKERS="${2:-32}"
RUNS="${3:-100}"  # Number of alignment runs (default: 100)

echo "============================================"
echo "Job Profiling"
echo "============================================"
echo "Dataset: $DATASET"
echo "Workers: $WORKERS"
echo "Runs:    $RUNS"
echo "Started: $(date)"
echo ""

# Setup environment
module load python/3.13.7
source ~/pm_ws25/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"
cd ~/pm_ws25

# Ensure results directory exists
mkdir -p results

# Run with resource tracking
echo "Running label generation with tracking..."
echo "NOTE: Using --force-recompute to ensure fresh profiling data"
echo ""
/usr/bin/time -v python scripts/create_labels.py \
    --config "$CONFIG" \
    --path "$DATASET" \
    --seed 1 \
    --train 0.7 \
    --test 0.2 \
    --eval 0.1 \
    --runs $RUNS \
    --workers $WORKERS \
    --force-recompute \
    2>&1 | tee results/profile_output.txt

# Function to print and append to file
log_output() {
    echo "$1"
    echo "$1" >> results/profile_output.txt
}

log_output ""
log_output "============================================"
log_output "PROFILING RESULTS"
log_output "============================================"

# Extract metrics
MAX_RSS=$(grep "Maximum resident set size" results/profile_output.txt | awk '{print $6}')
ELAPSED=$(grep "Elapsed (wall clock) time" results/profile_output.txt | awk '{print $8}')
CPU_PCT=$(grep "Percent of CPU" results/profile_output.txt | awk '{print $7}' | tr -d '%')

# Convert memory to GB
MEM_GB=$((MAX_RSS / 1024 / 1024))

log_output ""
log_output "Resource Usage:"
log_output "  Memory:  ${MEM_GB}GB"
log_output "  Runtime: $ELAPSED"
log_output "  CPU:     ${CPU_PCT}%"
log_output ""

# Generate recommendations
log_output "============================================"
log_output "RECOMMENDATIONS FOR SLURM CONFIG"
log_output "============================================"
log_output ""

# Memory recommendation
if [ $MEM_GB -lt 20 ]; then
    MEM_REC="24G"
elif [ $MEM_GB -lt 30 ]; then
    MEM_REC="32G"
elif [ $MEM_GB -lt 45 ]; then
    MEM_REC="48G"
else
    MEM_REC="64G"
fi

# Time recommendation (convert HH:MM:SS to hours)
IFS=':' read -ra TIME_PARTS <<< "$ELAPSED"
if [ ${#TIME_PARTS[@]} -eq 3 ]; then
    HOURS=$((10#${TIME_PARTS[0]}))
    MINS=$((10#${TIME_PARTS[1]}))
    TOTAL_HOURS=$(echo "scale=1; $HOURS + $MINS/60" | bc)
else
    TOTAL_HOURS=1
fi

if (( $(echo "$TOTAL_HOURS < 1.5" | bc -l) )); then
    TIME_REC="02:00:00"
elif (( $(echo "$TOTAL_HOURS < 2.5" | bc -l) )); then
    TIME_REC="03:00:00"
elif (( $(echo "$TOTAL_HOURS < 3.5" | bc -l) )); then
    TIME_REC="04:00:00"
else
    TIME_REC="06:00:00"
fi

# CPU recommendation
CPU_USED=$(echo "scale=0; $CPU_PCT / 100" | bc)
if [ $CPU_USED -lt 5 ]; then
    CPU_REC="4"
elif [ $CPU_USED -lt 10 ]; then
    CPU_REC="8"
else
    CPU_REC="16"
fi

log_output "Edit lrz-cluster/run_create_labels.slurm:"
log_output ""
log_output "  #SBATCH --cpus-per-task=$CPU_REC"
log_output "  #SBATCH --mem=$MEM_REC"
log_output "  #SBATCH --time=$TIME_REC"
log_output ""

# Throttle recommendation based on cluster
log_output "Check cluster availability:"
log_output "  sinfo -o \"%P %a %T %c\""
log_output ""
log_output "Then adjust throttle in run_create_labels.slurm:"
log_output "  If idle CPUs > 128:  --array=0-15%16  (FAST)"
log_output "  If idle CPUs > 64:   --array=0-15%8   (BALANCED)"
log_output "  If idle CPUs > 32:   --array=0-15%4   (CONSERVATIVE)"
log_output ""

log_output "============================================"
log_output "Finished: $(date)"
log_output "============================================"
log_output ""
log_output "Full output saved to: results/profile_output.txt"
