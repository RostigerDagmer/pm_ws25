#!/bin/bash
#
# Parallel Label Generation using Bash Background Jobs
#
# Processes 21 datasets in 2 sequential batches within a SINGLE job
# Batch 1: 10 datasets, Batch 2: 11 datasets (22 workers each, 5 runs)
# Expected total runtime: ~2 hours
#
#SBATCH --job-name=create_labels_all
#SBATCH --partition=cm4_inter
#SBATCH --time=08:00:00                 # 8 hours (max for cm4_inter)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224             # 224 CPUs (all CPUs on 1 node)
#SBATCH --mem=480G                      # 480GB RAM (fits in ~488GB available)
#SBATCH -o logs/create_labels_all_%j.out
#SBATCH -e logs/create_labels_all_%j.err

echo "============================================"
echo "Parallel Label Generation (Bash Jobs)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 480GB"
echo "Started: $(date)"
echo "============================================"
echo ""

# Setup
mkdir -p logs data/runs

module load slurm_setup
module load python/3.13.7

source ~/pm_ws25/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${HOME}/pm_ws25"
cd ${HOME}/pm_ws25

export OMP_NUM_THREADS=22  # Each parallel job gets ~22 CPUs (224/10=22.4)
export MKL_NUM_THREADS=22
export NUMEXPR_NUM_THREADS=22

# Create dataset list file
cat > /tmp/datasets_${SLURM_JOB_ID}.txt <<'EOF'
data/d9769f3d-0ab0-4fb8-803b-0d1120ffcf54/Hospital_log.xes
data/63a8435a-077d-4ece-97cd-2c76d394d99c/BPIC15_2.xes
data/ed445cdd-27d5-4d77-a1f7-59fe7360cfbe/BPIC15_3.xes
data/679b11cf-47cd-459e-a6de-9ca614e25985/BPIC15_4.xes
data/3301445f-95e8-4ff0-98a4-901f1f204972/BPI%20Challenge%202018.xes
data/3926db30-f712-4394-aebc-75976070e91f/BPI_Challenge_2012.xes
data/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/RequestForPayment.xes
data/6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd/Hospital%20Billing%20-%20Event%20Log.xes
data/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/Sepsis%20Cases%20-%20Event%20Log.xes
data/d06aff4b-79f0-45e6-8ec8-e19730c248f1/BPI_Challenge_2019.xes
data/3537c19d-6c64-4b1d-815d-915ab0e479da/BPI_Challenge_2013_open_problems.xes
data/500573e6-accc-4b0c-9576-aa5468b10cee/BPI_Challenge_2013_incidents.xes
data/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/InternationalDeclarations.xes
data/fb84cf2d-166f-4de2-87be-62ee317077e5/PrepaidTravelCost.xes
data/12683249/Road_Traffic_Fine_Management_Process.xes
data/a0addfda-2044-4541-a450-fdcc9fe16d17/BPIC15_1.xes
data/5f3067df-f10b-45da-b98b-86ae4c7a310b/BPI%20Challenge%202017.xes
data/b32c6fe5-f212-4286-9774-58dd53511cf8/BPIC15_5.xes
data/c2c3b154-ab26-4b31-a0e8-8f2350ddac11/BPI_Challenge_2013_closed_problems.xes
data/6a0a26d2-82d0-4018-b1cd-89afb0e8627f/DomesticDeclarations.xes
data/db35afac-2133-40f3-a565-2dc77a9329a3/PermitLog.xes
EOF

echo "Running 21 datasets in 2 batches (22 workers/dataset, 5 runs each)..."
echo "Batch 1: 10 datasets | Batch 2: 11 datasets (estimated runtime: ~1 hour per batch, ~2 hours total)"
echo ""

# Process ALL datasets in parallel (batch size = total datasets)
DATASETS_FILE=/tmp/datasets_${SLURM_JOB_ID}.txt
SUCCESS_COUNT=0
FAIL_COUNT=0
BATCH_SIZE=10  # Run 10 datasets in parallel (224 CPUs / 22 per dataset ≈ 10)

# Read datasets into array
mapfile -t DATASETS < "$DATASETS_FILE"
TOTAL=${#DATASETS[@]}

echo "Processing $TOTAL datasets in batches of $BATCH_SIZE..."
echo ""

# Process in batches
for ((i=0; i<${#DATASETS[@]}; i+=BATCH_SIZE)); do
    BATCH_NUM=$((i/BATCH_SIZE + 1))
    BATCH_END=$((i+BATCH_SIZE > TOTAL ? TOTAL : i+BATCH_SIZE))

    echo "========================================"
    echo "Batch $BATCH_NUM (datasets $((i+1))-$BATCH_END)"
    echo "Started: $(date)"
    echo "========================================"

    # Start up to BATCH_SIZE jobs in background
    declare -a PIDS=()
    for ((j=0; j<BATCH_SIZE && i+j<TOTAL; j++)); do
        DATASET="${DATASETS[$((i+j))]}"
        DATASET_NUM=$((i+j+1))

        echo "[$DATASET_NUM/$TOTAL] Starting: $DATASET"

        (
            python scripts/create_labels.py \
                --config configs/default.yaml \
                --path "$DATASET" \
                --seed 1 \
                --train 0.7 \
                --test 0.2 \
                --eval 0.1 \
                --runs 5 \
                --workers 22 \
                > logs/dataset_${SLURM_JOB_ID}_${DATASET_NUM}.out 2>&1
            # --force-recompute  # Uncomment this line (and add \ to line above) to regenerate all data

            echo $? > /tmp/exitcode_${SLURM_JOB_ID}_${DATASET_NUM}
        ) &
        PIDS+=($!)
    done

    # Wait for all background jobs in this batch to complete
    for PID in "${PIDS[@]}"; do
        wait $PID
    done

    echo ""
    echo "Batch $BATCH_NUM completed: $(date)"
    echo ""
done

echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"

# Count successes and failures
for ((k=1; k<=TOTAL; k++)); do
    if [ -f /tmp/exitcode_${SLURM_JOB_ID}_${k} ]; then
        CODE=$(cat /tmp/exitcode_${SLURM_JOB_ID}_${k})
        if [ "$CODE" -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        rm -f /tmp/exitcode_${SLURM_JOB_ID}_${k}
    fi
done

echo "Total datasets:    $TOTAL"
echo "Succeeded:         $SUCCESS_COUNT"
echo "Failed:            $FAIL_COUNT"
echo "Finished:          $(date)"
echo "Exit Code:         $FAIL_COUNT"
echo "============================================"

# Cleanup
rm -f /tmp/datasets_${SLURM_JOB_ID}.txt

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓ ALL DATASETS PROCESSED SUCCESSFULLY"
    exit 0
else
    echo "⚠ $FAIL_COUNT datasets failed - check logs/dataset_${SLURM_JOB_ID}_*.out"
    exit 1
fi
