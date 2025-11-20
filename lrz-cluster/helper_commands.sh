#!/bin/bash
# ====================================
# Helper Commands für Process Mining Experiments
# ====================================

# ====================================
# 1. VORBEREITUNG
# ====================================

# Aktiviere Environment
setup_env() {
    module load python/3.10.12-base
    source ~/venv/.venv/bin/activate
    echo "✓ Environment aktiviert"
}

# Liste alle möglichen Experimente
list_experiments() {
    python sandbox_heuristics_parallel.py --list-experiments
}

# Test mit einem einzelnen Experiment
test_single() {
    local run_id=${1:-0}
    echo "Testing experiment $run_id..."
    python sandbox_heuristics_parallel.py --run_id $run_id --output-dir results_test
}

# ====================================
# 2. JOBS STARTEN
# ====================================

# Einzelner Test-Job (zur Überprüfung)
submit_test() {
    echo "Submitting test job (experiment 0)..."
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=pm_test
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err

mkdir -p logs results
module load slurm_setup
module load python/3.10.12-extended
source ~/venv/.venv/bin/activate

echo "Running test experiment 0..."
python sandbox_heuristics_parallel.py --run_id 0 --output-dir results

echo "Test completed at \$(date)"
EOF
}

# Alle Jobs starten
submit_all() {
    echo "Submitting all experiments..."
    sbatch run_heuristics_parallel.slurm
}

# Nur erste 10 Jobs (zum Testen)
submit_subset() {
    echo "Submitting first 10 experiments..."
    sbatch --array=0-9 run_heuristics_parallel.slurm
}

# Spezifische Job-Range
submit_range() {
    local start=$1
    local end=$2
    echo "Submitting experiments $start-$end..."
    sbatch --array=$start-$end run_heuristics_parallel.slurm
}

# ====================================
# 3. JOBS ÜBERWACHEN
# ====================================

# Status aller Jobs
check_jobs() {
    echo "=== Your SLURM Jobs ==="
    squeue -u $USER -o "%.10i %.9P %.30j %.8T %.10M %.6D %R"
}

# Detaillierter Status
check_detailed() {
    squeue -u $USER -o "%.10i %.9P %.12j %.8T %.10M %.10l %.6D %R"
}

# Jobs nach Status zählen
count_jobs() {
    echo "=== Job Status Summary ==="
    echo -n "Running:  "
    squeue -u $USER -t RUNNING | wc -l
    echo -n "Pending:  "
    squeue -u $USER -t PENDING | wc -l
    echo -n "Failed:   "
    sacct -u $USER --state=FAILED --starttime=today | wc -l
}

# Logs eines spezifischen Jobs anschauen
view_log() {
    local array_id=$1
    local task_id=$2
    
    if [ -z "$task_id" ]; then
        echo "Usage: view_log <array_id> <task_id>"
        echo "Example: view_log 123456 0"
        return 1
    fi
    
    echo "=== Output Log ==="
    cat logs/job_${array_id}_${task_id}.out
    echo ""
    echo "=== Error Log ==="
    cat logs/job_${array_id}_${task_id}.err
}

# Letzte Logs anschauen
view_latest_logs() {
    echo "=== Latest Output Logs ==="
    tail -n 20 logs/*.out | tail -n 50
}

# Fehler in allen Logs finden
check_errors() {
    echo "=== Checking for Errors ==="
    grep -r "Error\|Exception\|Traceback\|FAILED" logs/*.err 2>/dev/null | head -20
}

# ====================================
# 4. JOBS VERWALTEN
# ====================================

# Einzelnen Job abbrechen
cancel_job() {
    local job_id=$1
    if [ -z "$job_id" ]; then
        echo "Usage: cancel_job <job_id>"
        return 1
    fi
    scancel $job_id
    echo "Cancelled job $job_id"
}

# Alle deine Jobs abbrechen
cancel_all() {
    read -p "Cancel ALL your jobs? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        scancel -u $USER
        echo "All jobs cancelled"
    fi
}

# Nur wartende Jobs abbrechen
cancel_pending() {
    scancel -u $USER --state=PENDING
    echo "Cancelled all pending jobs"
}

# ====================================
# 5. ERGEBNISSE VERARBEITEN
# ====================================

# Ergebnisse aggregieren
aggregate_results() {
    echo "Aggregating results..."
    python sandbox_heuristics_parallel.py --aggregate --output-dir results
}

# Anzahl fertige Experimente
count_results() {
    echo "=== Results Summary ==="
    echo -n "Completed experiments: "
    ls results/exp_*.json 2>/dev/null | wc -l
    echo -n "Total expected: 80"
    echo ""
}

# Fehlende Experimente finden
find_missing() {
    echo "=== Missing Experiments ==="
    for i in {0..79}; do
        file=$(printf "results/exp_%04d.json" $i)
        if [ ! -f "$file" ]; then
            echo "Missing: experiment $i"
        fi
    done
}

# Ergebnisse als CSV exportieren
export_csv() {
    if [ -f "results/summary.csv" ]; then
        echo "Summary available at: results/summary.csv"
        echo ""
        echo "=== First 5 rows ==="
        head -5 results/summary.csv | column -t -s,
    else
        echo "No summary.csv found. Run: aggregate_results"
    fi
}

# Quick Statistics
quick_stats() {
    if [ -f "results/summary.csv" ]; then
        echo "=== Quick Statistics ==="
        python3 << EOF
import pandas as pd
df = pd.read_csv('results/summary.csv')
print(f"\nTotal experiments: {len(df)}")
print(f"\nBy Variant:")
print(df.groupby('variant')['total_runtime_ms'].agg(['mean', 'std', 'min', 'max']).round(2))
print(f"\nBy Dataset:")
print(df.groupby('dataset')['total_runtime_ms'].agg(['mean', 'count']).round(2))
EOF
    else
        echo "No summary.csv found. Run: aggregate_results"
    fi
}

# ====================================
# 6. AUFRÄUMEN
# ====================================

# Logs aufräumen
clean_logs() {
    read -p "Delete all log files? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f logs/*.out logs/*.err
        echo "Logs cleaned"
    fi
}

# Alte Ergebnisse archivieren
archive_results() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_dir="results_archive_${timestamp}"
    
    mkdir -p "$archive_dir"
    mv results/* "$archive_dir/" 2>/dev/null
    echo "Results archived to: $archive_dir"
}

# ====================================
# HELP
# ====================================

show_help() {
    cat << EOF
====================================
Process Mining Heuristics - Helper Commands
====================================

VORBEREITUNG:
  setup_env              - Environment aktivieren
  list_experiments       - Alle Experimente auflisten
  test_single [ID]       - Einzelnes Experiment testen (default: 0)

JOBS STARTEN:
  submit_test           - Test-Job starten (Exp. 0)
  submit_all            - Alle 80 Experimente starten
  submit_subset         - Erste 10 Experimente starten
  submit_range <s> <e>  - Experimente von <s> bis <e> starten

JOBS ÜBERWACHEN:
  check_jobs            - Job-Status anzeigen
  check_detailed        - Detaillierter Job-Status
  count_jobs            - Jobs nach Status zählen
  view_log <aid> <tid>  - Log anschauen (array_id task_id)
  view_latest_logs      - Letzte Logs anschauen
  check_errors          - Nach Fehlern suchen

JOBS VERWALTEN:
  cancel_job <id>       - Job abbrechen
  cancel_all            - Alle Jobs abbrechen
  cancel_pending        - Wartende Jobs abbrechen

ERGEBNISSE:
  aggregate_results     - Ergebnisse zusammenfassen
  count_results         - Fertige Experimente zählen
  find_missing          - Fehlende Experimente finden
  export_csv            - CSV anzeigen
  quick_stats           - Schnelle Statistiken

AUFRÄUMEN:
  clean_logs            - Logs löschen
  archive_results       - Ergebnisse archivieren

====================================
Beispiele:
  source helper_commands.sh
  setup_env
  list_experiments
  submit_test
  check_jobs
  aggregate_results
====================================
EOF
}

# Zeige Help wenn das Skript direkt aufgerufen wird
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    show_help
fi