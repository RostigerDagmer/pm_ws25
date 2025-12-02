
### Usage
To generate the Slurm script and save it to a file:
Throttle is the main dial for controlling concurrency. Keep it small to be a responsible cluster citizen.

```bash
python -m scripts.generate_slurm --time 04:00:00 --mem 16G --throttle 4 > submit_jobs.sh
```
Then you can submit it with `sbatch submit_jobs.sh`.

### Arguments

```bash
--data-dir: Directory containing .xes files (default: data).
--config: Path to the configuration file (default: 
configs/default.yaml
).
--job-name: Slurm job name (default: create_labels).
--time: Time limit (default: 01:00:00).
--mem: Memory limit (default: 4G).
--cpus-per-task: CPUs per task (default: 1).
--ntasks: Number of tasks (default: 1).
--output: Output file path (default: logs/%j.out).
--error: Error file path (default: logs/%j.err).
```