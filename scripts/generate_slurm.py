import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Slurm script for processing .xes files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing .xes files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--job-name", type=str, default="create_labels", help="Slurm job name."
    )
    parser.add_argument(
        "--time", type=str, default="01:00:00", help="Time limit."
    )
    parser.add_argument("--mem", type=str, default="4G", help="Memory limit.")
    parser.add_argument(
        "--cpus-per-task", type=str, default="1", help="CPUs per task."
    )
    parser.add_argument(
        "--ntasks", type=str, default="1", help="Number of tasks."
    )
    parser.add_argument(
        "--output", type=str, default="logs/%j.out", help="Output file path."
    )
    parser.add_argument(
        "--error", type=str, default="logs/%j.err", help="Error file path."
    )
    parser.add_argument(
        "--throttle", type=int, default=4, help="Max concurrent jobs."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_path = args.config

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    xes_files = sorted(list(data_dir.rglob("*.xes")))
    num_files = len(xes_files)

    if not xes_files:
        print(f"No .xes files found in '{data_dir}'.")
        return

    # Create Manifest File
    manifest_path = Path(f"manifest_{args.job_name}.txt")
    with open(manifest_path, 'w') as f:
        for xes_file in xes_files:
            try:
                # Use relative path for robustness on different systems
                file_path = xes_file.relative_to(os.getcwd())
            except ValueError:
                file_path = xes_file
            f.write(f"{file_path}\n")

    # Slurm script header
    print("#!/bin/bash")
    print(f"#SBATCH --job-name={args.job_name}")
    print(f"#SBATCH --output={args.output}")
    print(f"#SBATCH --error={args.error}")
    print(f"#SBATCH --time={args.time}")
    print(f"#SBATCH --mem={args.mem}")
    print(f"#SBATCH --ntasks={args.ntasks}")
    print(f"#SBATCH --cpus-per-task={args.cpus_per_task}")
    print(f"#SBATCH --array=1-{num_files}%{args.throttle}")
    print("")
    print("mkdir -p logs")
    print("mkdir -p results")
    print("")

    print(f"FILE_MANIFEST=\"{manifest_path}\"")
    print("TASK_ID=$SLURM_ARRAY_TASK_ID")
    print("INPUT_FILE=$(tail -n +$TASK_ID $FILE_MANIFEST | head -n 1)")
    print("")

    print("FILE_NAME_BASE=$(basename $INPUT_FILE .xes)")
    # We don't strictly need OUTPUT_FILE variable if we don't use it, but user had it.
    # The user's snippet didn't use OUTPUT_FILE in the command, but it's good practice.
    # However, the user's snippet had:
    # OUTPUT_FILE=results/${FILE_NAME_BASE}_${TASK_ID}.result
    # mkdir -p results
    # And then ran the command.
    # The command `scripts.create_labels` likely handles output internally or via config.
    # I will include the variable definition as requested.
    print("OUTPUT_FILE=results/${FILE_NAME_BASE}_${TASK_ID}.result")
    print("")

    print(
        "echo \"Processing file $INPUT_FILE on core count $SLURM_CPUS_PER_TASK...\""
    )
    print("srun python -m scripts.create_labels \\")
    print(f"     --config=\"{config_path}\" \\")
    print("     --path=\"$INPUT_FILE\" \\")
    print("     --workers=$SLURM_CPUS_PER_TASK")


if __name__ == "__main__":
    main()
