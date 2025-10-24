import subprocess
import sys


def lint_package_with_score(package_path, config_path=None, threshold=8):
    """
    Lint a Python package at the given path using flake8, compute a
    'quality score' based on the number of lint errors, and exit
    with code 0 if score >= threshold, or 1 otherwise.

    Args:
        package_path (str): Path to the Python package or directory to lint.
        config_path (str, optional): Path to a .flake8 config file.
        threshold (int, optional): The score threshold. Default is 8.
    """
    # Build the flake8 command
    command = ["flake8", package_path]

    if config_path:
        command.extend(["--config", config_path])

    # Run flake8 and capture output
    process = subprocess.run(command, capture_output=True, text=True)

    # Print raw flake8 output
    if process.stdout:
        print("Flake8 STDOUT:\n", process.stdout)
    if process.stderr:
        print("Flake8 STDERR:\n", process.stderr, file=sys.stderr)

    # Count the number of errors or warnings (each line in stdout usually represents one finding)
    # You may need to adapt parsing logic if you have a different flake8 output format or plugins
    lint_output_lines = process.stdout.strip().split('\n')

    # If flake8 found absolutely nothing, the above split may return [''] instead of []
    # so let's filter empty lines carefully.
    lint_output_lines = [line for line in lint_output_lines if line.strip()]

    num_errors = len(lint_output_lines)

    # Define a simple "quality score" formula
    score = max(1000 - num_errors, 0)

    # Print the derived metrics
    print(f"Number of lint findings (errors/warnings): {num_errors}")
    print(f"Derived Quality Score: {score} (Threshold = {threshold})")

    # Decide exit status based on whether we meet/exceed our threshold
    if score >= threshold:
        print("Quality threshold met or exceeded. Exiting with code 0.")
        sys.exit(0)
    else:
        print("Quality threshold not reached. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    package_path = "../pm4py"
    config_file = "flake8.ini"
    quality_threshold = 950

    lint_package_with_score(package_path, config_path=config_file, threshold=quality_threshold)
