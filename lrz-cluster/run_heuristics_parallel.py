#!/usr/bin/env python3
"""
Process Mining Heuristics Comparison Script - Parallel Version
Optimized for SLURM array jobs with configurable experiments.
Each job processes a specific trace-model combination.
"""

import pm4py
import os
import sys
import argparse
import cProfile
import pstats
import io
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path

from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware
from pm4py.objects.petri_net.utils import align_utils
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm
from pm4py.algo.conformance.alignments.petri_net.algorithm import Variants as AlignmentsVariants
from pm4py.discovery import discover_petri_net_inductive as inductive_miner


# -------------------------------
# CONFIGURATION
# -------------------------------

# Alignment algorithm variants
ALIGNMENT_VARIANTS = {
    "dijkstra": AlignmentsVariants.VERSION_DIJKSTRA_NO_HEURISTICS,
    "lp_heuristic": AlignmentsVariants.VERSION_STATE_EQUATION_A_STAR,
    "ilp_heuristic": AlignmentsVariants.VERSION_STATE_EQUATION_A_STAR_ILP,
    "incremental_astar": AlignmentsVariants.VERSION_INCREMENTAL_A_STAR,
}

# Dataset configurations
DATASETS = {
    "bpi2013": {
        "path": "data/c2c3b154-ab26-4b31-a0e8-8f2350ddac11/BPI_Challenge_2013_closed_problems.xes",
        "noise_threshold": 0.23,
        "trace_indices": [529, 823, 100, 200, 300, 400, 500, 600, 700, 800]  # Example traces
    },
    "bpi2017": {
        "path": "data/5f3067df-f10b-45da-b98b-86ae4c7a310b/BPI%20Challenge%202017.xes",
        "noise_threshold": 0.21,
        "trace_indices": [761, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    }
}


# -------------------------------
# EXPERIMENT CONFIGURATION
# -------------------------------

def generate_experiment_grid():
    """
    Generate all experiment combinations.
    Returns list of dicts with experiment parameters.
    """
    experiments = []
    exp_id = 0
    
    for dataset_name, dataset_config in DATASETS.items():
        for trace_idx in dataset_config["trace_indices"]:
            for variant_name in ALIGNMENT_VARIANTS.keys():
                experiments.append({
                    "exp_id": exp_id,
                    "dataset": dataset_name,
                    "trace_idx": trace_idx,
                    "variant": variant_name,
                    "noise_threshold": dataset_config["noise_threshold"]
                })
                exp_id += 1
    
    return experiments


def get_experiment_by_id(run_id):
    """Get experiment configuration for specific run_id."""
    experiments = generate_experiment_grid()
    if run_id >= len(experiments):
        raise ValueError(f"run_id {run_id} out of range (max: {len(experiments)-1})")
    return experiments[run_id]


# -------------------------------
# PROFILING & ALIGNMENT
# -------------------------------

def profile_alignment(trace, pm_net, pm_im, pm_fm, variant):
    """Profile alignment call and return detailed metrics."""
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.perf_counter()

    res = alignments_algorithm.apply_trace(
        trace, pm_net, pm_im, pm_fm, 
        parameters=None, 
        variant=variant
    )

    end_time = time.perf_counter()
    pr.disable()

    # Collect profiling stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(20)
    profile_summary = s.getvalue()

    # Extract timing information
    search_time = 0.0
    lp_time = 0.0
    for func, (cc, nc, tt, ct, callers) in ps.stats.items():
        fname = func[2]
        if "__search" in fname:
            search_time += ct
        elif "cvxopt.glpk.lp" in fname or "cvxopt.glpk.ilp" in fname:
            lp_time += ct

    total_runtime = end_time - start_time
    search_time_perc = (search_time / total_runtime) * 100 if total_runtime > 0 else 0
    lp_time_perc = (lp_time / search_time) * 100 if search_time > 0 else 0
    
    return res, total_runtime, search_time, search_time_perc, lp_time, lp_time_perc, profile_summary


def run_single_experiment(exp_config, output_dir="results"):
    """
    Run a single alignment experiment.
    
    Args:
        exp_config: Dict with experiment parameters
        output_dir: Directory to save results
        
    Returns:
        Dict with experiment results
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_config['exp_id']}")
    print(f"Dataset: {exp_config['dataset']}")
    print(f"Trace Index: {exp_config['trace_idx']}")
    print(f"Variant: {exp_config['variant']}")
    print("="*80)
    
    # Load dataset
    dataset_config = DATASETS[exp_config['dataset']]
    log_path = dataset_config['path']
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Dataset not found: {log_path}")
    
    print(f"\nLoading event log: {log_path}")
    event_log = xes_importer.apply(log_path)
    print(f"Total traces in log: {len(event_log)}")
    
    # Get specific trace
    trace_idx = exp_config['trace_idx']
    if trace_idx >= len(event_log):
        raise ValueError(f"Trace index {trace_idx} out of range (max: {len(event_log)-1})")
    
    trace = event_log[trace_idx]
    activity_key = pm4py.util.xes_constants.DEFAULT_NAME_KEY
    trace_activities = [event[activity_key] for event in trace]
    
    print(f"\nTrace {trace_idx} (Case ID: {trace.attributes.get('concept:name', 'N/A')}):")
    print(f"Activities: {trace_activities}")
    print(f"Length: {len(trace_activities)}")
    
    # Discover process model
    print("\nDiscovering process model...")
    model_net, model_im, model_fm = inductive_miner(
        event_log, 
        disable_fallthroughs=False, 
        noise_threshold=exp_config['noise_threshold']
    )
    print(f"Model: {len(model_net.places)} places, {len(model_net.transitions)} transitions")
    
    # Run alignment with specific variant
    variant = ALIGNMENT_VARIANTS[exp_config['variant']]
    print(f"\nRunning alignment with {exp_config['variant']}...")
    
    res, total_runtime, search_time, search_time_perc, lp_time, lp_time_perc, profile_summary = \
        profile_alignment(trace, model_net, model_im, model_fm, variant)
    
    # Extract results
    if not res:
        print("ERROR: Alignment returned None")
        return None
    
    result = {
        # Experiment metadata
        "exp_id": exp_config['exp_id'],
        "dataset": exp_config['dataset'],
        "trace_idx": trace_idx,
        "trace_length": len(trace_activities),
        "case_id": trace.attributes.get('concept:name', 'N/A'),
        "variant": exp_config['variant'],
        "noise_threshold": exp_config['noise_threshold'],
        
        # Model info
        "model_places": len(model_net.places),
        "model_transitions": len(model_net.transitions),
        
        # Alignment results
        "cost": res.get("cost", np.nan),
        "visited_states": res.get("visited_states", np.nan),
        "lp_solved": res.get("lp_solved", np.nan),
        "alignment_length": len(res.get("alignment", [])),
        
        # Performance metrics
        "total_runtime_ms": total_runtime * 1000,
        "search_time_ms": search_time * 1000,
        "search_time_percent": search_time_perc,
        "lp_time_ms": lp_time * 1000,
        "lp_time_percent": lp_time_perc,
        
        # Profile summary (truncated)
        "profile_summary": profile_summary[:500]  # First 500 chars
    }
    
    # Print summary
    print(f"\nRESULTS:")
    print(f"  Cost: {result['cost']}")
    print(f"  Visited States: {result['visited_states']}")
    print(f"  LP Solved: {result['lp_solved']}")
    print(f"  Total Runtime: {result['total_runtime_ms']:.2f} ms")
    print(f"  Search Time: {result['search_time_ms']:.2f} ms ({result['search_time_percent']:.1f}%)")
    print(f"  LP/ILP Time: {result['lp_time_ms']:.2f} ms ({result['lp_time_percent']:.1f}%)")
    
    # Save individual result
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"exp_{exp_config['exp_id']:04d}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nSaved result to: {result_file}")
    
    return result


# -------------------------------
# RESULTS AGGREGATION
# -------------------------------

def aggregate_results(results_dir="results", output_file="results/summary.csv"):
    """
    Aggregate all individual experiment results into a summary CSV.
    """
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("exp_*.json"))
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return None
    
    print(f"Found {len(json_files)} result files")
    
    # Load all results
    all_results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
            all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by experiment ID
    df = df.sort_values('exp_id')
    
    # Save summary
    df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")
    
    # Print statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"\nBy Variant:")
    summary = df.groupby('variant').agg({
        'total_runtime_ms': ['mean', 'std', 'min', 'max'],
        'visited_states': ['mean', 'std'],
        'cost': ['mean', 'std']
    }).round(2)
    print(summary)
    
    print(f"\nBy Dataset:")
    summary = df.groupby('dataset').agg({
        'total_runtime_ms': ['mean', 'std'],
        'visited_states': ['mean', 'std']
    }).round(2)
    print(summary)
    
    return df


# -------------------------------
# COMMAND LINE INTERFACE
# -------------------------------

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process Mining Heuristics Comparison - Parallel Version"
    )
    parser.add_argument(
        '--run_id', 
        type=int, 
        required=False,
        help='Experiment ID for SLURM array job (0-based)'
    )
    parser.add_argument(
        '--aggregate', 
        action='store_true',
        help='Aggregate all results into summary CSV'
    )
    parser.add_argument(
        '--list-experiments', 
        action='store_true',
        help='List all experiment configurations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results/)'
    )
    
    args = parser.parse_args()
    
    # List experiments
    if args.list_experiments:
        experiments = generate_experiment_grid()
        print(f"\nTotal experiments: {len(experiments)}")
        print("\nFirst 10 experiments:")
        for exp in experiments[:10]:
            print(f"  ID {exp['exp_id']:3d}: {exp['dataset']:8s} | "
                  f"trace={exp['trace_idx']:3d} | variant={exp['variant']}")
        print(f"  ...")
        print(f"\nUse --run_id 0-{len(experiments)-1} to run specific experiment")
        return
    
    # Aggregate results
    if args.aggregate:
        aggregate_results(args.output_dir)
        return
    
    # Run single experiment
    if args.run_id is not None:
        exp_config = get_experiment_by_id(args.run_id)
        result = run_single_experiment(exp_config, args.output_dir)
        
        if result is None:
            sys.exit(1)
        
        print("\n" + "="*80)
        print(f"EXPERIMENT {args.run_id} COMPLETED SUCCESSFULLY")
        print("="*80)
        return
    
    # No arguments - show help
    parser.print_help()
    print("\nExamples:")
    print("  # List all experiments")
    print("  python sandbox_heuristics_parallel.py --list-experiments")
    print("\n  # Run specific experiment (for SLURM)")
    print("  python sandbox_heuristics_parallel.py --run_id 0")
    print("\n  # Aggregate all results")
    print("  python sandbox_heuristics_parallel.py --aggregate")


if __name__ == "__main__":
    main()