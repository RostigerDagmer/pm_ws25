"""
Explore State Space Size Feature

This script creates synthetic process models with varying complexity
(number of AND-splits) to visualize the relationship between model
structure and state space size.

The script:
1. Generates manually constructed process models with different numbers of parallel branches
2. Generates random synthetic process models with varying complexity
3. Calculates state space size for each model
4. Visualizes all models with their state space size
5. Saves visualizations and statistics
"""

import logging
from pathlib import Path
from experiments.simulation.models import seq
from experiments.simulation.structured_net import StructuredNet
from features.base_extractor import StateSpaceSizeExtractor, ModelFeatureExtractor
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from util.distributions import CategoricalSpec, PoissonSpec, BernoulliDepthLinearSpec
from dataloaders.synthetic import SyntheticProcessModelDataset
from util.rng import RNG
import numpy as np

logging.basicConfig(level=logging.INFO)


def create_simple_activity(label: str) -> StructuredNet:
    """Create a simple sequential activity."""
    return seq(f"activity_{label}", [label])


def create_models_with_varying_parallelism():
    """
    Create a set of process models with increasing parallelism.

    Returns:
        List of tuples (model_name, net, im, fm, num_parallel_branches)
    """
    models = []

    # Model 1: Simple sequential (no parallelism)
    A = create_simple_activity("A")
    B = create_simple_activity("B")
    C = create_simple_activity("C")
    seq_model = A >> B >> C
    models.append(("Sequential (A->B->C)", seq_model, 0))

    # Model 2: Simple XOR (no parallelism)
    D = create_simple_activity("D")
    E = create_simple_activity("E")
    xor_model = D ^ E
    models.append(("XOR (D xor E)", xor_model, 0))

    # Model 3: 2 parallel branches
    F = create_simple_activity("F")
    G = create_simple_activity("G")
    par2_model = F & G
    models.append(("Parallel-2 (F || G)", par2_model, 2))

    # Model 4: 3 parallel branches
    H = create_simple_activity("H")
    I = create_simple_activity("I")
    J = create_simple_activity("J")
    par3_model = StructuredNet.n_and([H, I, J])
    models.append(("Parallel-3 (H || I || J)", par3_model, 3))

    # Model 5: 4 parallel branches
    K = create_simple_activity("K")
    L = create_simple_activity("L")
    M = create_simple_activity("M")
    N = create_simple_activity("N")
    par4_model = StructuredNet.n_and([K, L, M, N])
    models.append(("Parallel-4 (K || L || M || N)", par4_model, 4))

    # Model 6: 5 parallel branches
    O = create_simple_activity("O")
    P = create_simple_activity("P")
    Q = create_simple_activity("Q")
    R = create_simple_activity("R")
    S = create_simple_activity("S")
    par5_model = StructuredNet.n_and([O, P, Q, R, S])
    models.append(("Parallel-5 (O || P || Q || R || S)", par5_model, 5))

    # Model 7: Nested parallelism - 2x(2 parallel)
    T = create_simple_activity("T")
    U = create_simple_activity("U")
    V = create_simple_activity("V")
    W = create_simple_activity("W")
    nested1 = (T & U) & (V & W)
    models.append(("Nested 2x2 ((T||U) || (V||W))", nested1, 4))

    # Model 8: Complex - Sequential with embedded parallelism
    X = create_simple_activity("X")
    Y = create_simple_activity("Y")
    Z = create_simple_activity("Z")
    A2 = create_simple_activity("A2")
    B2 = create_simple_activity("B2")
    C2 = create_simple_activity("C2")
    complex_model = X >> (Y & Z & A2) >> (B2 & C2)
    models.append(("Complex (X -> (Y||Z||A2) -> (B2||C2))", complex_model, 5))

    return models


def create_synthetic_dataset(n_models_per_config: int = 5, seed: int = 42):
    """
    Create a synthetic dataset of process models with varying complexity
    using SyntheticProcessModelDataset.

    Args:
        n_models_per_config: Number of models to generate per configuration
        seed: Random seed for reproducibility

    Returns:
        List of tuples (model_name, structured_net, config_type)
    """
    RNG.initialize(seed)

    # Configuration 1: Low AND probability (mostly sequential/XOR)
    config_low_and = {
        "dist_params": {
            "op": CategoricalSpec([0.1, 0.1, 0.6, 0.2]),  # [XOR, AND, SEQ, LOOP]
            "seq_len": PoissonSpec(3),
            "p_stop": BernoulliDepthLinearSpec(base=0.3, slope=0.2),
            "width": PoissonSpec(2),
        },
        "min_depth": 1,
        "max_depth": 3,
    }

    # Configuration 2: Medium AND probability
    config_medium_and = {
        "dist_params": {
            "op": CategoricalSpec([0.2, 0.3, 0.4, 0.1]),  # [XOR, AND, SEQ, LOOP]
            "seq_len": PoissonSpec(3),
            "p_stop": BernoulliDepthLinearSpec(base=0.3, slope=0.2),
            "width": PoissonSpec(3),
        },
        "min_depth": 1,
        "max_depth": 3,
    }

    # Configuration 3: High AND probability (lots of parallelism)
    config_high_and = {
        "dist_params": {
            "op": CategoricalSpec([0.1, 0.6, 0.2, 0.1]),  # [XOR, AND, SEQ, LOOP]
            "seq_len": PoissonSpec(3),
            "p_stop": BernoulliDepthLinearSpec(base=0.3, slope=0.2),
            "width": PoissonSpec(4),
        },
        "min_depth": 1,
        "max_depth": 3,
    }

    # Create param_grid for SyntheticProcessModelDataset
    param_grid = [
        (config_low_and, n_models_per_config),
        (config_medium_and, n_models_per_config),
        (config_high_and, n_models_per_config),
    ]

    # Create the dataset
    synthetic_dataset = SyntheticProcessModelDataset(
        param_grid=param_grid,
        cached=False,
    )

    # Extract models from dataset
    models = []
    config_names = ["Low-AND", "Medium-AND", "High-AND"]

    for i in range(len(synthetic_dataset)):
        item = synthetic_dataset[i]

        # Determine which config this model belongs to
        config_idx = i // n_models_per_config
        config_name = config_names[config_idx] if config_idx < len(config_names) else "High-AND"

        models.append((
            f"Synthetic-{i+1:02d} ({config_name})",
            item.stnet,
            config_name
        ))

    return models


def count_and_splits(net):
    """Count the number of AND-split patterns in a Petri net."""
    model_extractor = ModelFeatureExtractor(use_cache=False)
    features = model_extractor.extract(net, None, None, return_as_dict=True)
    return features.get('model_n_and_split', 0)


def main():
    """Main execution function."""
    output_dir = Path("outputs") / "state_space_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: Manual models
    logging.info("=" * 80)
    logging.info("PART 1: Creating manually constructed models with varying parallelism...")
    logging.info("=" * 80)
    manual_models = create_models_with_varying_parallelism()

    # Part 2: Synthetic dataset
    logging.info("\n" + "=" * 80)
    logging.info("PART 2: Creating synthetic dataset...")
    logging.info("=" * 80)
    synthetic_models = create_synthetic_dataset(n_models_per_config=5, seed=42)

    # Combine all models
    all_models = []
    for name, model, num_parallel in manual_models:
        all_models.append(("Manual", name, model, num_parallel))
    for name, model, config_type in synthetic_models:
        all_models.append(("Synthetic", name, model, config_type))

    logging.info(f"\nTotal models to process: {len(all_models)}")
    logging.info(f"  - Manual models: {len(manual_models)}")
    logging.info(f"  - Synthetic models: {len(synthetic_models)}")

    logging.info("\nExtracting state space sizes...")
    extractor = StateSpaceSizeExtractor(use_cache=False)

    results = []
    for i, model_data in enumerate(all_models):
        if len(model_data) == 4 and model_data[0] == "Manual":
            _, name, model, num_parallel = model_data
            model_type = "Manual"
        else:
            _, name, model, config_type = model_data
            model_type = "Synthetic"
            num_parallel = None

        logging.info(f"\nProcessing model {i+1}/{len(all_models)}: {name}")

        net, im, fm = model.net, model.im, model.fm

        # Extract state space size
        state_space_features = extractor.extract(
            net, im, fm, return_as_dict=True
        )
        state_space_size = state_space_features['state_space_size']

        # Count AND splits
        num_and_splits = count_and_splits(net)

        logging.info(f"  State space size: {state_space_size:.4f}")
        logging.info(f"  Number of AND-splits: {num_and_splits}")
        logging.info(f"  Number of transitions: {len(net.transitions)}")
        logging.info(f"  Number of places: {len(net.places)}")

        # Visualize and save the Petri net
        gviz = pn_visualizer.apply(
            net, im, fm,
            parameters={
                pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"
            }
        )

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("|", "_").replace("->", "-")
        viz_path = output_dir / f"model_{i+1:02d}_{safe_name}.png"
        pn_visualizer.save(gviz, str(viz_path))
        logging.info(f"  Saved visualization to: {viz_path}")

        results.append({
            'model_id': i + 1,
            'model_type': model_type,
            'name': name,
            'num_parallel_branches': num_parallel if model_type == "Manual" else None,
            'config_type': config_type if model_type == "Synthetic" else None,
            'state_space_size': state_space_size,
            'num_and_splits': num_and_splits,
            'num_transitions': len(net.transitions),
            'num_places': len(net.places),
        })

    # Save statistics to file
    stats_file = output_dir / "statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("STATE SPACE SIZE EXPLORATION\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'ID':<4} {'Type':<10} {'Name':<35} {'AND':<5} {'SS':<10} {'Trans':<7} {'Places':<7}\n")
        f.write("-" * 90 + "\n")

        for r in results:
            name_short = r['name'][:34]
            f.write(
                f"{r['model_id']:<4} "
                f"{r['model_type']:<10} "
                f"{name_short:<35} "
                f"{r['num_and_splits']:<5} "
                f"{r['state_space_size']:<10.3f} "
                f"{r['num_transitions']:<7} "
                f"{r['num_places']:<7}\n"
            )

        # Summary statistics
        f.write("\n" + "=" * 90 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 90 + "\n\n")

        manual_results = [r for r in results if r['model_type'] == 'Manual']
        synthetic_results = [r for r in results if r['model_type'] == 'Synthetic']

        f.write(f"Total models: {len(results)}\n")
        f.write(f"  - Manual models: {len(manual_results)}\n")
        f.write(f"  - Synthetic models: {len(synthetic_results)}\n\n")

        f.write(f"State Space Size Range:\n")
        f.write(f"  - Min: {min(r['state_space_size'] for r in results):.3f}\n")
        f.write(f"  - Max: {max(r['state_space_size'] for r in results):.3f}\n")
        f.write(f"  - Mean: {np.mean([r['state_space_size'] for r in results]):.3f}\n")
        f.write(f"  - Median: {np.median([r['state_space_size'] for r in results]):.3f}\n\n")

        f.write(f"AND-splits Range:\n")
        f.write(f"  - Min: {min(r['num_and_splits'] for r in results)}\n")
        f.write(f"  - Max: {max(r['num_and_splits'] for r in results)}\n")
        f.write(f"  - Mean: {np.mean([r['num_and_splits'] for r in results]):.2f}\n")

    logging.info(f"\nStatistics saved to: {stats_file}")

    # Print summary
    manual_results = [r for r in results if r['model_type'] == 'Manual']
    synthetic_results = [r for r in results if r['model_type'] == 'Synthetic']
    and_splits = [r['num_and_splits'] for r in results]
    state_space_sizes = [r['state_space_size'] for r in results]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models created: {len(results)}")
    print(f"  - Manual models: {len(manual_results)}")
    print(f"  - Synthetic models: {len(synthetic_results)}")
    print(f"\nState space size range: {min(state_space_sizes):.3f} - {max(state_space_sizes):.3f}")
    print(f"AND-splits range: {min(and_splits)} - {max(and_splits)}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
