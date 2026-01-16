#!/usr/bin/env python3
from __future__ import annotations

from dataloaders.labels import LabelDataset
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def interp_categorical(
    p0: list[float], p1: list[float], t: float
) -> list[float]:
    p = np.array([lerp(a, b, t) for a, b in zip(p0, p1)], dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p.tolist()


def _extract_probs(op_spec) -> list[float]:
    return list(map(float, op_spec.probs))


def _extract_poisson_rate(poisson_spec) -> float:
    return float(poisson_spec.rate)


def _extract_pstop(pstop_spec) -> tuple[float, float]:
    return float(pstop_spec.base), float(pstop_spec.slope)


def make_interpolated_param_grid(
    count_per_alpha: int,
    alphas: np.ndarray,
    min_depth: int,
    max_depth: int,
):
    """
    Builds a param_grid suitable for SyntheticProcessModelDataset(param_grid=[(cfg, reps), ...])

    IMPORTANT:
      - Each cfg includes "alpha" so you can plot against it later.
      - This interpolates between two *endpoints* that you can tweak.
    """
    from util.distributions import (
        CategoricalSpec,
        PoissonSpec,
        BernoulliDepthLinearSpec,
    )

    # Endpoint A: shallow+wide xor-dominant
    A = {
        "dist_params": {
            "op": [0.6, 0.1, 0.2, 0.1],
            "seq_len_mu": 1.0,
            "p_stop_base": 0.5,
            "p_stop_slope": 0.3,
            "width_mu": 12.0,
        },
        "min_depth": min_depth,
        "max_depth": max_depth,
    }

    # Endpoint B: more and-dominant family
    B = {
        "dist_params": {
            "op": [0.1, 0.6, 0.2, 0.1],
            "seq_len_mu": 4.0,
            "p_stop_base": 0.2,
            "p_stop_slope": 0.1,
            "width_mu": 3.0,
        },
        "min_depth": min_depth,
        "max_depth": max_depth,
    }

    grid: list[tuple[dict, int]] = []
    for t in alphas:
        cfg = deepcopy(A)

        op = interp_categorical(
            A["dist_params"]["op"], B["dist_params"]["op"], float(t)
        )
        seq_len_mu = lerp(
            A["dist_params"]["seq_len_mu"],
            B["dist_params"]["seq_len_mu"],
            float(t),
        )
        width_mu = lerp(
            A["dist_params"]["width_mu"],
            B["dist_params"]["width_mu"],
            float(t),
        )
        base = lerp(
            A["dist_params"]["p_stop_base"],
            B["dist_params"]["p_stop_base"],
            float(t),
        )
        slope = lerp(
            A["dist_params"]["p_stop_slope"],
            B["dist_params"]["p_stop_slope"],
            float(t),
        )

        cfg["dist_params"] = {
            "op": CategoricalSpec(op),
            "seq_len": PoissonSpec(seq_len_mu),
            "p_stop": BernoulliDepthLinearSpec(base=base, slope=slope),
            "width": PoissonSpec(width_mu),
        }

        # Persist alpha for plotting
        cfg["alpha"] = float(t)

        grid.append((cfg, count_per_alpha))

    return grid


def plot_interpolation_params_from_param_grid(
    param_grid: list[tuple[dict, int]],
    out_path: Path = Path("./interp_params.png"),
    op_labels: list[str] | None = None,
):
    """
    param_grid: list of (cfg, reps) where cfg contains:
      cfg["alpha"] (float)
      cfg["dist_params"]["op"] (CategoricalSpec-like)
      cfg["dist_params"]["seq_len"] (PoissonSpec-like)
      cfg["dist_params"]["p_stop"] (BernoulliDepthLinearSpec-like)
      cfg["dist_params"]["width"] (PoissonSpec-like)
    """

    # one row per cfg (ignore reps for this visualization)
    cfgs = [cfg for (cfg, _reps) in param_grid]
    cfgs = sorted(cfgs, key=lambda c: float(c.get("alpha", 0.0)))

    alphas = np.array([float(c["alpha"]) for c in cfgs], dtype=float)

    # extract series
    op_probs = np.array(
        [_extract_probs(c["dist_params"]["op"]) for c in cfgs], dtype=float
    )

    seq_rate = np.array(
        [_extract_poisson_rate(c["dist_params"]["seq_len"]) for c in cfgs],
        dtype=float,
    )
    width_rate = np.array(
        [_extract_poisson_rate(c["dist_params"]["width"]) for c in cfgs],
        dtype=float,
    )

    p_base, p_slope = zip(
        *[_extract_pstop(c["dist_params"]["p_stop"]) for c in cfgs]
    )
    p_base = np.array(p_base, dtype=float)
    p_slope = np.array(p_slope, dtype=float)

    # normalize seq_len + width to [0,1] for right axis
    def _norm01(v: np.ndarray) -> tuple[np.ndarray, float, float]:
        vmin, vmax = float(v.min()), float(v.max())
        if vmax <= vmin:
            return np.zeros_like(v), vmin, vmax
        return (v - vmin) / (vmax - vmin), vmin, vmax

    seq_norm, seq_min, seq_max = _norm01(seq_rate)
    width_norm, w_min, w_max = _norm01(width_rate)

    # normalize width to [0,1] for right axis
    w_min, w_max = float(width_rate.min()), float(width_rate.max())
    width_norm = (
        (width_rate - w_min) / (w_max - w_min)
        if w_max > w_min
        else np.zeros_like(width_rate)
    )

    n_ops = op_probs.shape[1]
    if op_labels is None:
        op_labels = [f"op[{i}]" for i in range(n_ops)]

    # --- colors ---
    # Operators: close in hue but distinguishable (same colormap, different shades)
    op_cmap = mpl.cm.Blues
    op_colors = [
        op_cmap(0.35 + 0.5 * (i / max(1, n_ops - 1))) for i in range(n_ops)
    ]

    # Other params: distinct “type” hues
    seq_color = mpl.cm.Oranges(0.70)
    pbase_color = mpl.cm.Greens(0.70)
    pslope_color = mpl.cm.Greens(0.45)
    width_color = mpl.cm.Purples(0.70)

    fig, ax = plt.subplots(figsize=(12, 5))

    # operator curves (left axis)
    for i in range(n_ops):
        ax.plot(
            alphas,
            op_probs[:, i],
            label=f"{op_labels[i]} prob",
            color=op_colors[i],
            linewidth=2,
        )

    # other left-axis curves
    ax.plot(
        alphas, p_base, label="p_stop.base", color=pbase_color, linewidth=2
    )
    ax.plot(
        alphas,
        p_slope,
        label="p_stop.slope",
        color=pslope_color,
        linewidth=2,
        linestyle=":",
    )

    ax.set_xlabel("interpolation alpha")
    ax.set_ylabel("parameter value (native scale)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("probabilities / Bernoulli params")

    # right axis: normalized width
    ax2 = ax.twinx()
    ax2.plot(
        alphas,
        seq_norm,
        label=f"seq_len.rate (norm)  [raw {seq_min:.2f}..{seq_max:.2f}]",
        color=seq_color,
        linewidth=2,
        linestyle="--",
    )

    ax2.plot(
        alphas,
        width_norm,
        label=f"width.rate (norm)    [raw {w_min:.2f}..{w_max:.2f}]",
        color=width_color,
        linewidth=2,
    )

    ax2.set_ylabel("normalized Poisson rates (0-1)")
    ax2.yaxis.set_label_coords(0.97, 0.5)

    # combined legend (both axes)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(Path(out_path).resolve(), dpi=200)
    plt.close(fig)

    print(f"[OK] wrote: {Path(out_path).resolve()}")


def winner_dataframe(run_ds) -> pd.DataFrame:
    """
    Produces one row per comb_id:
      - alpha (from synthetic model params)
      - winner algo
      - winner score (mean duration-like)
    """
    rows = []

    for comb_id in run_ds.combinations:
        # select the items related to this combination
        items = [
            run_ds.serialized[item_id]
            for item_id in run_ds.combinations[comb_id]
        ]

        # Identify winner by criterion
        min_item = min(items, key=LabelDataset.label_criterion)
        if all([p["duration"] == float("inf") for p in min_item.perf]):
            # ignore always-timeout combinations
            continue

        # Extract alpha (interpolant) we added to synthetic model params
        model_obj = items[0].model.deserialize()
        params = model_obj.params
        alpha = params.get("alpha", None)
        cfg_index = params.get(
            "index", None
        )  # synthetic dataset injects "index"

        rows.append(
            {
                "comb_id": comb_id,
                "alpha": alpha,
                "cfg_index": cfg_index,
                "winner_algo": min_item.algo,
                "winner_score": float(LabelDataset.label_criterion(min_item)),
                "winner_time": (
                    sum(
                        [
                            (
                                20.0
                                if p["duration"] == float("inf")
                                else p["duration"]
                            )
                            for p in min_item.perf
                        ]
                    )
                    / len(min_item.perf)
                ),
            }
        )

    df = pd.DataFrame(rows)

    # Choose x-axis: alpha if present, else discrete config index
    if "alpha" in df.columns and not df["alpha"].isna().all():
        df["x"] = df["alpha"].astype(float)
        df["_x_label"] = "interpolation alpha"
    else:
        df["x"] = df["cfg_index"].astype(int)
        df["_x_label"] = "config index (alpha missing)"

    return df


def plot_and_save_strip(df, out_path: Path):

    if df.empty:
        raise RuntimeError("Empty dataframe.")

    fig = plt.figure(figsize=(12, 5))

    ax = sns.stripplot(
        data=df,
        x="x",
        y="winner_algo",
        hue="winner_time",
        palette="flare",
        jitter=0.28,
        size=3,
        alpha=1.0,
    )

    ax.set_xlabel("interpolation alpha")
    ax.set_ylabel("winning aligner")
    ax.set_title(
        "Winning aligner vs. synthetic model distribution interpolation (colored by duration)"
    )

    fig.tight_layout()
    fig.savefig(Path(out_path).resolve(), dpi=200)
    plt.close(fig)


def plot_and_save_violin(df, out_path: Path):

    if df.empty:
        raise RuntimeError("Empty dataframe.")

    fig = plt.figure(figsize=(12, 5))
    ax = sns.violinplot(
        data=df,
        x="x",
        y="winner_algo",
    )

    ax.set_xlabel(df["_x_label"].iloc[0])
    ax.set_ylabel("winning aligner")
    ax.set_title(
        "Winning aligner vs. interpolation (swarm; colored by duration)"
    )

    fig.tight_layout()
    fig.savefig(Path(out_path).resolve(), dpi=200)
    plt.close(fig)


# ----------------------------
# Main: build dataset, run, plot
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_path",
        type=str,
        default="./experiments/dataset/plots/data",
        help="Base path passed to RunDataset",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--count", type=int, default=10, help="models per alpha"
    )
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--slice_end", type=int, default=16)
    parser.add_argument("--alpha_bins", type=int, default=11)
    parser.add_argument("--min_depth", type=int, default=1)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--out", type=str, default="./")
    args = parser.parse_args()

    from util.rng import RNG
    from dataloaders.synthetic import SyntheticProcessModelDataset
    from dataloaders.runs import RunDataset, AlignerSpec, SyntheticTraceSampler

    RNG.initialize(args.seed)

    alphas = np.linspace(0.0, 1.0, args.alpha_bins)

    param_grid = make_interpolated_param_grid(
        count_per_alpha=args.count,
        alphas=alphas,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    plot_interpolation_params_from_param_grid(
        param_grid,
        Path(args.out) / "interp_params.png",
        op_labels=["xor", "and", "loop", "seq"],
    )

    synthetic_dataset = SyntheticProcessModelDataset(param_grid=param_grid)

    trace_sampler = SyntheticTraceSampler(
        ds=synthetic_dataset,
        seed=RNG.get_seed(),
        batch_size=16,
        slice=range(0, args.slice_end),
        steps=args.steps,
        device=args.device,
    )

    run_ds = RunDataset(
        Path(args.cache_path),
        synthetic_dataset,
        AlignerSpec.A_STAR.value,
        trace_sampler,
        n_runs=args.n_runs,
        n_workers=args.n_workers,
    )

    df = winner_dataframe(run_ds)
    plot_and_save_strip(df, Path(args.out) / "winner_strip_heat.png")
    plot_and_save_violin(df, Path(args.out) / "winner_violin_heat.png")


if __name__ == "__main__":
    main()
