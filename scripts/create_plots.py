import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
def build_dataset_mapping():
    """
    Build mapping from UUID to dataset name by scanning data directory.
    
    Returns:
        dict: {uuid: dataset_name}
    """
    mapping = {}
    
    for uuid_dir in DATA_DIR.iterdir():
        if not uuid_dir.is_dir():
            continue
        
        uuid = uuid_dir.name
        
        # Find the first .xes or .csv file
        for file in uuid_dir.iterdir():
            if file.suffix in ['.xes', '.csv']:
                # Use filename without extension as dataset name
                name = file.stem.replace('%20', ' ')
                mapping[uuid] = name
                break
    
    return mapping

DATASET_NAMES = build_dataset_mapping()
MODEL_RENAME = {
    "XGBoostClassifier": "XGBoost",
    "GNNTransformer": "GNN-Transformer",
}
GRAYSCALE_MODELS = {
    "RandomClassifier",
    "SingleBestSolver"
}
models = ["SingleBestSolver", "RandomClassifier", "GNN-Transformer", "XGBoost"]

def performance_ratio(results: dict, model_palette: dict, normalize: bool = False):
    import pandas as pd
    rows = []

    for model, evaluated_data in results.items():
        for dataset_name, eval_result in evaluated_data.items():
            rows.append({
                "model": MODEL_RENAME.get(model, model),
                "dataset": DATASET_NAMES.get(dataset_name, dataset_name),
                "performance_ratio": eval_result["performance_ratio_with_prediction"]
            })

    df = pd.DataFrame(rows)
    datasets = sorted(df["dataset"].unique())
    if normalize:
        df["performance_ratio_norm"] = (
            df
            .groupby("dataset")["performance_ratio"]
            .transform(lambda x: x / x.max())
        )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(
        data=df,
        x="dataset",
        order=datasets,
        y="performance_ratio_norm" if normalize else "performance_ratio",
        hue="model",
        hue_order=models,
        palette=model_palette,
        errorbar=None  # important if values are single measurements
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if not normalize:
        # draw the 1.0 "optimal" line
        ax.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.7,
            label="optimal"
        )

    plt.ylabel("$R_{align}$")
    # plt.yscale("log")
    plt.xlabel("Dataset")
    # plt.title("Model Performance Ratio $R_{align}$ Across Datasets")
    plt.legend(title="Model")
    plt.tight_layout()

def tolerance_based(results: dict, model_palette: dict, metric_key: str = "overall_accuracy"):

    # 1) Build long-form dataframe: one row per (model, dataset, tolerance)
    rows = []
    for model, evaluated_data in results.items():
        for dataset_name, eval_result in evaluated_data.items():
            tol_metrics = eval_result.get("tolerance_metrics", {})
            for tol_key, tol_payload in tol_metrics.items():
                rows.append({
                    "model": MODEL_RENAME.get(model, model),
                    "dataset": DATASET_NAMES.get(dataset_name, dataset_name),
                    "tolerance": tol_key,
                    "tolerance_int": int(tol_key.rstrip("%")),
                    metric_key: float(tol_payload[metric_key]),
                })
    df = pd.DataFrame(rows)

    datasets = sorted(df["dataset"].unique())

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No macro_accuracy values found under tolerance_metrics.")

    # 2) Sort tolerances and compute stackable deltas within each (model, dataset)
    df = df.sort_values(["dataset", "model", "tolerance_int"])
    df["prev_acc"] = df.groupby(["dataset", "model"])[metric_key].shift(1).fillna(0.0)
    df["delta_acc"] = df[metric_key] - df["prev_acc"]

    # 3) Pivot deltas to wide: columns=tolerances, rows=(dataset, model)
    wide = df.pivot_table(
        index=["dataset", "model"],
        columns="tolerance_int",
        values="delta_acc",
        aggfunc="first",
        fill_value=0.0
    ).sort_index()

    # Ensure tolerances are in ascending order
    tols = list(wide.columns)

    # 4) Plot grouped + stacked bars
    from matplotlib.patches import Patch

    sns.set_theme(style="whitegrid")

    x = np.arange(len(datasets))
    n_models = len(models)
    group_width = 0.8
    bar_width = group_width / max(n_models, 1)

    alphas = 1.0 - np.linspace(0.25, 0.75, num=len(tols))
    tol_alpha = dict(zip(tols, alphas))

    fig, ax = plt.subplots(figsize=(15, 8))

    for mi, model in enumerate(models):
        bottoms = np.zeros(len(datasets), dtype=float)
        xpos = x - group_width/2 + (mi + 0.5) * bar_width

        base_color = model_palette[model]

        for tol in tols:
            heights = np.array(
                [wide.loc[(ds, model), tol] if (ds, model) in wide.index else 0.0 for ds in datasets],
                dtype=float
            )

            ax.bar(
                xpos,
                heights,
                bottom=bottoms,
                width=bar_width,
                color=base_color,          # model color
                alpha=tol_alpha[tol],      # tolerance tone
                edgecolor="none"
            )
            bottoms += heights

    # Axes cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(f"{metric_key} (stacked as incremental gain by tolerance)")
    ax.set_xlabel("Dataset")
    # ax.set_title(f"{KEY} vs tolerance, grouped by model")

    # ---- Legends ----
    # Legend 1: models
    model_handles = [Patch(facecolor=model_palette[m], edgecolor="none", label=m) for m in models]
    leg1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0
    )
    ax.add_artist(leg1)

    # Legend 2: tolerances (alpha)
    tol_handles = [
        Patch(facecolor="black", edgecolor="none", alpha=tol_alpha[t], label=f"{t}%")
        for t in tols
    ]
    ax.legend(
        handles=tol_handles,
        title="Tolerance (opacity)",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.75),
        borderaxespad=0.0
    )

    plt.tight_layout(w_pad=1.5)

def create_all(fp: Path):
    plot_dir = fp / "plots"
    plot_dir.mkdir(exist_ok=True)
    results = json.load(open(fp / "full.json"))
    # models = sorted(list(set(results.keys())))
    
    # model_palette = dict(zip(models, sns.color_palette("tab10", n_colors=len(models))))
    grayscale_models = set(GRAYSCALE_MODELS)  # if it’s a dict, we only care about keys

    # split models
    gray_models = [m for m in models if m in grayscale_models]
    color_models = [m for m in models if m not in grayscale_models]

    # generate palettes
    color_palette = sns.color_palette("tab10", n_colors=len(color_models))
    gray_palette = sns.color_palette("Greys", n_colors=len(gray_models) + 4)[2:-2]

    # build explicit mapping
    model_palette = {}

    for m, c in zip(color_models, color_palette):
        model_palette[m] = c

    for m, c in zip(gray_models, gray_palette):
        model_palette[m] = c


    performance_ratio(results, model_palette)
    plt.savefig(plot_dir / "performance_ratios.png")

    performance_ratio(results, model_palette, normalize=True)
    plt.savefig(plot_dir / "performance_ratios_normalized.png")

    tolerance_based(results, model_palette, "overall_accuracy")
    plt.savefig(plot_dir / "overall_accuracy.png")

    tolerance_based(results, model_palette, "macro_accuracy")
    plt.savefig(plot_dir / "macro_accuracy.png")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--eval-dir", type=str)

    args = parser.parse_args()

    create_all(Path(args.eval_dir))


