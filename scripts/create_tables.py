from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence, Literal
import math
import re
from pathlib import Path
import json

@dataclass(frozen=True)
class MetricSpec:
    """One table row."""
    label_tex: str
    getter: Callable[[Mapping[str, Any]], Any]
    fmt: Callable[[Any], str]


def _escape_latex(text: str) -> str:
    # conservative LaTeX escaping for captions/labels/plain text cells
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in text)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fmt_float(ndp: int = 3) -> Callable[[Any], str]:
    def f(x: Any) -> str:
        if x is None:
            return "-"
        if _is_number(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return "-"
            return f"{float(x):.{ndp}f}"
        return "-"
    return f


def _fmt_ms(ndp: int = 3) -> Callable[[Any], str]:
    # input seconds -> milliseconds formatted
    def f(x: Any) -> str:
        if x is None or not _is_number(x):
            return "-"
        v = float(x) * 1000.0
        return f"{v:.{ndp}f}"
    return f


def _fmt_percent_from_unit(ndp: int = 2) -> Callable[[Any], str]:
    # input in [0,1] -> percent string
    def f(x: Any) -> str:
        if x is None or not _is_number(x):
            return "-"
        v = float(x) * 100.0
        return f"{v:.{ndp}f}\\%"
    return f


def _get_path(d: Mapping[str, Any], path: Sequence[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _tol_to_int(t: str) -> int:
    m = re.match(r"^\s*(\d+)\s*%\s*$", str(t))
    if not m:
        raise ValueError(f"Bad tolerance key: {t!r}")
    return int(m.group(1))


def _fmt_int(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "-"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float) and math.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return "-"  # keep strict


def _fmt_percent_from_unit(x: Any, ndp: int = 2) -> str:
    if x is None or not isinstance(x, (int, float)) or isinstance(x, bool):
        return "-"
    x = float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x * 100.0:.{ndp}f}"


def _fmt_percent_from_percent(x: Any, ndp: int = 2) -> str:
    # if metric is already in percent space [0..100]
    if x is None or not isinstance(x, (int, float)) or isinstance(x, bool):
        return "-"
    x = float(x)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{ndp}f}"


def make_default_metric_specs(tolerances: Iterable[str] = ("0%", "10%", "20%")) -> list[MetricSpec]:
    """
    Produces a row-set similar to your example table.
    Adjust tolerances and/or add/remove specs as needed.
    """
    specs: list[MetricSpec] = [
        MetricSpec(
            label_tex=r"$R_{align}$",
            getter=lambda r: r.get("performance_ratio_alignment_only"),
            fmt=_fmt_float(3),
        ),
        MetricSpec(
            label_tex=r"$R_{align + pred}$",
            getter=lambda r: r.get("performance_ratio_with_prediction"),
            fmt=_fmt_float(3),
        ),
        MetricSpec(
            label_tex=r"Align Time (s)",
            getter=lambda r: r.get("mean_alignment_time_only"),
            fmt=_fmt_float(3),
        ),
        MetricSpec(
            label_tex=r"Pred + Align Time (s)",
            getter=lambda r: r.get("mean_alignment_time_with_prediction"),
            fmt=_fmt_float(3),
        ),
        MetricSpec(
            label_tex=r"Pred Time (ms)",
            getter=lambda r: r.get("mean_prediction_time"),  # optional if you have it
            fmt=_fmt_ms(3),
        ),
        MetricSpec(
            label_tex=r"Feature Extraction Time (ms)",
            getter=lambda r: r.get("mean_prediction_time"),  # optional if you have it
            fmt=_fmt_ms(3),
        ),
        MetricSpec(
            label_tex=r"$S_{worst}$",
            getter=lambda r: r.get("time_savings_vs_worst"),
            fmt=_fmt_percent_from_unit(2),
        ),
    ]

    for tol in tolerances:
        specs.extend([
            MetricSpec(
                label_tex=rf"Acc {tol}",
                getter=lambda r, tol=tol: _get_path(r, ("tolerance_metrics", tol, "overall_accuracy")),
                fmt=_fmt_float(3),
            ),
            MetricSpec(
                label_tex=rf"Macro {tol}",
                getter=lambda r, tol=tol: _get_path(r, ("tolerance_metrics", tol, "macro_accuracy")),
                fmt=_fmt_float(3),
            ),
        ])

    return specs


def iter_tex_tables(
    res: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    dataset_keys: Sequence[str] | None = None,
    include_overall: bool = False,
    overall_key: str = "overall",
    model_order: Sequence[str] | None = None,
    model_rename: Mapping[str, str] | None = None,
    metric_specs: Sequence[MetricSpec] | None = None,
    caption_fn: Callable[[str], str] | None = None,
    label_fn: Callable[[str], str] | None = None,
    table_position: str = "ht",
    float_fmt_fallback: Callable[[Any], str] | None = None,
) -> Iterable[tuple[str, str, str]]:
    """
    Generator yielding (dataset_key, tex_string, debug_info).

    res is assumed to be:
      res[model][dataset_key] -> dict with fields including:
        tolerance_metrics, mean_alignment_time_only, performance_ratio_with_prediction, ...

    Notes:
    - If dataset_keys is None, it will use the union of all dataset keys across models.
    - If include_overall is True, it will also generate a table for overall_key if present.
    - metric_specs controls rows; use make_default_metric_specs(...) or provide your own.
    """
    if model_rename is None:
        model_rename = {}
    if metric_specs is None:
        metric_specs = make_default_metric_specs()
    if float_fmt_fallback is None:
        float_fmt_fallback = _fmt_float(3)

    models = list(model_order) if model_order is not None else sorted(res.keys())

    # compute dataset keys
    if dataset_keys is None:
        key_set = set()
        for m in models:
            if m in res:
                key_set.update(res[m].keys())
        keys = sorted(key_set)
    else:
        keys = list(dataset_keys)

    if include_overall and overall_key not in keys:
        keys.append(overall_key)

    def default_caption(ds: str) -> str:
        # customize as you like
        if ds == overall_key:
            return r"\textbf{Overall} Model Comparison"
        return rf"\textbf{{{_escape_latex(ds)}}} Model Comparison"

    def default_label(ds: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", ds).strip("_").lower()
        return f"tab:{safe}_model_comparison"

    caption_fn = caption_fn or default_caption
    label_fn = label_fn or default_label

    for ds in keys:
        # Collect per-model result dict (may be missing)
        per_model: dict[str, Mapping[str, Any] | None] = {
            m: res.get(m, {}).get(ds) for m in models
        }

        # Build header
        display_models = [model_rename.get(m, m) for m in models]
        cols_spec = "l" + ("r" * len(models))

        lines: list[str] = []
        lines.append(rf"\begin{{table}}[{table_position}]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{caption_fn(ds)}}}")
        lines.append(rf"\label{{{_escape_latex(label_fn(ds))}}}")
        lines.append(rf"\begin{{tabular}}{{{cols_spec}}}")
        lines.append(r"\hline")
        header = " & ".join([r"\textbf{Metric}"] + [rf"\textbf{{{_escape_latex(name)}}}" for name in display_models])
        lines.append(header + r" \\")
        lines.append(r"\hline")

        # Rows
        missing_notes = []
        for spec in metric_specs:
            row = [spec.label_tex]
            for m in models:
                rdict = per_model[m]
                if rdict is None:
                    row.append("-")
                    missing_notes.append(f"missing {ds} for model={m}")
                    continue
                try:
                    val = spec.getter(rdict)
                except Exception as e:
                    val = None
                    missing_notes.append(f"getter error on {ds}/{m}/{spec.label_tex}: {e!r}")

                try:
                    cell = spec.fmt(val)
                except Exception as e:
                    cell = float_fmt_fallback(val)
                    missing_notes.append(f"fmt error on {ds}/{m}/{spec.label_tex}: {e!r}")

                row.append(cell if cell is not None else "-")

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        tex = "\n".join(lines)
        debug = "\n".join(missing_notes)
        yield ds, tex, debug

def iter_tex_detail_tables(
    res: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    model: str,
    dataset: str,
    tolerances: Sequence[str] = ("0%", "10%", "20%"),
    detail: Literal["combination", "heuristic"] = "combination",
    metric_key: str = "recall",
    metric_is: Literal["unit", "percent"] = "unit",
    sort_by: Literal["support", "correct", "metric", "name"] = "support",
    sort_desc: bool = True,
    top_k: int | None = None,
    min_support: int | None = None,
    caption_prefix: str | None = None,
    label_prefix: str = "tab",
    table_position: str = "ht",
    model_display: str | None = None,
    dataset_display: str | None = None,
    tolerance_in_caption: bool = True,
) -> Iterable[tuple[str, str]]:
    """
    Generates LaTeX tables for per-combination or per-heuristic metrics at given tolerances.

    Yields: (tolerance_key, tex_table_string)

    Expects your JSON shape (as shown):
      res[model][dataset]["tolerance_metrics"][tol]["combination_metrics"][name] -> {support, correct_predictions, recall, ...}
      res[model][dataset]["tolerance_metrics"][tol]["per_heuristic_metrics"][name] -> {total_optimal_samples, true_positives, recall, f1_score, accuracy, ...}

    For combination_metrics:
      - support key: "support"
      - correct key: "correct_predictions"
      - metric key: metric_key (e.g. "recall", "f1_score", ...)

    For per_heuristic_metrics:
      - support key defaults to "total_optimal_samples" if present else ("true_positives"+"false_negatives")
      - correct key defaults to "true_positives"
      - metric key: metric_key (e.g. "recall", "f1_score", "accuracy", ...)

    metric_is:
      - "unit": metric values are in [0,1] (converted to %)
      - "percent": metric values already in [0,100]
    """
    model_display = model_display or model
    dataset_display = dataset_display or dataset

    if model not in res or dataset not in res[model]:
        raise KeyError(f"Missing res[{model!r}][{dataset!r}]")

    root = res[model][dataset]
    tol_metrics = root.get("tolerance_metrics", {})
    if not isinstance(tol_metrics, Mapping):
        raise ValueError("tolerance_metrics missing or not a dict")

    # nice ordering if user passes unsorted tolerances
    tolerances = sorted(tolerances, key=_tol_to_int)

    # Choose section + default header nouns
    if detail == "combination":
        section_key = "combination_metrics"
        first_col_name = "Combination"
        # keys inside each entry
        support_get: Callable[[Mapping[str, Any]], Any] = lambda d: d.get("support")
        correct_get: Callable[[Mapping[str, Any]], Any] = lambda d: d.get("correct_predictions")
    else:
        section_key = "per_heuristic_metrics"
        first_col_name = "Heuristic"

        def support_get(d: Mapping[str, Any]) -> Any:
            if "total_optimal_samples" in d:
                return d.get("total_optimal_samples")
            # fallback: tp + fn if present
            tp = d.get("true_positives")
            fn = d.get("false_negatives")
            if isinstance(tp, (int, float)) and isinstance(fn, (int, float)):
                return int(tp + fn)
            return d.get("support")  # last resort

        correct_get = lambda d: d.get("true_positives")

    # metric formatter
    if metric_is == "unit":
        metric_fmt = lambda v: _fmt_percent_from_unit(v, ndp=2)
    else:
        metric_fmt = lambda v: _fmt_percent_from_percent(v, ndp=2)

    # caption/label defaults
    def default_caption(tol: str) -> str:
        # Example wants: "Per-Combination Performance \textbf{GNN-Transformer} (10\% Tolerance)"
        prefix = caption_prefix or f"Per-{first_col_name} Performance"
        tol_part = f" ({_escape_latex(tol)} Tolerance)" if tolerance_in_caption else ""
        return rf"{_escape_latex(prefix)} \textbf{{{_escape_latex(model_display)}}}{tol_part}"

    def default_label(tol: str) -> str:
        # Example: tab:per_combination_10_gnn
        tol_int = _tol_to_int(tol)
        md = re.sub(r"[^a-zA-Z0-9]+", "_", model_display).strip("_").lower()
        return f"{label_prefix}:per_{detail}_{tol_int}_{md}"

    for tol in tolerances:
        tol_block = tol_metrics.get(tol)
        if not isinstance(tol_block, Mapping):
            continue

        section = tol_block.get(section_key, {})
        if not isinstance(section, Mapping) or not section:
            continue

        # collect rows
        rows = []
        for name, entry in section.items():
            if not isinstance(entry, Mapping):
                continue
            support = support_get(entry)
            correct = correct_get(entry)
            metric_val = entry.get(metric_key)

            # min_support filter
            if min_support is not None:
                try:
                    s_int = int(support) if support is not None else 0
                except Exception:
                    s_int = 0
                if s_int < min_support:
                    continue

            rows.append({
                "name": str(name),
                "support": support,
                "correct": correct,
                "metric": metric_val,
            })

        # sorting
        if sort_by == "support":
            key_fn = lambda r: (int(r["support"]) if str(r["support"]).isdigit() else -1)
        elif sort_by == "correct":
            key_fn = lambda r: (int(r["correct"]) if str(r["correct"]).isdigit() else -1)
        elif sort_by == "metric":
            def key_fn(r):
                v = r["metric"]
                return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else float("-inf")
        else:
            key_fn = lambda r: r["name"].lower()

        rows.sort(key=key_fn, reverse=sort_desc)

        if top_k is not None:
            rows = rows[:top_k]

        # build LaTeX
        lines = []
        lines.append(rf"\begin{{table}}[{table_position}]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{default_caption(tol)}}}")
        lines.append(rf"\label{{{_escape_latex(default_label(tol))}}}")
        lines.append(r"\begin{tabular}{l r r r}")
        lines.append(r"\hline")
        metric_title = f"{_escape_latex(metric_key.replace('_', ' ').title())} (\\%)"
        lines.append(rf"\textbf{{{first_col_name}}} & \textbf{{Support}} & \textbf{{Correct}} & \textbf{{{metric_title}}} \\")
        lines.append(r"\hline")

        for r in rows:
            name_tex = _escape_latex(r["name"]).replace(r"\+", "+")  # keep '+' readable if present
            lines.append(
                f"{name_tex} & {_fmt_int(r['support'])} & {_fmt_int(r['correct'])} & {metric_fmt(r['metric'])} \\\\"
            )

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        yield tol, "\n".join(lines)

def create_all(fp: Path):
    plot_dir = fp / "tables"
    plot_dir.mkdir(exist_ok=True)
    res = json.load(open(fp / "full.json"))

    specs = make_default_metric_specs(tolerances=("0%", "10%", "20%"))
    with open(plot_dir / "ds_summaries.tex", "w") as f:
        for ds, tex, debug in iter_tex_tables(
            res,
            dataset_keys=None,                 # or a list in the order you want
            include_overall=True,
            overall_key="overall",
            model_order=["SingleBestSolver", "RandomClassifier", "GNNTransformer", "XGBoostClassifier"],
            model_rename={
                "XGBoostClassifier": "XGBoost",
                "GNNTransformer": "GNN-Transformer",
            },
            metric_specs=specs,
        ):
            f.write(f"=== {ds} ===\n")
            f.write(tex)
            f.write("\n")


    models = ["GNNTransformer", "XGBoostClassifier"]

    for model in models:
        with open(plot_dir / f"{model}_recall.tex", "w") as f:
            for tol, tex in iter_tex_detail_tables(
                res,
                model=model,
                dataset="overall",                 # or your dataset id/name
                tolerances=("0%", "10%", "20%"),
                detail="combination",
                metric_key="recall",               # or "f1_score", "precision", ...
                metric_is="unit",                  # your JSON recall is in [0,1]
                sort_by="support",
                sort_desc=True,
            ):
                f.write(tex)
                f.write("\n")

        with open(plot_dir / f"{model}_f1.tex", "w") as f:
            for tol, tex in iter_tex_detail_tables(
                res,
                model=model,
                dataset="overall",                 # or your dataset id/name
                tolerances=("0%", "10%", "20%"),
                detail="combination",
                metric_key="f1_score",               # or "f1_score", "precision", ...
                metric_is="unit",                  # your JSON recall is in [0,1]
                sort_by="support",
                sort_desc=True,
            ):
                f.write(tex)
                f.write("\n")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--eval-dir", type=str)

    args = parser.parse_args()
    create_all(Path(args.eval_dir))

