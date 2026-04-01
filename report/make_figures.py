from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def make_pipeline(assets: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_axis_off()

    labels = [
        "Patient folder\npatient_XX/\ntext_0..N.md",
        "Stage A\nNote extraction\n(line-numbered)",
        "Candidate conditions\nper note\n(taxonomy-valid)",
        "Stage B\nPatient consolidation\n(dedupe, onset, status)",
        "Output\npatient_XX.json",
    ]

    x0, y0 = 0.02, 0.35
    w, h = 0.18, 0.35
    xs = [x0 + i * (w + 0.02) for i in range(len(labels))]

    for x, lab in zip(xs, labels):
        box = FancyBboxPatch(
            (x, y0),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#2f2f2f",
            facecolor="#f6f8ff",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h / 2, lab, ha="center", va="center", fontsize=10)

    for i in range(len(labels) - 1):
        x1 = xs[i] + w
        x2 = xs[i + 1]
        ax.annotate(
            "",
            xy=(x2, y0 + h / 2),
            xytext=(x1, y0 + h / 2),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#2f2f2f"),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig(fig, assets / "pipeline")
    plt.close(fig)


def make_taxonomy_overview(assets: Path, repo_root: Path) -> None:
    taxonomy_path = repo_root / "clinical_nlp_assignment" / "taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    cats = taxonomy["condition_categories"]
    cat_names = list(cats.keys())
    sub_counts = [len(cats[k].get("subcategories", {})) for k in cat_names]

    order = np.argsort(sub_counts)[::-1]
    cat_names_ord = [cat_names[i] for i in order]
    sub_counts_ord = [sub_counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(cat_names_ord, sub_counts_ord, color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Number of subcategories")
    ax.set_title("Taxonomy overview: subcategories per category")
    for y, v in enumerate(sub_counts_ord):
        ax.text(v + 0.05, y, str(v), va="center", fontsize=9)

    save_fig(fig, assets / "taxonomy_overview")
    plt.close(fig)


def make_metrics_template(assets: Path) -> None:
    # Template plot to include in the report even if evaluation hasn't been run.
    metrics = {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(keys, vals, color=["#59a14f", "#edc948", "#e15759"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Development metrics (template)")
    ax.text(
        0.5,
        -0.18,
        "Update after running: python -m clinical_nlp_assignment.train",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_fig(fig, assets / "metrics_plot")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets = repo_root / "report" / "assets"
    ensure_dir(assets)

    make_pipeline(assets)
    make_taxonomy_overview(assets, repo_root)
    make_metrics_template(assets)

    print(f"Wrote figures to: {assets}")


if __name__ == "__main__":
    main()

