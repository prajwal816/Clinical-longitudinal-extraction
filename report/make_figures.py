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
    taxonomy_path = repo_root / "Data" / "taxonomy.json"
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
        "Update after running: python -m Clinical_Nlp_Extraction.train",
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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _patient_ids_from_dir(dirpath: Path) -> list[str]:
    ids: list[str] = []
    for p in dirpath.glob("patient_*"):
        if p.is_dir():
            ids.append(p.name)
    return sorted(ids)


def _count_note_files(patient_dir: Path) -> int:
    return len(list(patient_dir.glob("text_*.md")))


def make_dataset_stats(assets: Path, repo_root: Path) -> None:
    train_dir = repo_root / "Data" / "train"
    dev_dir = repo_root / "Data" / "dev"
    train_labels_dir = train_dir / "labels"

    # Notes-per-patient (histograms)
    def note_counts(d: Path) -> list[int]:
        counts: list[int] = []
        for pid in _patient_ids_from_dir(d):
            pdir = d / pid
            counts.append(_count_note_files(pdir))
        return counts

    train_note_counts = note_counts(train_dir)
    dev_note_counts = note_counts(dev_dir)

    all_counts = sorted(set(train_note_counts + dev_note_counts))
    if all_counts:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        bins = list(range(min(all_counts), max(all_counts) + 2))
        ax.hist(
            train_note_counts,
            bins=bins,
            alpha=0.65,
            color="#4c78a8",
            label="Train",
            edgecolor="black",
        )
        ax.hist(
            dev_note_counts,
            bins=bins,
            alpha=0.65,
            color="#f58518",
            label="Dev",
            edgecolor="black",
        )
        ax.set_xlabel("Number of notes per patient")
        ax.set_ylabel("Number of patients")
        ax.set_title("Dataset notes-per-patient distribution")
        ax.legend()
        save_fig(fig, assets / "notes_per_patient_train_dev")
        plt.close(fig)

    # Condition / evidence / status statistics from train labels
    label_files = sorted(train_labels_dir.glob("patient_*.json"))
    if not label_files:
        return

    cond_per_patient: list[int] = []
    evidence_counts_per_condition: list[int] = []
    status_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    subcategory_counts: dict[str, int] = {}

    for lf in label_files:
        obj = _load_json(lf)
        conds = obj.get("conditions", []) or []
        cond_per_patient.append(len(conds))

        for c in conds:
            evidence = c.get("evidence", []) or []
            evidence_counts_per_condition.append(len(evidence))

            st = c.get("status")
            if st:
                status_counts[st] = status_counts.get(st, 0) + 1

            cat = c.get("category")
            sub = c.get("subcategory")
            if cat:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            if sub:
                subcategory_counts[sub] = subcategory_counts.get(sub, 0) + 1

    # Conditions per patient histogram
    if cond_per_patient:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        unique = sorted(set(cond_per_patient))
        bins = list(range(min(unique), max(unique) + 2))
        ax.hist(
            cond_per_patient,
            bins=bins,
            alpha=0.8,
            color="#4c78a8",
            edgecolor="black",
        )
        ax.set_xlabel("Number of conditions per patient (train labels)")
        ax.set_ylabel("Number of patients")
        ax.set_title("Label density (conditions per patient)")
        save_fig(fig, assets / "conditions_per_patient_train")
        plt.close(fig)

    # Status distribution
    if status_counts:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        statuses = sorted(status_counts.keys())
        vals = [status_counts[s] for s in statuses]
        palette = ["#59a14f", "#edc948", "#e15759", "#4c78a8", "#f58518"]
        ax.bar(statuses, vals, color=palette[: len(statuses)])
        ax.set_xlabel("Status")
        ax.set_ylabel("Number of conditions")
        ax.set_title("Status distribution (train labels)")
        save_fig(fig, assets / "status_distribution_train")
        plt.close(fig)

    # Category distribution
    if category_counts:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        cats_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        cat_names = [c for c, _ in cats_sorted]
        cat_vals = [v for _, v in cats_sorted]
        ax.barh(cat_names, cat_vals, color="#4c78a8")
        ax.invert_yaxis()
        ax.set_xlabel("Number of conditions")
        ax.set_title("Category distribution (train labels)")
        maxv = max(cat_vals) if cat_vals else 1
        for y, v in enumerate(cat_vals):
            ax.text(v + maxv * 0.01, y, str(v), va="center", fontsize=9)
        save_fig(fig, assets / "label_category_distribution_train")
        plt.close(fig)

    # Top subcategory distribution
    if subcategory_counts:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        subs_sorted = sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True)
        topn = 15
        subs_top = subs_sorted[:topn]
        sub_names = [s for s, _ in subs_top]
        sub_vals = [v for _, v in subs_top]
        ax.barh(sub_names[::-1], sub_vals[::-1], color="#f58518")
        ax.set_xlabel("Number of conditions")
        ax.set_title(f"Top-{topn} subcategories (train labels)")
        maxv = max(sub_vals) if sub_vals else 1
        for y, v in enumerate(sub_vals[::-1]):
            ax.text(v + maxv * 0.01, y, str(v), va="center", fontsize=9)
        save_fig(fig, assets / "label_subcategory_top15_train")
        plt.close(fig)

    # Evidence density proxy: evidence entries per condition
    if evidence_counts_per_condition:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        unique = sorted(set(evidence_counts_per_condition))
        bins = list(range(min(unique), max(unique) + 2))
        ax.hist(
            evidence_counts_per_condition,
            bins=bins,
            alpha=0.8,
            color="#e15759",
            edgecolor="black",
        )
        ax.set_xlabel("Number of evidence entries per condition (train)")
        ax.set_ylabel("Number of conditions")
        ax.set_title("Evidence density (train labels)")
        save_fig(fig, assets / "evidence_entries_histogram_train")
        plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets = repo_root / "report" / "assets"
    ensure_dir(assets)

    make_pipeline(assets)
    make_taxonomy_overview(assets, repo_root)
    make_metrics_template(assets)
    make_dataset_stats(assets, repo_root)

    print(f"Wrote figures to: {assets}")


if __name__ == "__main__":
    main()

