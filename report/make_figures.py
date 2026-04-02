"""
Generate all figures for the Clinical NLP Extraction report.

Produces publication-quality plots saved as both PNG (for README/Markdown)
and PDF (for LaTeX/Overleaf) under report/assets/.

Usage:
    python report/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Global styling — modern, publication-quality
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Color palette — curated for clinical/professional appearance
PALETTE = {
    "blue":     "#2563EB",
    "purple":   "#7C3AED",
    "teal":     "#0EA5E9",
    "green":    "#16A34A",
    "orange":   "#F97316",
    "red":      "#DC2626",
    "amber":    "#F59E0B",
    "slate":    "#475569",
    "gray":     "#94A3B8",
    "indigo":   "#4F46E5",
    "rose":     "#E11D48",
    "emerald":  "#059669",
}

# Gradient palette for bar charts
BAR_COLORS = ["#2563EB", "#7C3AED", "#0EA5E9", "#16A34A", "#F97316",
              "#DC2626", "#F59E0B", "#4F46E5", "#059669", "#E11D48",
              "#8B5CF6", "#06B6D4", "#84CC16"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  -> {out_base.with_suffix('.png').name}")


# ---------------------------------------------------------------------------
# Figure 1: Pipeline overview (professional architecture diagram)
# ---------------------------------------------------------------------------

def make_pipeline(assets: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.set_axis_off()

    # Title
    ax.text(0.5, 0.97, "Clinical NLP Extraction Pipeline",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color="#0F172A", transform=ax.transAxes)
    ax.text(0.5, 0.90, "Two-Pass LLM Architecture with Deterministic Evidence Hardening",
            ha="center", va="top", fontsize=11, color="#64748B",
            transform=ax.transAxes)

    stages = [
        {"title": "DATA INGESTION",
         "items": ["Patient notes\n(text_0 … text_N)", "Line-numbered\nindexing", "Taxonomy\nloading"],
         "accent": "#2563EB", "icon": "📥"},
        {"title": "PASS 1: EXTRACTION",
         "items": ["Per-note LLM\nextraction", "Evidence span\ncoercion", "Fuzzy taxonomy\nrecovery"],
         "accent": "#7C3AED", "icon": "🔍"},
        {"title": "PASS 2: CONSOLIDATION",
         "items": ["Cross-note\ndeduplication", "Status/onset\nresolution", "LLM-driven\nmerging"],
         "accent": "#16A34A", "icon": "🔗"},
        {"title": "PASS 3: HARDENING",
         "items": ["Deterministic\nevidence scan", "Fuzzy line\nmatching", "Deduplication\n& validation"],
         "accent": "#F97316", "icon": "🛡️"},
        {"title": "OUTPUT",
         "items": ["patient_XX.json", "Schema-validated", "Taxonomy-strict"],
         "accent": "#DC2626", "icon": "📄"},
    ]

    n = len(stages)
    box_w, box_h = 0.165, 0.55
    gap = 0.022
    total_w = n * box_w + (n - 1) * gap
    x_start = (1 - total_w) / 2
    y_base = 0.08

    for i, s in enumerate(stages):
        x = x_start + i * (box_w + gap)

        # Shadow
        shadow = FancyBboxPatch(
            (x + 0.003, y_base - 0.003), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0, facecolor="#E2E8F0", alpha=0.5,
        )
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (x, y_base), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5, edgecolor=s["accent"], facecolor="#FFFFFF",
        )
        ax.add_patch(box)

        # Accent header bar
        header = FancyBboxPatch(
            (x, y_base + box_h - 0.09), box_w, 0.09,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            linewidth=0, facecolor=s["accent"], alpha=0.9,
        )
        ax.add_patch(header)

        # Title text
        ax.text(x + box_w / 2, y_base + box_h - 0.045, s["title"],
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white", family="sans-serif")

        # Items
        for j, item in enumerate(s["items"]):
            y_item = y_base + box_h - 0.16 - j * 0.15
            ax.text(x + box_w / 2, y_item, item,
                    ha="center", va="center", fontsize=8.5,
                    color="#334155", linespacing=1.2)

        # Arrow to next
        if i < n - 1:
            x_end = x + box_w
            x_next = x_start + (i + 1) * (box_w + gap)
            ax.annotate("", xy=(x_next - 0.002, y_base + box_h / 2),
                        xytext=(x_end + 0.002, y_base + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                        color=s["accent"], mutation_scale=18))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig(fig, assets / "pipeline")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Taxonomy overview
# ---------------------------------------------------------------------------

def make_taxonomy_overview(assets: Path, repo_root: Path) -> None:
    taxonomy_path = repo_root / "Data" / "taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    cats = taxonomy["condition_categories"]
    cat_names = list(cats.keys())
    sub_counts = [len(cats[k].get("subcategories", {})) for k in cat_names]

    order = np.argsort(sub_counts)[::-1]
    cat_names_ord = [cat_names[i].replace("_", " ").title() for i in order]
    sub_counts_ord = [sub_counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(cat_names_ord))]
    bars = ax.barh(cat_names_ord, sub_counts_ord, color=colors, height=0.65,
                   edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Subcategories", fontweight="bold")
    ax.set_title("Clinical Taxonomy: Subcategories per Category")
    ax.set_xlim(0, max(sub_counts_ord) + 1.5)

    for bar, v in zip(bars, sub_counts_ord):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=10, fontweight="bold", color="#334155")

    ax.grid(axis="x", alpha=0.2, linestyle="--")
    save_fig(fig, assets / "taxonomy_overview")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Notes per patient distribution
# ---------------------------------------------------------------------------

def _patient_ids_from_dir(dirpath: Path) -> list[str]:
    return sorted([p.name for p in dirpath.glob("patient_*") if p.is_dir()])


def _count_note_files(patient_dir: Path) -> int:
    return len(list(patient_dir.glob("text_*.md")))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_dataset_stats(assets: Path, repo_root: Path) -> None:
    train_dir = repo_root / "Data" / "train"
    dev_dir = repo_root / "Data" / "dev"
    train_labels_dir = train_dir / "labels"

    # --- Notes per patient ---
    def note_counts(d: Path) -> list[int]:
        return [_count_note_files(d / pid) for pid in _patient_ids_from_dir(d)]

    train_nc = note_counts(train_dir)
    dev_nc = note_counts(dev_dir)
    all_nc = sorted(set(train_nc + dev_nc))

    if all_nc:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = list(range(min(all_nc), max(all_nc) + 2))
        ax.hist(train_nc, bins=bins, alpha=0.75, color=PALETTE["blue"],
                label=f"Train (n={len(train_nc)})", edgecolor="white", linewidth=1.2)
        ax.hist(dev_nc, bins=bins, alpha=0.75, color=PALETTE["orange"],
                label=f"Dev (n={len(dev_nc)})", edgecolor="white", linewidth=1.2)
        ax.set_xlabel("Number of Notes per Patient", fontweight="bold")
        ax.set_ylabel("Number of Patients", fontweight="bold")
        ax.set_title("Notes-per-Patient Distribution")
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.set_xticks(range(min(all_nc), max(all_nc) + 1))
        save_fig(fig, assets / "notes_per_patient_train_dev")
        plt.close(fig)

    # --- Load training labels ---
    label_files = sorted(train_labels_dir.glob("patient_*.json"))
    if not label_files:
        return

    cond_per_patient: list[int] = []
    evidence_counts: list[int] = []
    status_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    subcategory_counts: dict[str, int] = {}
    onset_present = 0
    onset_null = 0

    for lf in label_files:
        obj = _load_json(lf)
        conds = obj.get("conditions", []) or []
        cond_per_patient.append(len(conds))
        for c in conds:
            evidence_counts.append(len(c.get("evidence", []) or []))
            st = c.get("status")
            if st:
                status_counts[st] = status_counts.get(st, 0) + 1
            cat = c.get("category")
            sub = c.get("subcategory")
            if cat:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            if sub:
                subcategory_counts[sub] = subcategory_counts.get(sub, 0) + 1
            if c.get("onset"):
                onset_present += 1
            else:
                onset_null += 1

    # --- Conditions per patient ---
    if cond_per_patient:
        fig, ax = plt.subplots(figsize=(8, 5))
        patient_labels = [lf.stem.replace("patient_", "P") for lf in label_files]
        colors_bar = [PALETTE["blue"], PALETTE["purple"], PALETTE["green"], PALETTE["orange"]]
        bars = ax.bar(patient_labels, cond_per_patient,
                      color=colors_bar[:len(cond_per_patient)],
                      width=0.55, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, cond_per_patient):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                    str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xlabel("Patient", fontweight="bold")
        ax.set_ylabel("Number of Conditions", fontweight="bold")
        ax.set_title("Conditions per Patient (Training Labels)")
        ax.set_ylim(0, max(cond_per_patient) + 4)
        avg = sum(cond_per_patient) / len(cond_per_patient)
        ax.axhline(y=avg, color=PALETTE["red"], linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(len(cond_per_patient) - 0.5, avg + 0.5, f"avg = {avg:.1f}",
                color=PALETTE["red"], fontsize=10, ha="right")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        save_fig(fig, assets / "conditions_per_patient_train")
        plt.close(fig)

    # --- Status distribution (pie chart) ---
    if status_counts:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        status_order = ["active", "resolved", "suspected"]
        status_colors = {"active": PALETTE["green"], "resolved": PALETTE["blue"],
                         "suspected": PALETTE["amber"]}
        labels = [s for s in status_order if s in status_counts]
        sizes = [status_counts[s] for s in labels]
        colors = [status_colors.get(s, PALETTE["gray"]) for s in labels]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=[s.capitalize() for s in labels],
            colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
            textprops=dict(fontsize=12, fontweight="bold"),
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_color("white")
            t.set_fontweight("bold")
        ax.set_title("Condition Status Distribution (Training Labels)")

        # Add count annotations
        legend_labels = [f"{s.capitalize()}: {status_counts[s]}" for s in labels]
        ax.legend(wedges, legend_labels, loc="lower center", ncol=3,
                  fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.05))
        save_fig(fig, assets / "status_distribution_train")
        plt.close(fig)

    # --- Category distribution ---
    if category_counts:
        fig, ax = plt.subplots(figsize=(10, 6))
        cats_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        cat_names = [c.replace("_", " ").title() for c, _ in cats_sorted]
        cat_vals = [v for _, v in cats_sorted]
        colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(cat_names))]

        bars = ax.barh(cat_names, cat_vals, color=colors, height=0.65,
                       edgecolor="white", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Conditions", fontweight="bold")
        ax.set_title("Category Distribution (Training Labels)")
        maxv = max(cat_vals) if cat_vals else 1
        ax.set_xlim(0, maxv * 1.15)
        for bar, v in zip(bars, cat_vals):
            ax.text(v + maxv * 0.015, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=10, fontweight="bold", color="#334155")
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        save_fig(fig, assets / "label_category_distribution_train")
        plt.close(fig)

    # --- Top subcategory distribution ---
    if subcategory_counts:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        subs_sorted = sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True)
        topn = min(15, len(subs_sorted))
        subs_top = subs_sorted[:topn]
        sub_names = [s.replace("_", " ").title() for s, _ in subs_top][::-1]
        sub_vals = [v for _, v in subs_top][::-1]

        colors = [PALETTE["orange"]] * len(sub_names)
        bars = ax.barh(sub_names, sub_vals, color=colors, height=0.6,
                       edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Number of Conditions", fontweight="bold")
        ax.set_title(f"Top-{topn} Subcategories (Training Labels)")
        maxv = max(sub_vals) if sub_vals else 1
        ax.set_xlim(0, maxv * 1.15)
        for bar, v in zip(bars, sub_vals):
            ax.text(v + maxv * 0.015, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=10, fontweight="bold", color="#334155")
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        save_fig(fig, assets / "label_subcategory_top15_train")
        plt.close(fig)

    # --- Evidence density histogram ---
    if evidence_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        unique = sorted(set(evidence_counts))
        bins = list(range(min(unique), max(unique) + 2))
        ax.hist(evidence_counts, bins=bins, alpha=0.85, color=PALETTE["purple"],
                edgecolor="white", linewidth=1.2)
        ax.set_xlabel("Evidence Entries per Condition", fontweight="bold")
        ax.set_ylabel("Number of Conditions", fontweight="bold")
        ax.set_title("Evidence Density Distribution (Training Labels)")
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        avg_ev = sum(evidence_counts) / len(evidence_counts)
        ax.axvline(x=avg_ev, color=PALETTE["red"], linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(avg_ev + 0.3, ax.get_ylim()[1] * 0.9, f"avg = {avg_ev:.1f}",
                color=PALETTE["red"], fontsize=10)
        save_fig(fig, assets / "evidence_entries_histogram_train")
        plt.close(fig)

    # --- Onset coverage (donut) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    onset_labels = ["Has Date", "Unknown (null)"]
    onset_vals = [onset_present, onset_null]
    onset_colors = [PALETTE["green"], PALETTE["gray"]]
    if onset_null == 0:
        onset_labels = ["Has Date"]
        onset_vals = [onset_present]
        onset_colors = [PALETTE["green"]]

    wedges, texts, autotexts = ax.pie(
        onset_vals, labels=onset_labels, colors=onset_colors,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2.5),
        textprops=dict(fontsize=11, fontweight="bold"),
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("Onset Date Coverage (Training Labels)")
    save_fig(fig, assets / "onset_coverage_train")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Metrics template
# ---------------------------------------------------------------------------

def make_metrics_template(assets: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    metrics = {
        "Condition\nPrecision": 0.0,
        "Condition\nRecall": 0.0,
        "Condition\nF1": 0.0,
        "Status\nAccuracy": 0.0,
        "Onset\n(Exact)": 0.0,
        "Onset\n(Partial)": 0.0,
        "Evidence\nRecall": 0.0,
        "Evidence\nPrecision": 0.0,
    }
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["indigo"],
              PALETTE["green"], PALETTE["amber"], PALETTE["orange"],
              PALETTE["purple"], PALETTE["rose"]]

    bars = ax.bar(range(len(keys)), vals, color=colors[:len(keys)],
                  width=0.6, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Evaluation Metrics Dashboard (Template — Populate with Results)")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.text(0.5, -0.18,
            "Run: python -m Clinical_Nlp_Extraction.train --results-json results.json",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color=PALETTE["slate"], style="italic")

    save_fig(fig, assets / "metrics_plot")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: Evidence quality heatmap concept
# ---------------------------------------------------------------------------

def make_evidence_heatmap(assets: Path, repo_root: Path) -> None:
    """Create a heatmap showing evidence coverage across notes per patient."""
    train_labels_dir = repo_root / "Data" / "train" / "labels"
    label_files = sorted(train_labels_dir.glob("patient_*.json"))
    if not label_files:
        return

    fig, axes = plt.subplots(1, len(label_files), figsize=(14, 5),
                             gridspec_kw={"wspace": 0.4})
    if len(label_files) == 1:
        axes = [axes]

    for ax, lf in zip(axes, label_files):
        obj = _load_json(lf)
        pid = obj["patient_id"]
        conds = obj.get("conditions", [])

        # Get all note_ids and condition names
        all_note_ids = sorted(set(
            ev["note_id"] for c in conds for ev in c.get("evidence", [])
        ))
        cond_names = [c["condition_name"][:25] + ("..." if len(c["condition_name"]) > 25 else "")
                      for c in conds]

        if not all_note_ids or not cond_names:
            continue

        # Build matrix
        matrix = np.zeros((len(cond_names), len(all_note_ids)))
        for i, c in enumerate(conds):
            for ev in c.get("evidence", []):
                if ev["note_id"] in all_note_ids:
                    j = all_note_ids.index(ev["note_id"])
                    matrix[i, j] += 1

        # Normalize to presence/absence for clarity
        matrix_binary = (matrix > 0).astype(float)

        im = ax.imshow(matrix_binary, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_note_ids)))
        ax.set_xticklabels([n.replace("text_", "t") for n in all_note_ids],
                           fontsize=7, rotation=45)
        ax.set_yticks(range(len(cond_names)))
        ax.set_yticklabels(cond_names, fontsize=6)
        ax.set_title(pid.replace("patient_", "Patient "), fontsize=10, fontweight="bold")
        ax.set_xlabel("Note", fontsize=8)

    fig.suptitle("Evidence Coverage Heatmap: Conditions × Notes", fontsize=13,
                 fontweight="bold", y=1.02)
    save_fig(fig, assets / "evidence_heatmap_train")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets = repo_root / "report" / "assets"
    ensure_dir(assets)

    print("Generating figures...")
    make_pipeline(assets)
    make_taxonomy_overview(assets, repo_root)
    make_dataset_stats(assets, repo_root)
    make_metrics_template(assets)
    make_evidence_heatmap(assets, repo_root)

    print(f"\nAll figures written to: {assets}")


if __name__ == "__main__":
    main()
