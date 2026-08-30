from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
DEFAULT_OUTPUT = PAPER_ROOT / "figures" / "available"
HISTORICAL = PAPER_ROOT / "configs" / "historical_figure_data.json"
SCHEDULE_CONFIG = PAPER_ROOT / "configs" / "qwen3_4b_equal_dose_temporal_schedules.json"

# Color-blind friendly, muted palette. These figures are explanatory assets, not
# inferential result figures; the same semantic color is reused consistently.
COLORS = {
    "blue": "#2F6690",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "green": "#4C956C",
    "purple": "#6D597A",
    "red": "#C44536",
    "ink": "#243447",
    "muted": "#667085",
    "line": "#CBD5E1",
    "panel": "#F8FAFC",
    "pending": "#F1F5F9",
    "white": "#FFFFFF",
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": COLORS["line"],
        "axes.linewidth": 0.8,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "figure.facecolor": COLORS["white"],
        "axes.facecolor": COLORS["white"],
        "svg.hashsalt": "paper-a-strong-mechanism-v1",
        "svg.fonttype": "none",
    }
)


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
        )
        fig.tight_layout()
    for fmt in formats:
        kwargs = {"bbox_inches": "tight", "metadata": {"Date": None}}
        if fmt.lower() == "png":
            kwargs["dpi"] = dpi
        fig.savefig(output_dir / f"{stem}.{fmt}", **kwargs)
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    title_color: str | None = None,
    body_color: str | None = None,
    linewidth: float = 1.5,
    fontsize: float = 9.4,
) -> None:
    edge = edgecolor or facecolor
    title_c = title_color or COLORS["ink"]
    body_c = body_color or COLORS["ink"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + 0.035 * width, y + height - 0.22 * height, title, weight="bold", color=title_c, va="top")
    ax.text(
        x + 0.035 * width,
        y + height - 0.43 * height,
        body,
        color=body_c,
        va="top",
        fontsize=fontsize,
        linespacing=1.25,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#94A3B8") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def figure_paper_story(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.965, "Paper A: from diagnostic structure to a falsifiable persistence mechanism", fontsize=16, weight="bold")
    ax.text(
        0.01,
        0.915,
        "Mechanism and measurement study; no candidate is presented as a validated new method.",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    w, h = 0.42, 0.255
    positions = [(0.04, 0.58), (0.54, 0.58), (0.04, 0.255), (0.54, 0.255)]
    boxes = [
        (
            "1. Diagnostic geometry",
            "Risk LoRA shows reproducible\noutput-side structure.\n\nSimple spectral and post-hoc controls\ndo not recover behavior.",
            "#EAF2F8",
            COLORS["blue"],
        ),
        (
            "2. Behavior-aligned signal",
            "Sequence-level safety replay\nchanges unsafe behavior.\n\nScalar margins and tested local\nproxies are behaviorally insufficient.",
            "#E8F6F3",
            COLORS["teal"],
        ),
        (
            "3. Persistence–quality tension",
            "Late safety contribution may\ncontrol unsafe rebound.\n\nPersistent supervision can narrow\nprompt coverage and response quality.",
            "#FEF3E8",
            COLORS["orange"],
        ),
        (
            "4. Confirmatory mechanism test",
            "Equal cumulative safety dose;\nonly temporal placement changes.\n\nThree seeds, prompt-paired labels,\nhuman validity, second setting,\nand one sealed confirmation panel.",
            "#F1ECF4",
            COLORS["purple"],
        ),
    ]
    for position, (title, body, face, edge) in zip(positions, boxes):
        _box(ax, position, w, h, title, body, facecolor=face, edgecolor=edge, fontsize=9.2)

    _arrow(ax, (0.465, 0.71), (0.535, 0.71))
    _arrow(ax, (0.75, 0.575), (0.25, 0.515))
    _arrow(ax, (0.465, 0.385), (0.535, 0.385))

    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.075),
            0.92,
            0.10,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor=COLORS["pending"],
            edgecolor=COLORS["line"],
            linewidth=1.1,
        )
    )
    ax.text(0.065, 0.125, "Current evidence state", weight="bold", va="center")
    ax.text(
        0.255,
        0.125,
        "Historical mechanism evidence exists; formal equal-dose results, final blinded quality measurement, human validity, second-setting replication, and confirmation remain pending.",
        va="center",
        fontsize=8.9,
        color=COLORS["muted"],
        wrap=True,
    )
    _save(fig, output_dir, "fig00_paper_story", formats, dpi)

def _schedule_values(config: dict, arm_name: str, steps: np.ndarray) -> np.ndarray:
    arm = config["intervention_arms"][arm_name]
    values = np.zeros_like(steps, dtype=float)
    for window in arm["anchor_schedule"]:
        start = int(window["start_step"])
        end = int(window["end_step"])
        mask = (steps >= start) & (steps < end)
        values[mask] = float(window["anchor_mix"])
    return values


def figure_equal_dose_design(output_dir: Path, formats: list[str], dpi: int) -> None:
    config = json.loads(SCHEDULE_CONFIG.read_text(encoding="utf-8"))
    steps = np.arange(int(config["paper_design"]["total_steps"]))
    arms = [
        ("front_loaded_equal_dose", "Front-loaded", COLORS["orange"]),
        ("uniform_equal_dose", "Uniform", COLORS["blue"]),
        ("back_loaded_equal_dose", "Back-loaded", COLORS["teal"]),
    ]

    fig = plt.figure(figsize=(11.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for arm_name, label, color in arms:
        values = _schedule_values(config, arm_name, steps)
        ax.step(steps, values, where="post", linewidth=2.3, label=label, color=color)
    ax.axvline(32, linestyle="--", linewidth=1.1, color="#94A3B8")
    ax.text(31.2, 0.208, "midpoint", ha="right", va="bottom", fontsize=8.5, color=COLORS["muted"])
    ax.set_xlim(0, 63)
    ax.set_ylim(0.105, 0.215)
    ax.set_xticks([0, 6, 16, 32, 48, 63])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Safety replay weight")
    ax.set_title("A. Same cumulative dose, different temporal placement", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)

    ax2 = fig.add_subplot(gs[0, 1])
    cumulative = []
    risk = []
    labels = []
    colors = []
    for arm_name, label, color in arms:
        values = _schedule_values(config, arm_name, steps)
        cumulative.append(values.sum())
        risk.append((1.0 - values).sum())
        labels.append(label)
        colors.append(color)
    x = np.arange(len(labels))
    risk_bars = ax2.bar(x, risk, color="#D7DEE8", edgecolor="#B6C2D1", width=0.62, label="Risk coefficient")
    safety_bars = ax2.bar(x, cumulative, bottom=risk, color=colors, width=0.62, label="Safety coefficient")
    ax2.set_ylim(0, 68)
    ax2.set_xticks(x, labels, rotation=18, ha="right")
    ax2.set_ylabel("Cumulative loss coefficient")
    ax2.set_title("B. Dose equality is exact", loc="left", weight="bold")
    ax2.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    for rb, sb, safety_total, risk_total in zip(risk_bars, safety_bars, cumulative, risk):
        center = rb.get_x() + rb.get_width() / 2
        ax2.text(center, risk_total / 2, f"risk\n{risk_total:.2f}", ha="center", va="center", fontsize=8.5, color=COLORS["ink"], weight="bold")
        ax2.text(center, risk_total + safety_total / 2, f"safety\n{safety_total:.2f}", ha="center", va="center", fontsize=8.5, color=COLORS["white"], weight="bold")
        ax2.text(center, 64.8, "total 64", ha="center", fontsize=8.3, color=COLORS["muted"])
    ax2.text(
        0.5,
        -0.24,
        "All arms use identical cumulative coefficients; only when safety pressure is applied differs.",
        transform=ax2.transAxes,
        ha="center",
        fontsize=8.9,
        color=COLORS["muted"],
    )

    fig.suptitle("Equal-dose temporal-distribution design", x=0.02, ha="left", fontsize=16, weight="bold")
    _save(fig, output_dir, "fig01_equal_dose_design", formats, dpi)


def figure_evidence_pipeline(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.95, "Prospective evidence pipeline and stopping gates", fontsize=16, weight="bold")
    ax.text(
        0.01,
        0.89,
        "Green = infrastructure frozen or available; gray = scientific evidence not yet produced.",
        fontsize=10,
        color=COLORS["muted"],
    )

    stages = [
        ("M0", "Claims, protocols,\ncode and LaTeX", True),
        ("M1", "Pinned runtime,\ndry-runs and smoke", False),
        ("M2", "15 formal runs +\n45 downstream jobs", False),
        ("M3", "Blind safety, quality\nand utility labels", False),
        ("M4", "Independent human\nvalidity subset", False),
        ("M5", "Second model or\nsecond risk setting", False),
        ("M6", "Analysis freeze + one\nsealed confirmation", False),
        ("M7", "Release validator +\nfinal PDF", False),
    ]
    x_positions = np.linspace(0.02, 0.88, len(stages))
    y = 0.55
    w, h = 0.105, 0.22
    for idx, ((milestone, label, done), x) in enumerate(zip(stages, x_positions)):
        face = "#E9F5EE" if done else COLORS["pending"]
        edge = COLORS["green"] if done else "#94A3B8"
        _box(ax, (x, y), w, h, milestone, label, facecolor=face, edgecolor=edge, fontsize=8.3)
        status = "complete" if done else "pending"
        ax.text(x + w / 2, y - 0.045, status, ha="center", fontsize=8.4, color=edge, weight="bold")
        if idx < len(stages) - 1:
            _arrow(ax, (x + w + 0.002, y + h / 2), (x_positions[idx + 1] - 0.006, y + h / 2))

    gate_y = 0.22
    gate_items = [
        (0.18, "No result-dependent\nschedule changes"),
        (0.40, "No heuristic labels\nas primary evidence"),
        (0.62, "No seed × prompt\npseudoreplication"),
        (0.84, "No confirmation access\nbefore analysis freeze"),
    ]
    for x, text in gate_items:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.10, gate_y),
                0.20,
                0.13,
                boxstyle="round,pad=0.01,rounding_size=0.018",
                facecolor="#FFF7ED",
                edgecolor=COLORS["gold"],
                linewidth=1.2,
            )
        )
        ax.text(x, gate_y + 0.065, text, ha="center", va="center", fontsize=8.7, linespacing=1.2)

    ax.text(0.01, 0.08, "Release rule:", weight="bold")
    ax.text(
        0.11,
        0.08,
        "missing human validity, second-setting replication, confirmation, final figures/tables, or unresolved PENDING EVIDENCE blocks submission.",
        color=COLORS["muted"],
        fontsize=9.3,
    )
    _save(fig, output_dir, "fig02_evidence_pipeline", formats, dpi)


def figure_data_boundaries(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.95, "Data roles and leakage boundaries", fontsize=16, weight="bold")
    ax.text(
        0.01,
        0.89,
        "Training targets, development evaluation, confirmation, and the global heldout retain different roles.",
        fontsize=10,
        color=COLORS["muted"],
    )

    _box(
        ax,
        (0.03, 0.48),
        0.21,
        0.30,
        "Safety training pool",
        "40 prompt IDs\n80 frozen natural targets\n\nUsed only for sequence-level\nsafety replay and exposure audits.",
        facecolor="#E8F6F3",
        edgecolor=COLORS["teal"],
        fontsize=8.9,
    )
    _box(
        ax,
        (0.29, 0.48),
        0.21,
        0.30,
        "Core100 development",
        "100 safety prompts\n\nUsed for trajectory analysis,\nmechanism development and\ncandidate comparison.",
        facecolor="#EAF2F8",
        edgecolor=COLORS["blue"],
        fontsize=8.9,
    )
    _box(
        ax,
        (0.55, 0.48),
        0.20,
        0.30,
        "Paper-A confirmation",
        "New prompt-disjoint panel\nSize frozen prospectively\n\nOpened exactly once after\nanalysis and decision freeze.",
        facecolor="#FEF3E8",
        edgecolor=COLORS["orange"],
        fontsize=8.9,
    )
    _box(
        ax,
        (0.80, 0.48),
        0.17,
        0.30,
        "Global heldout",
        "120 rows\n\nRemains reserved for the\nfuture method paper unless a\nseparate decision changes it.",
        facecolor="#F1ECF4",
        edgecolor=COLORS["purple"],
        fontsize=8.7,
    )

    for x in [0.255, 0.515, 0.765]:
        ax.plot([x, x], [0.45, 0.82], color="#CBD5E1", linestyle="--", linewidth=1.2)
        ax.text(x, 0.425, "no prompt reuse", ha="center", va="top", fontsize=7.8, color=COLORS["muted"])

    ax.add_patch(
        FancyBboxPatch(
            (0.16, 0.10),
            0.68,
            0.18,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=COLORS["panel"],
            edgecolor=COLORS["line"],
            linewidth=1.2,
        )
    )
    ax.text(0.19, 0.225, "Separate benign utility stream", weight="bold")
    ax.text(
        0.19,
        0.175,
        "Utility100 is evaluated independently from the safety–quality frontier; utility is never hidden inside a composite safety score.",
        fontsize=9,
        color=COLORS["muted"],
    )
    ax.text(
        0.19,
        0.13,
        "Primary output tables contain source hashes, opaque review IDs, arm/seed/checkpoint metadata, and prompt-paired joins.",
        fontsize=9,
        color=COLORS["muted"],
    )
    _save(fig, output_dir, "fig03_data_boundaries", formats, dpi)


def figure_historical_subspace(output_dir: Path, formats: list[str], dpi: int, data: dict) -> None:
    rows = data["subspace_top_direction_cosine"]
    labels = [row["comparison"] for row in rows]
    values = [row["value"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(labels, values, color=[COLORS["blue"], "#8DB3CF", "#B7CCE0"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Top-direction cosine")
    ax.set_title("Historical: risk adapters share an output-side signature", loc="left", weight="bold")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", weight="bold")
    ax.text(
        0.0,
        -0.26,
        "Frozen Step 1 evidence. Diagnostic association only; not a verified causal control direction.",
        transform=ax.transAxes,
        fontsize=8.7,
        color=COLORS["muted"],
    )
    _save(fig, output_dir, "fig10_historical_subspace_signature", formats, dpi)


def figure_historical_posthoc(output_dir: Path, formats: list[str], dpi: int, data: dict) -> None:
    rows = data["posthoc_best_unsafe"]
    labels = [row["intervention"] for row in rows]
    values = [row["unsafe"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(labels, values, color=[COLORS["ink"]] + [COLORS["orange"]] * (len(labels) - 1))
    ax.set_ylim(0.94, 1.01)
    ax.set_ylabel("Unsafe rate")
    ax.set_title("Historical: tested post-hoc controls do not recover safety", loc="left", weight="bold")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.tick_params(axis="x", rotation=22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=8.5)
    ax.text(
        0.0,
        -0.33,
        "Best observed reductions remain close to the unsafe baseline. This bounds the tested interventions; it is not a universal impossibility claim.",
        transform=ax.transAxes,
        fontsize=8.7,
        color=COLORS["muted"],
    )
    _save(fig, output_dir, "fig11_historical_posthoc_control_failure", formats, dpi)


def figure_historical_sequence_anchor(output_dir: Path, formats: list[str], dpi: int, data: dict) -> None:
    rows = data["step16_sequence_anchor_unsafe"]
    seeds = [str(row["seed"]) for row in rows]
    values = [row["unsafe"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(seeds, values, color=COLORS["teal"], width=0.58)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("Risk seed")
    ax.set_ylabel("Unsafe rate at step 16")
    ax.set_title("Historical: sequence-level safety supervision is behaviorally active", loc="left", weight="bold")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", weight="bold")
    ax.text(
        0.0,
        -0.23,
        "Three-seed early-checkpoint result. The final persistence and quality claims still require the registered full replication.",
        transform=ax.transAxes,
        fontsize=8.7,
        color=COLORS["muted"],
    )
    _save(fig, output_dir, "fig12_historical_sequence_anchor_multiseed", formats, dpi)


def figure_historical_persistence_context(output_dir: Path, formats: list[str], dpi: int, data: dict) -> None:
    rows = data["historical_seed1_step64"]
    labels = [row["arm"] for row in rows]
    values = [row["unsafe"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bars = ax.bar(labels, values, color=[COLORS["orange"], COLORS["teal"]], width=0.58)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Blind-adjudicated unsafe rate")
    ax.set_title("Historical seed 1: sustained replay reduces late unsafe rebound", loc="left", weight="bold")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", weight="bold")
    ax.annotate(
        "−0.28",
        xy=(1, values[1]),
        xytext=(0.5, 0.51),
        textcoords="data",
        ha="center",
        fontsize=11,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "linewidth": 1.2},
    )
    ax.text(
        0.0,
        -0.27,
        "Context-only historical result. It will be rerun for seeds 1–3 under one pinned model snapshot and environment.",
        transform=ax.transAxes,
        fontsize=8.7,
        color=COLORS["muted"],
    )
    _save(fig, output_dir, "fig13_historical_seed1_persistence_context", formats, dpi)


def figure_completion_gates(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-0.45, 0.94, "Paper A completion gates", fontsize=16, weight="bold")
    labels = [
        ("M0", "Workspace\n& protocols", True),
        ("M1", "Runtime\npreflight", False),
        ("M2", "Formal\nrun matrix", False),
        ("M3", "Blind\nremeasurement", False),
        ("M4", "Human\nvalidity", False),
        ("M5", "Second\nsetting", False),
        ("M6", "Sealed\nconfirmation", False),
        ("M7", "Release\nverification", False),
    ]
    ax.plot(range(8), [0.52] * 8, color="#CBD5E1", linewidth=3, zorder=1)
    for x, (mid, text, done) in enumerate(labels):
        face = COLORS["green"] if done else COLORS["white"]
        edge = COLORS["green"] if done else "#94A3B8"
        text_color = COLORS["white"] if done else COLORS["ink"]
        ax.scatter([x], [0.52], s=1150, facecolor=face, edgecolor=edge, linewidth=2.2, zorder=3)
        ax.text(x, 0.55, mid, ha="center", va="center", weight="bold", color=text_color, fontsize=11, zorder=4)
        ax.text(x, 0.27, text, ha="center", va="center", fontsize=9, linespacing=1.2)
        ax.text(x, 0.12, "complete" if done else "pending", ha="center", fontsize=8.5, color=edge, weight="bold")
    ax.text(
        -0.45,
        0.02,
        "The branch can be operationally complete while the scientific paper remains non-submittable. M7 is the only submission-ready state.",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    _save(fig, output_dir, "fig20_completion_gates", formats, dpi)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render all Paper A figures that do not depend on missing formal results.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--formats",
        default="svg",
        help="Comma-separated output formats. Recommended: svg for Git; svg,pdf,png for local export.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unsupported = set(formats) - {"svg", "pdf", "png"}
    if unsupported:
        raise ValueError(f"Unsupported formats: {sorted(unsupported)}")

    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    figure_paper_story(args.output_dir, formats, args.dpi)
    figure_equal_dose_design(args.output_dir, formats, args.dpi)
    figure_evidence_pipeline(args.output_dir, formats, args.dpi)
    figure_data_boundaries(args.output_dir, formats, args.dpi)
    figure_historical_subspace(args.output_dir, formats, args.dpi, historical)
    figure_historical_posthoc(args.output_dir, formats, args.dpi, historical)
    figure_historical_sequence_anchor(args.output_dir, formats, args.dpi, historical)
    figure_historical_persistence_context(args.output_dir, formats, args.dpi, historical)
    figure_completion_gates(args.output_dir, formats, args.dpi)

    expected = {
        "fig00_paper_story",
        "fig01_equal_dose_design",
        "fig02_evidence_pipeline",
        "fig03_data_boundaries",
        "fig10_historical_subspace_signature",
        "fig11_historical_posthoc_control_failure",
        "fig12_historical_sequence_anchor_multiseed",
        "fig13_historical_seed1_persistence_context",
        "fig20_completion_gates",
    }
    missing = [
        f"{stem}.{fmt}"
        for stem in sorted(expected)
        for fmt in formats
        if not (args.output_dir / f"{stem}.{fmt}").exists()
    ]
    if missing:
        raise RuntimeError(f"Figure generation incomplete: {missing}")
    print(f"wrote {len(expected) * len(formats)} available figure files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
