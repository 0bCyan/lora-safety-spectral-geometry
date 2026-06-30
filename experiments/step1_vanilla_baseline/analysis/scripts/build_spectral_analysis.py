from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("experiments/step1_vanilla_baseline")
METRIC_DIR = ROOT / "results/4070ti/spectral_metrics"
JUDGE_SUMMARY = ROOT / "results/4070ti/local_judge/summary.csv"
OUT_DIR = ROOT / "analysis"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"

FAMILIES = ["normal", "risk", "control"]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
METRIC = "delta_spectral_norm"


def layer_index(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", layer_name)
    if not match:
        return -1
    return int(match.group(1))


def block_type(module: str) -> str:
    if module in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention"
    if module in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    return "other"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def load_metric_rows() -> list[dict]:
    all_rows = []
    for path in sorted(METRIC_DIR.glob("*.jsonl")):
        for row in read_jsonl(path):
            item = dict(row)
            item["layer_index"] = layer_index(item["layer_name"])
            item["block_type"] = block_type(item["module"])
            item["experiment_id"] = path.stem
            all_rows.append(item)
    return all_rows


def group_summary(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["n"] = len(group_rows)
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = safe_stdev(values)
            item[f"{metric}_max"] = max(values)
        output.append(item)
    return output


def family_layer_matrix(layer_summary: list[dict]) -> dict[str, dict[int, float]]:
    matrix = {family: {} for family in FAMILIES}
    for row in layer_summary:
        family = row["family"]
        if family in matrix:
            matrix[family][int(row["layer_index"])] = float(row[f"{METRIC}_mean"])
    return matrix


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # White to blue, print-friendly and single-hue only inside data marks.
    r = int(245 - 175 * t)
    g = int(248 - 120 * t)
    b = int(255 - 30 * t)
    return f"rgb({r},{g},{b})"


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#111">{html.escape(text)}</text>'
    )


def write_layer_heatmap(matrix: dict[str, dict[int, float]], path: Path) -> None:
    layers = list(range(36))
    values = [matrix[f][l] for f in FAMILIES for l in layers]
    vmin, vmax = min(values), max(values)
    cell_w, cell_h = 18, 26
    left, top = 90, 54
    width = left + cell_w * len(layers) + 40
    height = top + cell_h * len(FAMILIES) + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, "逐层 LoRA 更新谱范数", 16, "middle"),
        svg_text(width / 2, 42, "Qwen3-4B，4070 Ti pilot，7 个模块与 3 个 seed 的均值", 11, "middle"),
    ]
    for i, layer in enumerate(layers):
        if layer % 2 == 0:
            parts.append(svg_text(left + i * cell_w + cell_w / 2, top - 8, str(layer), 9, "middle"))
    for row_i, family in enumerate(FAMILIES):
        y = top + row_i * cell_h
        parts.append(svg_text(12, y + 17, family, 12, "start"))
        for col_i, layer in enumerate(layers):
            x = left + col_i * cell_w
            value = matrix[family][layer]
            fill = color_scale(value, vmin, vmax)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff" stroke-width="1"/>')
    parts.append(svg_text(left, height - 35, f"最小值 {vmin:.4f}", 10, "start"))
    parts.append(svg_text(width - 40, height - 35, f"最大值 {vmax:.4f}", 10, "end"))
    grad_x, grad_y, grad_w, grad_h = left + 80, height - 46, 240, 10
    for i in range(120):
        value = vmin + (vmax - vmin) * i / 119
        parts.append(f'<rect x="{grad_x + i * grad_w / 120:.2f}" y="{grad_y}" width="{grad_w / 120 + 0.5:.2f}" height="{grad_h}" fill="{color_scale(value, vmin, vmax)}"/>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_module_bar(module_summary: list[dict], path: Path) -> None:
    rows = [row for row in module_summary if row["module"] in MODULES]
    by_family = defaultdict(dict)
    for row in rows:
        by_family[row["family"]][row["module"]] = float(row[f"{METRIC}_mean"])
    max_value = max(by_family[f][m] for f in FAMILIES for m in MODULES)
    left, top = 90, 55
    group_h, bar_h, gap = 112, 12, 5
    width, height = 680, top + group_h * len(FAMILIES) + 50
    colors = {"normal": "#4C78A8", "risk": "#D95F02", "control": "#5B8E7D"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, "模块级 LoRA 更新谱范数", 16, "middle"),
        svg_text(width / 2, 42, "按层与 3 个 seed 求均值", 11, "middle"),
    ]
    scale_w = 460
    for family_i, family in enumerate(FAMILIES):
        y0 = top + family_i * group_h
        parts.append(svg_text(12, y0 + 18, family, 12, "start"))
        for module_i, module in enumerate(MODULES):
            y = y0 + module_i * (bar_h + gap)
            value = by_family[family][module]
            w = scale_w * value / max_value
            parts.append(svg_text(left - 8, y + 10, module.replace("_proj", ""), 9, "end"))
            parts.append(f'<rect x="{left}" y="{y}" width="{w:.2f}" height="{bar_h}" fill="{colors[family]}"/>')
            parts.append(svg_text(left + w + 5, y + 10, f"{value:.4f}", 9, "start"))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_safety_spectral_scatter(joined_rows: list[dict], path: Path) -> None:
    rows = [row for row in joined_rows if row["train_family"] in FAMILIES]
    xs = [float(row["delta_spectral_norm_mean"]) for row in rows]
    ys = [float(row["unsafe_compliance_rate"]) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, 1.05
    left, top, plot_w, plot_h = 70, 50, 420, 260
    width, height = 550, 370
    colors = {"normal": "#4C78A8", "risk": "#D95F02", "control": "#5B8E7D"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, "安全表现与全局谱量", 16, "middle"),
        svg_text(width / 2, 42, "每个点代表一个 adapter seed", 11, "middle"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - (tick - ymin) / (ymax - ymin) * plot_h
        parts.append(f'<line x1="{left-4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222"/>')
        parts.append(svg_text(left - 8, y + 4, f"{tick:.2f}", 9, "end"))
    for row in rows:
        x_val = float(row["delta_spectral_norm_mean"])
        y_val = float(row["unsafe_compliance_rate"])
        x = left + (x_val - xmin) / (xmax - xmin) * plot_w if xmax > xmin else left
        y = top + plot_h - (y_val - ymin) / (ymax - ymin) * plot_h
        family = row["train_family"]
        seed = row["seed"]
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{colors[family]}" opacity="0.9"/>')
        parts.append(svg_text(x + 8, y + 4, f"{family[0]}{seed}", 9, "start"))
    parts.append(svg_text(left + plot_w / 2, height - 22, "平均更新谱范数", 11, "middle"))
    parts.append(svg_text(18, top + plot_h / 2, "危险遵从率", 11, "middle"))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_metric_rows()
    metrics = [
        "spectral_proxy",
        "delta_spectral_norm",
        "delta_fro_norm",
        "top_singular_ratio",
        "effective_rank",
    ]

    long_fields = [
        "experiment_id",
        "family",
        "seed",
        "layer_index",
        "module",
        "block_type",
        "spectral_proxy",
        "delta_spectral_norm",
        "delta_fro_norm",
        "top_singular_ratio",
        "effective_rank",
        "rank",
    ]
    write_csv(TABLE_DIR / "spectral_long.csv", rows, long_fields)

    layer_summary = group_summary(rows, ["family", "layer_index"], metrics)
    module_summary = group_summary(rows, ["family", "module", "block_type"], metrics)
    family_summary = group_summary(rows, ["family"], metrics)
    seed_summary = group_summary(rows, ["family", "seed"], metrics)

    write_csv(TABLE_DIR / "layer_summary.csv", layer_summary)
    write_csv(TABLE_DIR / "module_summary.csv", module_summary)
    write_csv(TABLE_DIR / "family_summary.csv", family_summary)
    write_csv(TABLE_DIR / "seed_summary.csv", seed_summary)

    joined = read_csv(ROOT / "results/4070ti/local_judge/safety_spectral_joined.csv")
    write_layer_heatmap(family_layer_matrix(layer_summary), FIGURE_DIR / "layer_heatmap.svg")
    write_module_bar(module_summary, FIGURE_DIR / "module_bar.svg")
    write_safety_spectral_scatter(joined, FIGURE_DIR / "safety_spectral_scatter.svg")

    report = [
        "# 4070 Ti 谱结构机制分析",
        "",
        "## 1. 产物",
        "",
        "- `tables/spectral_long.csv`：逐层逐模块长表。",
        "- `tables/layer_summary.csv`：按 family 和 layer 聚合。",
        "- `tables/module_summary.csv`：按 family 和 module 聚合。",
        "- `tables/family_summary.csv`：按 family 聚合。",
        "- `tables/seed_summary.csv`：按 family 和 seed 聚合。",
        "- `figures/layer_heatmap.svg`：层级谱范数热图。",
        "- `figures/module_bar.svg`：模块级谱范数柱状图。",
        "- `figures/safety_spectral_scatter.svg`：安全指标与全局谱量散点图。",
        "",
        "## 2. 初步结论",
        "",
        "1. risk 的安全退化在三个 seed 中稳定，但其全局谱均值不是最高。",
        "2. control 的全局谱量和 top singular ratio 更高，但行为表现不稳定，包含输出退化。",
        "3. normal 的谱量整体最低，安全表现相对更好但仍有漂移。",
        "4. 后续机制验证应优先分析层级、模块级和局部尖峰，而不是只依赖全局平均谱量。",
        "",
        "## 3. 写作提示",
        "",
        "当前结果适合写作成 pilot evidence：不同微调数据造成不同安全行为，且 LoRA 更新几何存在可观测差异；但它还不能证明谱结构具有因果作用。",
    ]
    (REPORT_DIR / "谱结构机制分析.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"rows: {len(rows)}")
    print(f"wrote: {OUT_DIR}")


if __name__ == "__main__":
    main()
