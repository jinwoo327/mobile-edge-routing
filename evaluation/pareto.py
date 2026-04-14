from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    maximize: list[str],
    minimize: list[str],
) -> bool:
    better_or_equal = True
    strictly_better = False

    for key in maximize:
        if left[key] < right[key]:
            better_or_equal = False
            break
        if left[key] > right[key]:
            strictly_better = True

    if not better_or_equal:
        return False

    for key in minimize:
        if left[key] > right[key]:
            return False
        if left[key] < right[key]:
            strictly_better = True

    return strictly_better


def pareto_frontier(
    rows: list[dict[str, Any]],
    maximize: list[str],
    minimize: list[str],
) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            if dominates(other, candidate, maximize=maximize, minimize=minimize):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    return sorted(frontier, key=lambda row: (row[minimize[0]], -row[maximize[0]]))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(
    rows: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    output_path: str | Path,
    x_key: str = "mean_latency_ms",
    y_key: str = "accuracy",
    color_key: str = "edge_rate",
    label_key: str = "threshold",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for Pareto plotting. Install it with: pip install matplotlib"
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        [row[x_key] for row in rows],
        [row[y_key] for row in rows],
        c=[row[color_key] for row in rows],
        cmap="viridis",
        s=60,
        alpha=0.8,
        label="Sweep Runs",
    )

    frontier_sorted = sorted(frontier, key=lambda row: row[x_key])
    ax.plot(
        [row[x_key] for row in frontier_sorted],
        [row[y_key] for row in frontier_sorted],
        color="black",
        linewidth=1.5,
        label="Pareto Frontier",
    )
    ax.scatter(
        [row[x_key] for row in frontier_sorted],
        [row[y_key] for row in frontier_sorted],
        color="crimson",
        s=70,
        zorder=3,
    )

    for row in frontier_sorted:
        ax.annotate(
            f"{row[label_key]:.2f}",
            (row[x_key], row[y_key]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontier: Accuracy vs Latency vs Edge Rate")
    ax.grid(alpha=0.25)
    ax.legend()
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Edge Rate")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
