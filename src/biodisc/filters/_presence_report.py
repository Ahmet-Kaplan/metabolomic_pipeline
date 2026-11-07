"""Stage 3 — HTML report (before/after the modified 80 % rule).

Three plots + an index:

    1. Max-presence-pct histogram — the canonical before/after view; every
       feature placed at its max_presence_pct, coloured by kept vs dropped,
       with the threshold line marked.
    2. Per-group presence breakdown — for each cell line, how many features
       survive if that group alone defined the rule, and the distribution of
       that group's presence %.
    3. m/z × RT scatter — kept (grey) vs dropped (red) features in
       chromatographic space. Reveals whether drops cluster in void volume or
       late-RT regions (artefacts vs real).

Reuses ``biodisc.eda._output.write_plot`` for HTML+PNG emission and
``biodisc.eda.report.build_index`` for the themed index page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index


# ================================================================
# Plot 1 — presence histogram
# ================================================================
def plot_presence_histogram(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Histogram of ``max_presence_pct`` with kept/dropped colouring."""
    mpp = metrics["max_presence_pct"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=mpp[mpp < threshold],
            xbins=dict(start=0.0, end=1.0, size=0.02),
            marker_color="#c7254e",
            opacity=0.85,
            name=f"Dropped (< {threshold:g})",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=mpp[mpp >= threshold],
            xbins=dict(start=0.0, end=1.0, size=0.02),
            marker_color="#1e4377",
            opacity=0.85,
            name=f"Kept (≥ {threshold:g})",
        )
    )
    fig.add_vline(
        x=threshold,
        line=dict(color="#c9a94e", width=2, dash="dash"),
        annotation_text=f"Threshold {threshold:g}",
        annotation_position="top right",
    )

    n_total = len(metrics)
    n_kept = int((mpp >= threshold).sum())
    n_dropped = n_total - n_kept

    fig.update_layout(
        title=dict(
            text=(
                "<b>Max-presence-% distribution (Stage 3 before/after)</b><br>"
                f"<sup>{n_total:,} features · "
                f"<b>KEPT {n_kept:,} ({n_kept / n_total * 100:.1f} %) · "
                f"DROPPED {n_dropped:,} ({n_dropped / n_total * 100:.1f} %)</b><br>"
                "Y-axis log-scale so the 27k-feature spike at 1.0 doesn't hide the tail.</sup>"
            )
        ),
        xaxis_title="max(presence % across the 5 biological groups)",
        yaxis_title="Feature count (log scale)",
        yaxis_type="log",
        barmode="stack",
        bargap=0.02,
        height=600,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="01_presence_histogram", emit_png=emit_png)


# ================================================================
# Plot 2 — per-group presence breakdown
# ================================================================
def plot_per_group_breakdown(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Stacked bar: for each group, features with presence ≥ / < threshold in THAT group.

    Reminder: "modified 80 % rule" keeps a feature if ≥ threshold in AT LEAST ONE group.
    This plot shows each group's individual pass rate as a diagnostic — a group whose
    bars are heavily left-shifted indicates sparser detection in that cell line.
    """
    pct_cols = sorted(
        [c for c in metrics.columns if c.endswith("_pct") and c != "max_presence_pct"]
    )
    groups = [c.replace("_pct", "") for c in pct_cols]

    rows = []
    for g, col in zip(groups, pct_cols, strict=True):
        n_above = int((metrics[col] >= threshold).sum())
        n_below = len(metrics) - n_above
        rows.append({"group": g, "above": n_above, "below": n_below})
    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["group"],
            y=df["above"],
            name=f"≥ {threshold:g} in this group",
            marker_color="#1e4377",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["group"],
            y=df["below"],
            name=f"< {threshold:g} in this group",
            marker_color="#d6dde6",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Per-group presence at the 80 % threshold</b><br>"
                "<sup>Stacked bar: features that would pass the rule if ONLY this group "
                "counted. The modified rule keeps a feature if any group's blue bar covers "
                "it — so the blue regions OVERLAP across groups for kept features.</sup>"
            )
        ),
        xaxis_title="Cell line",
        yaxis_title="Feature count",
        barmode="stack",
        height=540,
        legend=dict(title="Per-group fate"),
    )
    return write_plot(fig, output_dir, name="02_per_group_breakdown", emit_png=emit_png)


# ================================================================
# Plot 3 — m/z × RT scatter of kept vs dropped
# ================================================================
def plot_mz_rt_drops(
    metrics: pd.DataFrame,
    features: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Scatter of kept vs dropped features in m/z × RT space."""
    df = metrics[["feature_id", "max_presence_pct"]].merge(
        features[["feature_id", "rt_min", "m_over_z"]], on="feature_id", how="left"
    )
    df["fate"] = np.where(df["max_presence_pct"] >= threshold, "kept", "dropped")

    kept = df[df["fate"] == "kept"]
    dropped = df[df["fate"] == "dropped"]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=kept["rt_min"],
            y=kept["m_over_z"],
            mode="markers",
            marker=dict(size=3, color="#b8c4d6", opacity=0.35),
            name=f"Kept ({len(kept):,})",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=dropped["rt_min"],
            y=dropped["m_over_z"],
            mode="markers",
            marker=dict(size=5, color="#c7254e", opacity=0.85),
            name=f"Dropped ({len(dropped):,})",
            text=dropped["feature_id"],
            hovertemplate=("%{text}<br>RT=%{x:.2f} min<br>m/z=%{y:.4f}<extra></extra>"),
        )
    )

    # Summary: where do drops concentrate?
    void_drops = int(((df["fate"] == "dropped") & (df["rt_min"] < 0.5)).sum())
    late_drops = int(((df["fate"] == "dropped") & (df["rt_min"] >= 14)).sum())

    fig.update_layout(
        title=dict(
            text=(
                "<b>Kept vs dropped in m/z × RT space</b><br>"
                "<sup>Reveals whether drops cluster in void volume (RT &lt; 0.5 min) "
                "— a common sign of injection artefacts.</sup>"
            )
        ),
        xaxis_title="Retention time (min)",
        yaxis_title="m/z",
        height=640,
        legend=dict(title="Fate"),
        margin=dict(l=60, r=40, t=100, b=120),
    )
    fig.add_annotation(
        text=(
            f"<b>Drop spatial summary:</b> "
            f"void-volume drops (RT &lt; 0.5 min): {void_drops:,} · "
            f"late-RT drops (RT ≥ 14 min): {late_drops:,} · "
            f"elsewhere: {len(dropped) - void_drops - late_drops:,}"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=-0.18,
        showarrow=False,
        font=dict(size=10, color="#222"),
        align="left",
        bordercolor="#c9a94e",
        borderwidth=1,
        borderpad=8,
        bgcolor="#fdf9ed",
    )
    return write_plot(fig, output_dir, name="03_mz_rt_drops", emit_png=emit_png)


# ================================================================
# Report orchestrator
# ================================================================
def build_report(
    metrics: pd.DataFrame,
    features: pd.DataFrame,
    *,
    output_dir: Path,
    threshold: float,
    n_input: int,
    n_kept: int,
    n_dropped: int,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Generate all Stage 3 plots + index.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    spec = [
        (
            "01_presence_histogram",
            "1. Max-presence distribution (before/after)",
            (
                "Every feature placed at its max-across-groups presence %, coloured "
                "by kept vs dropped. The threshold line at 0.80 is the boundary."
            ),
            plot_presence_histogram,
            {"threshold": threshold},
        ),
        (
            "02_per_group_breakdown",
            "2. Per-group presence breakdown",
            (
                "Stacked bar per cell line: features that would pass the rule IF only "
                "this group defined it. The modified rule takes the max across groups."
            ),
            plot_per_group_breakdown,
            {"threshold": threshold},
        ),
        (
            "03_mz_rt_drops",
            "3. Kept vs dropped in m/z × RT space",
            (
                "Spatial map of the drop decisions. Clustering in void volume (RT < 0.5 min) "
                "is a classic sign of injection-artefact features being correctly removed."
            ),
            plot_mz_rt_drops,
            {"threshold": threshold},
        ),
    ]

    for key, title, desc, fn, extra in spec:
        if fn is plot_mz_rt_drops:
            paths = fn(metrics, features, output_dir=plots_dir, emit_png=emit_png, **extra)
        else:
            paths = fn(metrics, output_dir=plots_dir, emit_png=emit_png, **extra)
        outputs[key] = paths["html"]
        entries.append(
            ReportEntry(
                key=key,
                title=title,
                description=desc,
                html_href=f"plots/{paths['html'].name}",
                png_href=f"plots/{paths['png'].name}" if "png" in paths else None,
            )
        )

    extra_info = {
        "Input features": f"{n_input:,}",
        "Kept": f"{n_kept:,} ({n_kept / n_input * 100:.2f} %)",
        "Dropped": f"{n_dropped:,} ({n_dropped / n_input * 100:.2f} %)",
        "Threshold": f"{threshold}",
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 3 — Modified 80 % Rule",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
