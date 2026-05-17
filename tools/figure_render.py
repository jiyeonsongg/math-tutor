"""Render structured question diagrams with matplotlib."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, Polygon, Rectangle
from tools.diagram_models import QuestionDiagram

logger = logging.getLogger(__name__)

# Room inside the figure so labels, legend, and points are not clipped in Streamlit.
_FIGSIZE_SQUARE = (4.6, 4.6)
_FIGSIZE_WIDE = (6.0, 2.4)
_PAD_FRAC = 0.18
ZOOM_MIN = 0.35
ZOOM_MAX = 3.0
ZOOM_IN_STEP = 0.82
ZOOM_OUT_STEP = 1.22


def _apply_zoom(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    zoom: float,
) -> tuple[float, float, float, float]:
    """Smaller zoom factor = tighter limits = zoomed in."""
    if abs(zoom - 1.0) < 1e-9:
        return x0, x1, y0, y1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_x = (x1 - x0) / 2 * zoom
    half_y = (y1 - y0) / 2 * zoom
    return cx - half_x, cx + half_x, cy - half_y, cy + half_y


def _padded_limits(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    pad_frac: float = _PAD_FRAC,
) -> tuple[float, float, float, float]:
    xw = (x_max - x_min) or 1.0
    yw = (y_max - y_min) or 1.0
    return (
        x_min - xw * pad_frac,
        x_max + xw * pad_frac,
        y_min - yw * pad_frac,
        y_max + yw * pad_frac,
    )


def _finalize_figure(fig: plt.Figure) -> None:
    fig.tight_layout(pad=1.8)
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.16, top=0.88)


def _graph_limits(d: Any) -> tuple[float, float, float, float]:
    xs = [d.x_min, d.x_max]
    ys = [d.y_min, d.y_max]
    for pt in d.graph_points:
        xs.append(pt.x)
        ys.append(pt.y)
    for line in d.lines:
        for attr in ("x1", "x2", "y1", "y2"):
            v = getattr(line, attr, None)
            if v is not None:
                (xs if attr.startswith("x") else ys).append(v)
    return _padded_limits(min(xs), max(xs), min(ys), max(ys))


def _parse_diagram(data: Any) -> QuestionDiagram | None:
    if not data:
        return None
    if isinstance(data, dict):
        d = QuestionDiagram.model_validate(data)
        if d.type == "none":
            return None
        return d
    return None


def _style_axes(ax: plt.Axes, *, title: str, equal_aspect: bool = False) -> None:
    if title:
        ax.set_title(title, fontsize=11)
    ax.axhline(0, color="#666", linewidth=0.8)
    ax.axvline(0, color="#666", linewidth=0.8)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def _draw_graph(d: Any, *, zoom: float = 1.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    x0, x1, y0, y1 = _apply_zoom(*_graph_limits(d), zoom)
    if d.show_grid:
        ax.grid(True, alpha=0.35)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x", labelpad=6)
    ax.set_ylabel("y", labelpad=6)

    xs = np.linspace(x0, x1, 200)
    for line in d.lines:
        if line.x1 is not None and line.x2 is not None and line.y1 is not None and line.y2 is not None:
            ax.plot([line.x1, line.x2], [line.y1, line.y2], linewidth=2, label=line.label or None)
        elif line.slope is not None:
            ys = line.slope * xs + (line.intercept or 0)
            ax.plot(xs, ys, linewidth=2, label=line.label or None)
        elif line.intercept is not None:
            ax.axhline(line.intercept, linewidth=2, label=line.label or None)
    for pt in d.graph_points:
        ax.plot(pt.x, pt.y, "o", color="#c2185b", markersize=7)
        if pt.label:
            ax.annotate(pt.label, (pt.x, pt.y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    if d.lines:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    _style_axes(ax, title=d.title)
    _finalize_figure(fig)
    return fig


def _draw_polygon(d: Any, *, zoom: float = 1.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    if d.show_grid:
        ax.grid(True, alpha=0.35)
    if len(d.vertices) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    xs = [v.x for v in d.vertices]
    ys = [v.y for v in d.vertices]
    patch = Polygon(list(zip(xs, ys, strict=True)), closed=True, fill=False, edgecolor="#1565c0", linewidth=2)
    ax.add_patch(patch)
    for v in d.vertices:
        ax.plot(v.x, v.y, "o", color="#c2185b", markersize=6)
        if v.label:
            ax.annotate(v.label, (v.x, v.y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    x0, x1, y0, y1 = _apply_zoom(*_padded_limits(min(xs), max(xs), min(ys), max(ys)), zoom)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    _style_axes(ax, title=d.title, equal_aspect=True)
    _finalize_figure(fig)
    return fig


def _draw_right_triangle(d: Any, *, zoom: float = 1.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    if d.show_grid:
        ax.grid(True, alpha=0.35)
    verts = [(0, 0), (d.leg_x, 0), (0, d.leg_y)]
    patch = Polygon(verts, closed=True, fill=False, edgecolor="#1565c0", linewidth=2)
    ax.add_patch(patch)
    ax.plot([0, d.leg_x, 0, 0], [0, 0, d.leg_y, 0], "o", color="#c2185b", markersize=6)
    if d.label_horizontal:
        ax.annotate(d.label_horizontal, (d.leg_x / 2, 0), textcoords="offset points", xytext=(0, -12), ha="center")
    if d.label_vertical:
        ax.annotate(d.label_vertical, (0, d.leg_y / 2), textcoords="offset points", xytext=(-12, 0), va="center")
    if d.label_hypotenuse:
        ax.annotate(d.label_hypotenuse, (d.leg_x / 2, d.leg_y / 2), fontsize=9)
    x0, x1, y0, y1 = _apply_zoom(
        *_padded_limits(0, d.leg_x, 0, d.leg_y, pad_frac=0.22),
        zoom,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    _style_axes(ax, title=d.title, equal_aspect=True)
    _finalize_figure(fig)
    return fig


def _draw_rectangle(d: Any, *, zoom: float = 1.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    ax.grid(True, alpha=0.35)
    rect = Rectangle((0, 0), d.width, d.height, fill=False, edgecolor="#1565c0", linewidth=2)
    ax.add_patch(rect)
    if d.label_width:
        ax.annotate(d.label_width, (d.width / 2, 0), textcoords="offset points", xytext=(0, -12), ha="center")
    if d.label_height:
        ax.annotate(d.label_height, (0, d.height / 2), textcoords="offset points", xytext=(-14, 0), va="center")
    x0, x1, y0, y1 = _apply_zoom(
        *_padded_limits(0, d.width, 0, d.height, pad_frac=0.22),
        zoom,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    _style_axes(ax, title=d.title, equal_aspect=True)
    _finalize_figure(fig)
    return fig


def _draw_circle(d: Any, *, zoom: float = 1.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    if d.show_grid:
        ax.grid(True, alpha=0.35)
    circ = Circle((d.center_x, d.center_y), d.radius, fill=False, edgecolor="#1565c0", linewidth=2)
    ax.add_patch(circ)
    ax.plot(d.center_x, d.center_y, "o", color="#c2185b", markersize=6)
    x0, x1, y0, y1 = _apply_zoom(
        *_padded_limits(
            d.center_x - d.radius,
            d.center_x + d.radius,
            d.center_y - d.radius,
            d.center_y + d.radius,
            pad_frac=0.22,
        ),
        zoom,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    _style_axes(ax, title=d.title, equal_aspect=True)
    _finalize_figure(fig)
    return fig


def _draw_number_line(d: Any, *, zoom: float = 1.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.axhline(0.5, color="#333", linewidth=2)
    for t in range(int(d.number_line_min), int(d.number_line_max) + 1):
        ax.plot([t, t], [0.45, 0.55], color="#333", linewidth=1)
        ax.text(t, 0.38, str(t), ha="center", fontsize=8)
    for pt in d.number_line_points:
        ax.plot(pt.value, 0.5, "o", color="#c2185b", markersize=8)
        label = pt.label or str(pt.value)
        ax.annotate(label, (pt.value, 0.5), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=9)
    span = (d.number_line_max - d.number_line_min) or 1.0
    side = max(0.8, span * 0.12)
    x0, x1 = d.number_line_min - side, d.number_line_max + side
    x0, x1, _, _ = _apply_zoom(x0, x1, 0.0, 1.0, zoom)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.05, 1.12)
    ax.axis("off")
    if d.title:
        ax.set_title(d.title, fontsize=11, pad=10)
    _finalize_figure(fig)
    return fig


def build_figure(diagram: Any, *, zoom: float = 1.0) -> plt.Figure | None:
    parsed = _parse_diagram(diagram)
    if parsed is None:
        return None
    kind = parsed.type
    if kind == "graph":
        return _draw_graph(parsed, zoom=zoom)
    if kind == "polygon":
        return _draw_polygon(parsed, zoom=zoom)
    if kind == "right_triangle":
        return _draw_right_triangle(parsed, zoom=zoom)
    if kind == "rectangle":
        return _draw_rectangle(parsed, zoom=zoom)
    if kind == "circle":
        return _draw_circle(parsed, zoom=zoom)
    if kind == "number_line":
        return _draw_number_line(parsed, zoom=zoom)
    return None


def render_question_diagram(diagram: Any, *, control_key: str) -> None:
    """Draw diagram with + / − zoom controls (per question)."""
    if not diagram:
        return

    zoom_key = f"diag_zoom_{control_key}"
    if zoom_key not in st.session_state:
        st.session_state[zoom_key] = 1.0

    z_in, z_out, z_reset, _ = st.columns([0.07, 0.07, 0.08, 0.78])
    with z_in:
        if st.button("+", key=f"zoom_in_{control_key}", help="Zoom in"):
            st.session_state[zoom_key] = max(ZOOM_MIN, st.session_state[zoom_key] * ZOOM_IN_STEP)
            st.rerun()
    with z_out:
        if st.button("−", key=f"zoom_out_{control_key}", help="Zoom out"):
            st.session_state[zoom_key] = min(ZOOM_MAX, st.session_state[zoom_key] * ZOOM_OUT_STEP)
            st.rerun()
    with z_reset:
        if st.button("Reset", key=f"zoom_reset_{control_key}", help="Reset zoom"):
            st.session_state[zoom_key] = 1.0
            st.rerun()

    zoom = float(st.session_state[zoom_key])
    try:
        fig = build_figure(diagram, zoom=zoom)
        if fig is None:
            return
        st.pyplot(fig, clear_figure=True, use_container_width=True, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diagram render failed: %s", exc)
        st.caption("Figure could not be drawn for this question.")
