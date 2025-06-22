from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

# ──────────────────────────────────────────────────────────────────────────────
# 1. Global Matplotlib style tweaks
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 12,
        "axes.titlesize": 22,
        "axes.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 1.6,
        "patch.linewidth": 0.7,
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load scenario configuration (if present)
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("config.json")
config: Dict[str, Any] = {}
if CONFIG_PATH.exists():
    config = json.loads(CONFIG_PATH.read_text())

GRID_SIZE: int = int(config.get("map_size", 100))
VISION_RANGE: int = int(config.get("vision_range", 20))
TASK_NAME: str = str(config.get("scenario", "gather")).capitalize()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Colour palette
# ──────────────────────────────────────────────────────────────────────────────
GROUP_COLORS = np.array(
    [
        [213, 94, 0],   # predator – rust
        [0, 114, 178],  # prey – blue
    ]
) / 255.0

OBJECT_COLORS: Dict[str, str] = {
    "River": "#4FA3D1",
    "Rock": "#A68D6E",
    "Field": "#7FB77E",
}

OEDGE = {"ec": "#333333", "lw": 0.6, "zorder": 3}


def first_hit_distance(origin, direction, objects, max_dist):
    """Return the distance along *direction* until first obstacle or *max_dist*.
    Supports circles, rectangles (via their center/radius), AND polyline obstacles
    by sampling each segment with overlapping circles."""
    min_t = max_dist
    ox, oy = origin
    dx, dy = direction

    for obj in objects:
        rad = obj.radius

        if getattr(obj, "shape", None) == "polyline":
            # sample each segment at ~half-radius intervals
            pts = np.asarray(obj.points, dtype=float)
            for p0, p1 in zip(pts[:-1], pts[1:]):
                seg = p1 - p0
                seg_len = np.linalg.norm(seg)
                if seg_len <= 0:
                    centers = [p0]
                else:
                    n = max(int(np.ceil(seg_len / (rad * 0.5))), 1)
                    ts = np.linspace(0.0, 1.0, n + 1)
                    centers = [p0 + t * seg for t in ts]

                for cx, cy in centers:
                    # same quadratic for ray→circle
                    A = dx * dx + dy * dy
                    B = 2 * (dx * (ox - cx) + dy * (oy - cy))
                    C = (ox - cx) ** 2 + (oy - cy) ** 2 - rad * rad
                    disc = B * B - 4 * A * C
                    if disc <= 0:
                        continue
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-B - sqrt_disc) / (2 * A)
                    if 0 < t1 < min_t:
                        min_t = t1

        else:
            # non-polyline: assume obj.position & obj.radius define a circle
            cx, cy = obj.position
            A = dx * dx + dy * dy
            B = 2 * (dx * (ox - cx) + dy * (oy - cy))
            C = (ox - cx) ** 2 + (oy - cy) ** 2 - rad * rad
            disc = B * B - 4 * A * C
            if disc <= 0:
                continue
            sqrt_disc = np.sqrt(disc)
            t1 = (-B - sqrt_disc) / (2 * A)
            if 0 < t1 < min_t:
                min_t = t1

    return min_t


def fov_patch(agent, objects, num_rays: int = 90):
    clr = GROUP_COLORS[int(agent.group)]
    fov_deg = config.get(
        "predator_fov") if agent.group == 0 else config.get("prey_fov")

    origin = np.asarray(agent.position, dtype=float)
    start_ang = np.degrees(np.arctan2(
        agent.facing[0], agent.facing[1])) - fov_deg / 2
    angles_deg = np.linspace(start_ang, start_ang + fov_deg, num_rays)

    verts = [origin[::-1]]
    for ang in angles_deg:
        ang_rad = np.deg2rad(ang)
        dir_vec = np.array([np.sin(ang_rad), np.cos(ang_rad)])
        dist = first_hit_distance(origin, dir_vec, objects, VISION_RANGE)
        endpoint = origin + dir_vec * dist
        verts.append(endpoint[::-1])

    face_rgba = (*clr, 0.10)
    edge_rgba = (*clr, 0.50)
    return Polygon(verts, closed=True, fc=face_rgba, ec=edge_rgba, lw=0.7, zorder=0)


def render(
    objects,
    agents: Dict[Any, Any],
    savedir: Union[str, Path, None] = None,
    *,
    goal=None,
    dpi: int | None = None,
    tick_step: int = 25,
):
    """Render the current state and optionally save to *savedir*."""

    if not hasattr(render, "_initialised"):
        fig = plt.figure(figsize=(12, 8), constrained_layout=False)
        gs: GridSpec = fig.add_gridspec(
            nrows=1, ncols=2, width_ratios=[4, 1], wspace=0.05
        )
        ax_map = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[0, 1])

        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        ax_legend.set_facecolor("none")
        for side in ("top", "right", "bottom", "left"):
            ax_legend.spines[side].set_visible(False)

        render._fig = fig
        render._ax_map = ax_map
        render._ax_legend = ax_legend
        render._initialised = True

    fig = render._fig  # type: ignore[attr-defined]
    ax = render._ax_map  # type: ignore[attr-defined]
    legend_ax = render._ax_legend  # type: ignore[attr-defined]

    # ───────────────────────────── Map axes ──────────────────────────────
    ax.clear()
    ax.set_facecolor("#F9F6F4")
    fig.patch.set_facecolor("white")
    ax.set_axisbelow(True)

    # 1) Draw FOV polygons
    for ag in agents.values():
        ax.add_patch(fov_patch(ag, objects))

    # 2) Static map objects
    for obj in objects:
        pos, rad = obj.position, obj.radius
        col = OBJECT_COLORS.get(obj.__class__.__name__, "#808080")

        if obj.shape == "rectangle":
            ax.add_patch(
                plt.Rectangle(
                    (pos[1] - rad, pos[0] - rad),
                    2 * rad,
                    2 * rad,
                    fc=col,
                    alpha=0.8,
                    **OEDGE,
                )
            )

        elif obj.shape == "circle":
            ax.add_patch(
                plt.Circle((pos[1], pos[0]), rad, fc=col, alpha=0.8, **OEDGE)
            )

        elif obj.shape == "polyline":
            # Draw the river as densely-sampled overlapping circles
            # shape (N,2) in (row, col) coords
            pts = np.asarray(obj.points)
            # Compute segment lengths and cumulative positions
            deltas = np.diff(pts, axis=0)
            seg_lens = np.linalg.norm(deltas, axis=1)
            cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
            total_len = cumlen[-1]
            if total_len == 0:
                # Degenerate case: just one circle
                center = pts[0]
                ax.add_patch(plt.Circle((center[1], center[0]), obj.radius,
                                        fc=col, alpha=0.8, ec="none", zorder=3))
            else:
                # Sampling step = half the radius (you can tweak this)
                step = obj.radius * 0.5
                n_samples = max(int(np.ceil(total_len / step)) + 1, 2)
                sample_ds = np.linspace(0.0, total_len, n_samples)
                for d in sample_ds:
                    # find which segment we're in
                    idx = np.searchsorted(cumlen, d, side="right") - 1
                    idx = np.clip(idx, 0, len(seg_lens)-1)
                    local_t = (d - cumlen[idx]) / \
                        (seg_lens[idx] if seg_lens[idx] > 0 else 1)
                    # interpolate in data coords
                    p = (1 - local_t)*pts[idx] + local_t*pts[idx+1]
                    ax.add_patch(
                        plt.Circle((p[1], p[0]), obj.radius,
                                   fc=col, alpha=0.8, ec="none", zorder=3)
                    )
    # 3) Agents – trail, body, facing line
    for ag in agents.values():
        pos = np.asarray(ag.position, dtype=float)
        clr = GROUP_COLORS[int(ag.group)]

        if hasattr(ag, "previous_position") and ag.previous_position is not None and len(ag.previous_position) > 0:
            prev = np.asarray(ag.previous_position)
            if prev.ndim == 1:
                ax.plot(
                    [prev[1], pos[1]],
                    [prev[0], pos[0]],
                    "-",
                    lw=0.9,
                    alpha=0.4,
                    color=clr,
                    zorder=2,
                )
            else:
                ax.plot(prev[:, 1], prev[:, 0], "-", lw=0.9,
                        alpha=0.4, color=clr, zorder=2)

        ax.plot(pos[1], pos[0], marker="o", ms=6,
                mew=0.8, mec="k", color=clr, zorder=4)

        facing_dir = np.asarray(ag.facing, dtype=float)
        norm = np.linalg.norm(facing_dir)
        if norm > 0:
            facing_dir /= norm
            dist_line = first_hit_distance(
                pos, facing_dir, objects, VISION_RANGE)
            end_pt = pos + facing_dir * dist_line
            ax.plot(
                [pos[1], end_pt[1]],
                [pos[0], end_pt[0]],
                "--",
                lw=0.8,
                color="k",
                zorder=1,
            )

    if goal is not None:
        ax.plot(goal[1], goal[0], marker="X", ms=10, mew=2,
                mec="#7E30CB", color="none", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-1, GRID_SIZE + 1)
    ax.set_ylim(GRID_SIZE + 1, -1)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#444")
        ax.spines[side].set_linewidth(1)

    ticks = np.arange(0, GRID_SIZE + 1, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=12)
    ax.set_yticklabels(ticks, fontsize=12)
    ax.tick_params(axis="both", which="both", length=4,
                   width=0.8, colors="#444", labelsize=12)
    ax.grid(which="both", linestyle=":",
            linewidth=0.4, color="#BBBBBB", alpha=0.5)

    ax.set_title(f"Scenario – {TASK_NAME}", weight="bold", fontsize=22, pad=18)

    # ───────────────────────── Legend axes ────────────────────────────
    legend_ax.clear()
    legend_ax.axis("off")

    handles, labels = [], []
    for name, col in OBJECT_COLORS.items():
        handles.append(
            plt.Line2D([0], [0], marker="s",
                       mec=OEDGE["ec"], mfc=col, ms=10, lw=0)
        )
        labels.append(name)

    handles.extend(
        [
            plt.Line2D([0], [0], marker="o", mec="k",
                       mfc=GROUP_COLORS[i], ms=8, lw=0)
            for i in (0, 1)
        ]
    )
    labels.extend(["Predator", "Prey"])

    handles.append(
        plt.Line2D([0], [0], marker=(3, 0, 0),
                   mfc="grey", alpha=0.2, ms=12, lw=0)
    )
    labels.append("Field of view")

    handles.append(plt.Line2D([0], [0], linestyle="--", color="k", lw=1))
    labels.append("Facing direction")

    handles.append(plt.Line2D([0], [0], color="#666666", lw=1))
    labels.append("Trail")

    legend = legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        borderpad=0.6,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#444444",
        ncol=1,
        prop={"size": 14},
    )
    legend.get_frame().set_facecolor("#FFFFFF")
    legend.get_frame().set_linewidth(1)

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)

    if savedir:
        fig.savefig(savedir, dpi=dpi or 300, bbox_inches="tight")
    else:
        plt.pause(0.001)
