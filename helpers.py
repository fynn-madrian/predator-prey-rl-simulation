import numpy as np
import json
import os
import tensorflow as tf
import numba as nb

config = {}
config_path = "config.json"

if config_path and os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

max_age = config.get("max_age")
reproduction_cooldown = config.get("reproduction_cooldown")
max_speed = config.get("max_speed")
map_size = config.get("map_size", 100)



@nb.njit
def _circle_collides(px, py, ox, oy, radius):
    dx = px - ox
    dy = py - oy
    return dx * dx + dy * dy < radius * radius

@nb.njit
def _rect_collides(px, py, ox, oy, half_size):
    return (ox - half_size <= px <= ox + half_size) and (oy - half_size <= py <= oy + half_size)

@nb.njit
def _polyline_collides(px, py, pts_x, pts_y, radius):
    n = len(pts_x) - 1
    for i in range(n):
        x1, y1 = pts_x[i], pts_y[i]
        x2, y2 = pts_x[i+1], pts_y[i+1]
        dx, dy = x2 - x1, y2 - y1
        px1, py1 = px - x1, py - y1

        len2 = dx * dx + dy * dy
        if len2 == 0.0:
            proj_x, proj_y = x1, y1
        else:
            t = max(0.0, min(1.0, (px1 * dx + py1 * dy) / len2))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

        dxp = px - proj_x
        dyp = py - proj_y
        if dxp * dxp + dyp * dyp < radius * radius:
            return True
    return False

@nb.njit(fastmath=True, nogil=True, parallel=True)
def first_hit_circles(pts,          # (N,S,2) float32  – ray sample points
                    pos, rad,     # (M,2) / (M,)      – object centres / radii
                    ray_thick):   # float32
    N, S, _ = pts.shape
    out_dist = np.full(N, -1, np.int32)
    out_idx  = np.full(N, -1, np.int32)

    r2 = (rad + ray_thick) ** 2      # (M,)

    for i in nb.prange(N):           # ← parallel over rays
        for j in range(S):           # walk along the ray
            x, y = pts[i, j]
            for k in range(pos.shape[0]):
                dx = x - pos[k, 0]
                dy = y - pos[k, 1]
                if dx*dx + dy*dy < r2[k]:
                    out_dist[i] = j
                    out_idx[i]  = k
                    break                    # next ray
            if out_dist[i] != -1:
                break
    return out_dist, out_idx


@nb.njit(fastmath=True, nogil=True)
def first_hit_polyline(pts_one,      # (S,2) sample points for ONE ray
                    line_pts,     # (P,2) polyline vertices
                    rad,          # float32
                    ray_thick):
    P = line_pts.shape[0] - 1
    r2 = (rad + ray_thick) ** 2
    for j in range(pts_one.shape[0]):            # along ray
        x, y = pts_one[j]
        for s in range(P):                       # over segments
            # segment start/end
            x1, y1 = line_pts[s]
            x2, y2 = line_pts[s+1]
            dx, dy = x2 - x1, y2 - y1
            # projection factor t∈[0,1]
            if dx == 0 and dy == 0:
                t = 0.
            else:
                t = ((x - x1)*dx + (y - y1)*dy) / (dx*dx + dy*dy)
                t = 0. if t < 0. else 1. if t > 1. else t
            cx = x1 + t*dx
            cy = y1 + t*dy
            ddx = x - cx
            ddy = y - cy
            if ddx*ddx + ddy*ddy < r2:
                return j
    return -1

def collides_with(obj, point):
    """Wrapper to dispatch collision checks to Numba-accelerated implementations."""
    px, py = float(point[0]), float(point[1])
    shape = getattr(obj, "shape", None)

    if shape == "circle":
        ox, oy = obj.position
        return _circle_collides(px, py, float(ox), float(oy), float(obj.radius))

    elif shape == "rectangle":
        ox, oy = obj.position
        return _rect_collides(px, py, float(ox), float(oy), float(obj.radius))

    elif shape == "polyline" and hasattr(obj, "points"):
        pts = np.array(obj.points, dtype=np.float32)
        if len(pts) < 2:
            return False
        pts_x = pts[:, 0]
        pts_y = pts[:, 1]
        return _polyline_collides(px, py, pts_x, pts_y, float(obj.radius))

    return False

@nb.njit(fastmath=True, cache=True)
def point_to_segment_distance(point, seg_start, seg_end):
    """
    Distance from `point` to the line segment [seg_start, seg_end] in ℝ².
    Logic identical to the original; ~2 × faster in tight loops.
    """
    px, py   = point
    x1, y1   = seg_start
    x2, y2   = seg_end
    dx, dy   = x2 - x1, y2 - y1

    # Degenerate segment → distance to the single point
    if dx == 0 and dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    # Project P onto the segment, clamped to t ∈ [0,1]
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5



def convert_observation(observation):
    rays = np.array(observation["rays"], dtype=np.float32)

    flat_parts = []
    flat_parts.extend(observation["facing"])
    flat_parts.extend(observation["position"] / map_size)
    flat_parts.extend(
        np.array(observation["velocity"], dtype=np.float32) / max_speed)
    flat_parts.extend(observation["good_vector"])
    flat_parts.append(observation["good_distance"])
    flat_parts.extend(observation["bad_vector"])
    flat_parts.append(observation["bad_distance"])
    flat = np.array(flat_parts, dtype=np.float32)

    return flat, rays
