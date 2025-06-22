import numpy as np
import json
import os
import tensorflow as tf

config = {}
config_path = "config.json"

if config_path and os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

max_age = config.get("max_age")
reproduction_cooldown = config.get("reproduction_cooldown")
max_speed = config.get("max_speed")
map_size = config.get("map_size", 100)  # Default to 1000 if not specified


def collides_with(obj, point):
    px, py = point

    if obj.shape == "polyline" and hasattr(obj, "points"):
        for i in range(len(obj.points) - 1):
            if point_to_segment_distance(point, obj.points[i], obj.points[i + 1]) < obj.radius:
                return True

    elif obj.shape == "rectangle":
        ox, oy = obj.position
        r = obj.radius
        return (ox - r <= px <= ox + r) and (oy - r <= py <= oy + r)

    elif obj.shape == "circle":
        ox, oy = obj.position
        return np.linalg.norm([px - ox, py - oy]) < obj.radius

    return False


def point_to_segment_distance(point, seg_start, seg_end):
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == dy == 0:
        return np.linalg.norm([px - x1, py - y1])

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return np.linalg.norm([px - closest_x, py - closest_y])


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
