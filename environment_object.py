import random
import numpy as np


class environment_object():

    def __init__(self):

        self.position = (0, 0)
        self.shape = None
        self.color = None
        self.is_passable = True
        self.is_food = False
        self.is_pushable = False

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            **vars(self)  # grabs all instance attributes dynamically
        }


class River(environment_object):
    def __init__(self, points, radius=5):
        super().__init__()
        self.shape = "polyline"
        self.is_passable = False
        self.is_food = False
        self.points = points  # List of waypoints
        self.radius = radius  # Width of the river


class Rock(environment_object):
    def __init__(self, position, radius=10):
        super().__init__()
        self.position = position
        self.shape = "circle"
        self.is_passable = False
        self.is_food = False
        self.radius = radius  # Radius of rock
        self.is_pushable = True


class Field(environment_object):
    def __init__(self, position, food, max_food, radius=10):
        super().__init__()
        self.position = position
        self.food = food
        self.max_food = max_food
        self.cooldown = 0
        self.shape = "rectangle"
        self.is_passable = True
        self.is_food = True
        self.radius = radius  # Radius of field


class Forest(environment_object):
    def __init__(self, position, radius=10):
        super().__init__()
        self.position = position
        self.shape = "rectangle"
        self.is_passable = True
        self.is_food = False
        self.radius = radius  # Radius of forest


def place_non_overlapping(obj_class, count, config, size, radius_key, existing_objects, *args, **kwargs):
    placed = []
    attempts = 0
    max_attempts = 1000

    while len(placed) < count and attempts < max_attempts:
        attempts += 1
        position = (random.randint(0, size), random.randint(0, size))
        radius = config[radius_key] * random.uniform(0.5, 1.5)

        # Handle field-specific constructor with food + max_food
        if obj_class == Field:
            food_fn = args[0] if args else (lambda: 0)
            food = food_fn()
            max_food = kwargs.get("max_food", food)
            obj = Field(position, food, max_food, radius)
        else:
            obj = obj_class(position, radius)

        # Ensure rock is not too close to the map edge
        if isinstance(obj, Rock):
            edge_buffer = config.get("Rock_edge_buffer", 5)
            x, y = obj.position
            # Skip placement if the rock’s boundary would come within edge_buffer of any map edge
            if (x - obj.radius) < edge_buffer or (x + obj.radius) > (size - edge_buffer) \
               or (y - obj.radius) < edge_buffer or (y + obj.radius) > (size - edge_buffer):
                continue

        if all(not is_too_close(obj, other) for other in existing_objects + placed):
            placed.append(obj)

    return placed


def is_too_close(obj1, obj2, buffer=-2.0):
    # distance check for two rivers (polylines)
    if obj1.shape == "polyline" and obj2.shape == "polyline":
        for p1 in obj1.points:
            for p2 in obj2.points:
                if np.linalg.norm(np.array(p1) - np.array(p2)) < (obj1.radius + obj2.radius - buffer):
                    return True

    # distance check for river vs non-river
    if obj1.shape == "polyline" and obj2.shape != "polyline":
        pos2 = np.array(obj2.position)
        for p1 in obj1.points:
            if np.linalg.norm(np.array(p1) - pos2) < (obj1.radius + obj2.radius - buffer):
                return True

    if obj2.shape == "polyline" and obj1.shape != "polyline":
        pos1 = np.array(obj1.position)
        for p2 in obj2.points:
            if np.linalg.norm(np.array(p2) - pos1) < (obj1.radius + obj2.radius - buffer):
                return True

    # default circle-circle overlap
    pos1 = np.array(obj1.position)
    pos2 = np.array(obj2.position)
    if np.linalg.norm(pos1 - pos2) < (obj1.radius + obj2.radius - buffer):
        return True

    return False


def generate_all_objects(config, size):
    all_objects = []
    rivers = generate_rivers(config, size)
    all_objects.extend(rivers)

    forests = place_non_overlapping(
        Forest, config["Forest"], config, size, "Forest_base_radius", all_objects)
    all_objects.extend(forests)
    fields = place_non_overlapping(Field, config["Field"], config, size, "Field_base_radius", all_objects,
                                   lambda: random.randint(*config["Field_food_range"]), max_food=config["Field_max_food"])
    all_objects.extend(fields)
    rocks = place_non_overlapping(
        Rock, config["Rock"], config, size, "Rock_base_radius", all_objects)
    all_objects.extend(rocks)

    return all_objects


def generate_rivers(config, size):
    rivers = []
    for _ in range(config["River"]):
        length = max(5, config.get("River_length", 10) // 2)  # fewer points
        radius = config["River_base_radius"] * random.uniform(0.8, 1.2)

        edge = random.choice(["top", "bottom", "left", "right"])
        center_bias = (size * 0.3, size * 0.7)

        if edge == "top":
            start = np.array([random.uniform(*center_bias), 0])
        elif edge == "bottom":
            start = np.array([random.uniform(*center_bias), size])
        elif edge == "left":
            start = np.array([0, random.uniform(*center_bias)])
        else:
            start = np.array([size, random.uniform(*center_bias)])

        points = [start]
        direction = None

        for _ in range(length - 1):
            last = points[-1]
            target = np.array([
                size / 2 + random.uniform(-size * 0.15, size * 0.15),
                size / 2 + random.uniform(-size * 0.15, size * 0.15)
            ])
            to_target = target - last
            to_target /= np.linalg.norm(to_target)

            if direction is None:
                direction = to_target
            else:
                direction = 0.6 * direction + 0.4 * to_target
                direction /= np.linalg.norm(direction)

            # More wobble = more curvature
            wobble_strength = size * 0.03
            noise = np.random.normal(0, wobble_strength, size=2)
            step = direction * (size * 0.08) + noise

            new_point = np.clip(last + step, 0, size)
            points.append(new_point)

        # Light smoothing pass
        smoothed = []
        for i in range(len(points)):
            neighbors = points[max(0, i - 1):min(len(points), i + 2)]
            avg = np.mean(neighbors, axis=0)
            smoothed.append(tuple(avg))

        rivers.append(River(smoothed, radius))

    return rivers
