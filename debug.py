from custom_environment import CustomEnvironment
from render import render
from algorithms import AgentNet
from helpers import convert_observation
import os
import tensorflow as tf
import numpy as np

simple_config = {
    "map_size": 100,
    "max_age": 5_000_000,
    "scenario": "gather",
    "map_config": {
        "Rock": 2,
        "River": 0,
        "Field": 1,
        "Forest": 0,
        "Field_food_range": [10, 20],
        "Field_base_radius": 12,
        "Field_max_food": 25,
        "River_base_radius": 5,
        "Rock_base_radius": 2,
    },
    "render_enabled": True,
    "predator_fov": 120,
    "prey_fov": 180,
    "vision_range": 20,
    "vision_rays": 15,
    "agent_detection_radius": 1,
    "agent_collision_radius": 2,
    "buffer_size": 16,
    "mutation_rate": 0.0,
    "sequence_length": 32,
    "max_speed": 7.5,
}

env = CustomEnvironment(config=simple_config)
observations, infos = env.reset()

prey_agent = [a for a in env.agent_data.values() if a.group == 1][0]
prey_id = prey_agent.ID
"""predator_agent = [a for a in env.agent_data.values() if a.group == 0][0]
predator_id = predator_agent.ID
predator_agent.sequence_buffer = []
"""
prey_agent.sequence_buffer = []


for filename in os.listdir("visualizations_debug"):
    file_path = os.path.join("visualizations_debug", filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")


num_steps = 5_000_000
for step in range(num_steps):
    prey_obs = observations[prey_id]
    # predator_obs = observations[predator_id]

    manual_control = True
    if manual_control:
        prey_action = {
            "x_dir": int(input("prey_action move_x bin [0-4]: ")),
            "y_dir": int(input("prey_action move_y bin [0-4]: ")),
            "spin": int(input("prey_action move_spin bin [0-2]: ")),
        }
    else:
        prey_action, prey_lstm_state = prey_agent.get_action(prey_obs)
        prey_agent.last_lstm_state = prey_lstm_state

    # predator_action, _ = predator_agent.get_action(predator_obs)

    observations, rewards, terminations, truncations, infos = env.step({
        prey_id: prey_action,
        # predator_id: predator_action
    })
    prey_obs = observations[prey_id]
    print(env.scenario)
    print("Prey reward:", rewards[prey_id])
    # print rays as array, as well as support vector
    rays = prey_obs["rays"]
    rays = np.array(rays)
    # round rays to 2 decimal places
    rays = np.round(rays, 2)

    print(rays)

    print("good_vector:", prey_obs["good_vector"], "good_distance:", prey_obs["good_distance"],
          "bad_vector:", prey_obs["bad_vector"], "bad_distance:", prey_obs["bad_distance"])

    if simple_config["render_enabled"]:
        render(env.objects, env.agent_data, goal=env.goal_pos)

print("Final prey position:", prey_agent.position)
