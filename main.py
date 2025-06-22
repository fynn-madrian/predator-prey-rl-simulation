from custom_environment import CustomEnvironment
import json
import os
from helpers import convert_observation, collides_with
from algorithms import load_model, get_model
import random
import tensorflow as tf
import math
import numpy as np
from tensorflow.keras import backend as K


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs found: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Running on CPU.")

    config = {}
    config_path = "config.json"

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    render_enabled = config.get("render_enabled", False)
    env = CustomEnvironment(config=config)

    observations, _ = env.reset()
    alive_agent_count = len(env.agent_data)

    step = 0

    prey_agents = [agent for agent in env.agent_data.values()
                   if agent.group == 1]
    prey = prey_agents[0] if prey_agents else None
    prey.model.load_weights(
        "good_flee/models/prey/agent_2_model_5000000.weights.h5")
    # buffer for mean rewards over last 10 steps
    recent_mean_rewards = []

    while step < 5_000_000:
        step += 1
        actions = {}
        for agent_id, agent in env.agent_data.items():
            obs = observations[agent_id]

            action, lstm_state = agent.get_action(obs)
            actions[agent_id] = action

        observations, rewards, terminations, truncated, infos = env.step(
            actions)

        # assert rewards are numbers, not NaN or None
        for agent_id, reward in rewards.items():
            if not isinstance(reward, (int, float)):
                raise ValueError(
                    f"Reward for agent {agent_id} is not a number: {reward}")

        done = {agent_id: terminations[agent_id] or truncated[agent_id]
                for agent_id in env.agent_data.keys()}

        for agent_id, agent in env.agent_data.items():
            if agent_id in actions and agent.group == 1:
                log_prob = infos[agent_id]["log_prob"]
                flat, rays = convert_observation(observations[agent_id])
                action = actions[agent_id]

                reward = rewards[agent_id]
                agent_done = done[agent_id]
                value = agent.V

                agent.sequence_buffer.append(
                    [flat, rays, action, log_prob, reward,
                        agent_done, value])

                if len(agent.sequence_buffer) == env.sequence_length:
                    agent.append_to_buffer()
                if len(agent.buffer) >= env.buffer_size:
                    print(f"Optimizing model")

                    agent.optimize_model()
                    agent.buffer.clear()
                    agent.hidden_state_buffer.clear()

        if render_enabled:
            env.render()

        alive_agent_count = len(env.agents)
        # accumulate current mean reward
        if alive_agent_count > 0:

            current_mean = sum(rewards.values()) / alive_agent_count
        else:
            current_mean = 0
        recent_mean_rewards.append(current_mean)

        # every 10 steps, print average of the last 10
        if step % 100 == 0:
            avg_recent = sum(recent_mean_rewards) / len(recent_mean_rewards)
            print(
                f"step {step} - mean reward over last {len(recent_mean_rewards)} steps: {avg_recent} - alive agents: {alive_agent_count}")
            recent_mean_rewards.clear()

        if step % 1_000_000 == 0:
            print(f"Saving models at step {step}")
            current_run = sorted(os.listdir("logs"))[-1]

            predator_model_path = os.path.join(
                "logs", current_run, "models", "predator")
            prey_model_path = os.path.join(
                "logs", current_run, "models", "prey")

            for agent_id, agent in env.agent_data.items():
                if agent.group == 0:
                    agent.model.save_weights(os.path.join(
                        predator_model_path, f"agent_{agent_id}_model_{step}.weights.h5"))
                else:
                    agent.model.save_weights(os.path.join(
                        prey_model_path, f"agent_{agent_id}_model_{step}.weights.h5"))
                # navigation: treat reaching the goal as end of LSTM episode

        if any(done.values()):

            for agent in env.agent_data.values():
                if agent.sequence_buffer and len(agent.sequence_buffer) > 0 and agent.group == 1:
                    agent.append_to_buffer()
            observations, _ = env.reset()
            alive_agent_count = len(env.agent_data)
