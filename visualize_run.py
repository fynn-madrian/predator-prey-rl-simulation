import json
import cv2
import os
import pandas as pd
from custom_environment import Agent
from render import render
from environment_object import River, Field, Rock, Forest
# Get the most recent run
run = sorted(os.listdir("logs"))
# only consider directories, not files
run = [r for r in run if os.path.isdir(os.path.join("logs", r))][-1]
path = os.path.join("logs", run)

start_step = 15_500_000
end_step = start_step + 1000  # Change this to your desired end step

# Load environment data from JSONL
env_path = os.path.join(path, "environment.jsonl")

env_data = []
with open(env_path, "r") as f:
    for idx, line in enumerate(f):
        print(f"Processing environment step {idx}...")
        entry = json.loads(line)
        entry["step"] = idx
        if start_step <= idx <= end_step:
            env_data.append(entry)
        if idx == end_step:
            break
# Load agent data for needed timesteps only
species_dirs = ["prey", "predator"]
agent_data_by_id = {}

for species_dir in species_dirs:
    species = 0 if species_dir == "predator" else 1
    species_path = os.path.join(path, species_dir)
    for file in os.listdir(species_path):
        agent_id = file.replace("agent_", "").replace(".jsonl", "")
        file_path = os.path.join(species_path, file)
        agent_records = []
        with open(file_path, "r") as af:
            for idx, line in enumerate(af):
                if start_step <= idx <= end_step:
                    record = json.loads(line)
                    record["step"] = idx
                    agent_records.append(record)
                if idx > end_step:
                    break
        if agent_records:
            df = pd.DataFrame(agent_records)
            agent_data_by_id[agent_id] = [df, species]

# Ensure output directory exists
os.makedirs("visualizations", exist_ok=True)
# Clear previous visualizations
for filename in os.listdir("visualizations"):
    file_path = os.path.join("visualizations", filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")


def reconstruct_objects(obj_dicts):
    objs = []
    for obj in obj_dicts:
        cls = obj["type"]
        if obj["type"] == "River":
            instance = River(obj["points"], radius=obj["radius"])
        elif obj["type"] == "Field":
            instance = Field(obj["position"], obj["food"],
                             obj["max_food"], radius=obj["radius"])
        elif obj["type"] == "Rock":
            instance = Rock(obj["position"], radius=obj["radius"])
        elif obj["type"] == "Forest":
            instance = Forest(obj["position"], radius=obj["radius"])
        objs.append(instance)
    return objs


for env_entry in env_data:
    step = env_entry["step"]
    print(f"Rendering step {step}")

    objects = reconstruct_objects(env_entry["objects"])
    goal = env_entry.get("goal", None)

    agents = {}
    for agent_id, (df, species) in agent_data_by_id.items():
        row = df[df["step"] == step]
        if row.empty:
            continue
        row = row.iloc[0]

        agent = Agent(
            group=species,
            position=row["position"],
            age=row["age"],
            facing=row["facing"],
            max_speed=5,  # pulled from config
            max_age=10000,
            ID=int(agent_id)
        )
        agent.velocity = row["velocity"]
        agents[agent_id] = agent

    if agents or objects:
        render(objects, agents,
               savedir=f"visualizations/{step}.png", goal=goal)
    else:
        print(f"No agents or objects to render at step {step}, skipping.")


framerate = 30  # Adjust this value to control video speed

image_folder = 'visualizations'
video_name = f'video_{start_step}.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = [img for img in images if start_step <=
          int(img.split(".")[0]) <= end_step]
images.sort(key=lambda x: int(x.split(".")[0]))
print(f"Found {len(images)} images for video creation: {images[:5]}...")

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
    *'XVID'), framerate, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

print(f"Video saved as {video_name} with {framerate} FPS.")
