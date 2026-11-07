import sys
from pathlib import Path
import json

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment

with open(PROJECT_ROOT / "agenticnav_paper_replica/data/episode_list_100.json") as f:
    data = json.load(f)

ep0 = [e for e in data['episodes'] if e['scene_id'] == 'gZ6f7yhEvPG'][0]
print(f"Testing episode: {ep0['episode_id']} in {ep0['scene_id']}")

emb = HabitatGroundEmbodiment(scene_id=ep0['scene_id'], gpu_device_id=-1)
obs = emb.reset(ep0)

print(f"Start pose: {obs.allocentric_pose}")
print(f"Goal pos: {emb.current_goal_pos}")
print(f"Shortest path len: {emb.get_shortest_path_distance():.2f}m")
print(f"Initial geodesic dist to goal: {emb.get_geodesic_distance_to_goal():.2f}m")

emb.close()
print("Habitat reset test successful!")

