import habitat_sim
import numpy as np

# Test equivalence of pathfinder.try_step vs sim.step("move_forward")
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = "/home/krutik/data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb"
cfg.enable_physics = False

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
    )
}
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
pf = sim.pathfinder
if not pf.is_loaded:
    pf.load_nav_mesh("/home/krutik/data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.navmesh")

p = pf.get_random_navigable_point()
agent = sim.get_agent(0)
state = agent.get_state()
state.position = p
agent.set_state(state)

# 1. Step with sim.step
sim.step("move_forward")
pos_sim = agent.get_state().position

# 2. Step with try_step
agent.set_state(state)
from habitat_sim.utils.common import quat_rotate_vector
forward = quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))
target = state.position + forward * 0.25
pos_pf = pf.try_step(state.position, target)

diff = np.linalg.norm(np.array(pos_sim) - np.array(pos_pf))
print(f"Difference between sim.step and pf.try_step: {diff:.8f} m")
sim.close()
