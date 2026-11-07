"""
Live Demonstrator for Aerial Closed-Loop VLN Framework.
Connects to AirSim, executes SuperNav with Qwen3-VL, and streams
telemetry for each closed-loop cycle.
"""
import time
from framework import (
    create_adapter,
    create_backend,
    create_environment,
    EpisodeGoal
)

def run_live_demonstration():
    print("=" * 70)
    print("  LIVE CLOSED-LOOP VLN DEMO")
    print("  Model: qwen3-vl:8b | Adapter: SuperNav | Env: AirSimUnrealEnv")
    print("=" * 70)

    print("\n[Phase 1: Component Initialization]")
    backend = create_backend("ollama", model_name="qwen3-vl:8b")
    adapter = create_adapter("supernav", backend=backend)
    env = create_environment("airsim")

    print("\n[Phase 2: Defining Target Navigation Goal]")
    goal = EpisodeGoal(
        episode_id=101,
        instruction="Fly forward towards the large building structure ahead and stop upon reaching within 5 meters.",
        goal_pos=[24.0, 0.0, 5.0],
        gt_trajectory=[[0.0, 0.0, 5.0], [12.0, 0.0, 5.0], [24.0, 0.0, 5.0]],
        success_distance=5.0,
        max_steps=6
    )

    print("\n[Phase 3: Resetting Simulator & Drone State]")
    obs = env.reset(goal)
    adapter.reset(obs)
    start_p = obs.pose
    init_dist = ((start_p.x - 24.0)**2 + start_p.y**2)**0.5
    print(f"  Drone Origin:  X={start_p.x:.2f}m, Y={start_p.y:.2f}m, Z={start_p.z:.2f}m, Yaw={start_p.yaw:.2f}rad")
    print(f"  Goal Position: X=24.00m, Y=0.00m, Z=5.00m")
    print(f"  Initial Error: {init_dist:.2f}m (Threshold: 5.00m)")

    print("\n[Phase 4: Closed-Loop Execution Loop]")
    step = 0
    done = False

    while not done and step < 6:
        step += 1
        t0 = time.time()
        print(f"\n--- Closed-Loop Cycle #{step} ---")

        # Step A: Perception & Multi-Modal Visual Grounding
        print("  [Perception -> VLM]: Forwarding 384x216 RGB + Instruction to qwen3-vl...")
        action = adapter.step(obs)
        t_vlm = time.time() - t0

        print(f"  [VLM Anchor Output]: ({action.u:.2f}, {action.v:.2f}) | Standoff Depth={action.depth_m:.2f}m | Inferred in {t_vlm:.2f}s")
        print(f"  [VLM Reasoning]:     \"{action.reasoning}\"")
        print(f"  [Geometry & Hop]:    dx={action.dx:.2f}m, dy={action.dy:.2f}m, dz={action.dz:.2f}m, yaw_delta={action.yaw_delta:.2f}rad | Stop={action.is_stop}")

        # Step B: Simulator Actuation with Safety Shield
        print("  [Actuation -> AirSim]: Dispatching 3D trajectory to multirotor flight controller...")
        next_obs, done, info = env.step(action)
        t_cycle = time.time() - t0

        cur_p = next_obs.pose
        dist_now = ((cur_p.x - 24.0)**2 + cur_p.y**2)**0.5
        print(f"  [Updated Drone Pose]: X={cur_p.x:.2f}m, Y={cur_p.y:.2f}m, Z={cur_p.z:.2f}m | Yaw={cur_p.yaw:.2f}rad")
        print(f"  [Remaining Distance]: {dist_now:.2f}m (Delta: {init_dist - dist_now:+.2f}m) | Cycle Time: {t_cycle:.2f}s")

        if done:
            status = "SUCCESS" if dist_now <= 5.0 else "TERMINATED"
            print(f"\n>> Cycle Exit Condition Met: {status} | Final Dist={dist_now:.2f}m | Reason={info.get('reason', 'Goal Reached')}")
            break

        obs = next_obs

    env.close()
    print("\n" + "=" * 70)
    print("  DEMONSTRATION RUN FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    run_live_demonstration()
