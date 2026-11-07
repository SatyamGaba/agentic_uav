"""
Command-Line Interface for the Modular Aerial VLN Benchmark Framework.
Usage Examples:
    # 1. Run Heuristic baseline on OpenFly seen split (5 episodes):
    python run_framework_benchmark.py --adapter heuristic --env openfly --split benchmark/seen.json --num-episodes 5

    # 2. Run Direct-Pixel (See-Point-Fly) on Mock 3D simulator:
    python run_framework_benchmark.py --adapter direct_pixel --backend mock --env mock --num-episodes 5

    # 3. Run AgenticNav (Tool Calling) with local Ollama on AirSim:
    python run_framework_benchmark.py --adapter agentic_nav --backend ollama --model qwen3-vl:8b --env airsim --num-episodes 5
"""
import os
import sys
import argparse
import math

from framework import (
    create_adapter,
    create_backend,
    create_environment,
    VLNBenchmarkRunner,
    EpisodeGoal,
    ADAPTER_REGISTRY,
    BACKEND_REGISTRY,
    ENV_REGISTRY
)


def main():
    parser = argparse.ArgumentParser(description="Modular Aerial VLN Benchmark Framework CLI - Production Pipeline")
    parser.add_argument("--adapter", type=str, default="calibrated", choices=["calibrated", "calibrated_nav", "supernav", "agentic_nav", "tool_calling", "direct_pixel", "discrete_action", "openfly_agent"],
                        help="Navigation decision adapter (all heuristic rules removed)")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "openai"],
                        help="Live VLM backend provider (mock backends prohibited)")
    parser.add_argument("--model", type=str, default="qwen3.5:latest",
                        help="Model identifier running on live VLM backend")
    parser.add_argument("--env", type=str, default="openfly", choices=["openfly", "airsim"],
                        help="Simulation environment with closed-loop visual perception")
    parser.add_argument("--split", type=str, default="benchmark/seen.json",
                        help="Dataset JSON path for OpenFly environment")
    parser.add_argument("--num-episodes", type=int, default=40,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=35,
                        help="Maximum steps allowed per episode")
    parser.add_argument("--success-radius", type=float, default=5.0,
                        help="Success threshold distance in meters")
    parser.add_argument("--mode", type=str, default="continuous", choices=["continuous", "discrete"],
                        help="Flight kinematics mode: continuous (exact 3D visual servoing) or discrete (OpenFly discrete codes)")
    parser.add_argument("--output", type=str, default="benchmark_output.json",
                        help="Path to save benchmark JSON summary")
    args = parser.parse_args()

    # Enforce strict real pipeline execution (no fakes, no heuristics)
    if args.backend.lower() == "mock":
        raise ValueError("Mock backend prohibited. All evaluations must execute real VLM inference.")
    if args.adapter.lower() == "heuristic":
        raise ValueError("Heuristic rule adapter prohibited. Benchmarks must run end-to-end multimodal agents.")

    # 1. Instantiate Real VLM Backend
    print(f"\n[PIPELINE] Connecting to live VLM backend: {args.backend} ({args.model})...")
    backend = create_backend(args.backend, model_name=args.model)

    # 2. Instantiate Real Navigation Adapter
    print(f"[PIPELINE] Initializing navigation adapter: {args.adapter} (mode={args.mode}) with live VLM...")
    adapter_kwargs = {}
    if args.adapter in ["calibrated", "calibrated_nav"]:
        adapter_kwargs["mode"] = args.mode
    adapter = create_adapter(args.adapter, backend=backend, **adapter_kwargs)

    # 3. Instantiate Environment
    print(f"Initializing environment: {args.env}...")
    if args.env == "openfly":
        if not os.path.exists(args.split):
            print(f"Error: dataset split '{args.split}' not found.")
            sys.exit(1)
        env = create_environment(args.env, dataset_path=args.split)
    else:
        env = create_environment(args.env)

    # 4. Prepare Goals
    goals = []
    if os.path.exists(args.split):
        print(f"Loading {args.num_episodes} goals from dataset split: {args.split}...")
        from framework.environments.openfly_env import OpenFlyBenchmarkEnv
        helper_env = OpenFlyBenchmarkEnv(args.split)
        limit = min(args.num_episodes, len(helper_env.episodes))
        for i in range(limit):
            goals.append(helper_env.load_episode_goal(i))
    else:
        print(f"Generating {args.num_episodes} synthetic goals aligned with flight corridors...")
        for i in range(args.num_episodes):
            # Distances: 18m to 36m across realistic forward corridor angles within camera FOV
            dist = 18.0 + (i % 4) * 6.0
            # Sector angles within +/- 25 degrees (inside 90-deg camera FOV)
            angle = ((i % 5) - 2) * 0.18
            gx = dist * math.cos(angle)
            gy = dist * math.sin(angle)
            
            if abs(angle) < 0.05:
                dir_hint = "straight ahead"
            elif angle < 0:
                dir_hint = "bearing slightly to the left"
            else:
                dir_hint = "bearing slightly to the right"

            goals.append(EpisodeGoal(
                episode_id=i + 1,
                instruction=f"Fly forward towards the building {dir_hint} and stop upon reaching within 5 meters.",
                goal_pos=[gx, gy, 5.0],
                gt_trajectory=[[0.0, 0.0, 5.0], [gx * 0.5, gy * 0.5, 5.0], [gx, gy, 5.0]],
                success_distance=args.success_radius,
                max_steps=args.max_steps
            ))

    # 5. Execute Benchmark
    runner = VLNBenchmarkRunner(
        adapter=adapter,
        env=env,
        model_name=args.model if args.adapter != "heuristic" else "HeuristicRule",
        max_steps_per_episode=args.max_steps,
        success_radius_m=args.success_radius
    )

    summary = runner.run_benchmark(goals, output_json_path=args.output)
    env.close()


if __name__ == "__main__":
    main()
