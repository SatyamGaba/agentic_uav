import json

try:
    with open("runs/local_10_complex/benchmark_summary.json", "r") as f:
        data = json.load(f)
    
    print("Model:", data.get("model"))
    print("Episodes Completed:", data.get("episodes_completed"))
    results = data.get("results", data.get("episodes", []))
    print(f"Total entries: {len(results)}/10")
    for r in results:
        t_idx = r.get("test_index")
        eid = r.get("episode_id")
        ne = r.get("navigation_error_m")
        succ = r.get("success")
        spl = r.get("spl")
        steps = r.get("total_steps")
        traj = r.get("trajectory_length_m")
        print(f"[{t_idx:02d}/10] {eid}: Success={succ} | NE={ne:5.2f}m | SPL={spl:.4f} | Steps={steps} | Traj={traj:.2f}m")
except Exception as e:
    print("Error:", e)

