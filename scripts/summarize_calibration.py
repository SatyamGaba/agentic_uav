import json

with open("audit/interactive_session/independent_calibration_non_test_scenes.json") as f:
    data = json.load(f)

print(f"Total configs evaluated: {len(data)}")
print(f"{'r_a(m)':<8}{'w(m)':<8}{'y_min':<8}{'N_th':<8}{'Acc':<10}{'FBR(Deadlock)':<16}{'FFR(Bump)':<12}{'F1':<8}")
for r in data:
    if r["y_min"] == 0.15 and r["n_thresh"] == 5:
        print(f"{r['r_a']:<8.2f}{r['w']:<8.2f}{r['y_min']:<8.2f}{r['n_thresh']:<8}{r['accuracy']:<10.3f}{r['false_block_rate']:<16.3f}{r['false_free_rate']:<12.3f}{r['f1_score']:<8.3f}")

print("\nBest config overall by lowest FBR with Acc >= 0.75:")
valid = [r for r in data if r["accuracy"] >= 0.70]
valid.sort(key=lambda x: (x["false_block_rate"], -x["f1_score"]))
for r in valid[:5]:
    print(f"r_a={r['r_a']:.2f} (w={r['w']:.2f}m), y_min={r['y_min']:.2f}m, N={r['n_thresh']}: Acc={r['accuracy']:.3f}, FBR={r['false_block_rate']:.3f}, FFR={r['false_free_rate']:.3f}, F1={r['f1_score']:.3f}")

