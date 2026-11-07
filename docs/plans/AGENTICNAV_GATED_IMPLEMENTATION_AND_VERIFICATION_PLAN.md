# AgenticNav Local Replication: Gated Implementation and Verification Plan

## 0. How this plan works
Every phase has a **brief** (what gets built), **checks** (how you prove it works) and a **gate** (a pass criterion that must be met before the next phase starts). Each check writes evidence to `audit/phaseN/` (logs, hashes, images, JSON). A phase is closed only when its evidence folder exists and every check is green. If a check fails, fix the cause, then re-run the whole phase, not just the failed check.
Check IDs are written `G<phase>.<number>`. Numeric thresholds marked `(proposed)` are engineering choices made in this plan, not values from the paper. Change them only by editing the spec ledger (section 2.1), never silently in code.

---

## 1. Review of the existing plan: what must change first
1. **Model choice**. The old plan pulls `qwen2.5vl`. That is unnecessary. The official `qwen3.5` entry on Ollama is a multimodal family (vision, tools, thinking; 0.8b to 122b), and `Qwen3-VL` is also official (2b, 4b, 8b, 30b and up). The earlier failure came from a third-party abliterated text-only build, not from the Qwen 3.5 family. Rule: official library tags only, and only after `ollama show` confirms vision.
2. **VRAM**. The library lists `qwen3.5:9b` at about 6.6 GB on disk and `qwen3.5:4b` at about 3.4 GB. Your RTX 4050 has 6 GB, shared with Windows and with the simulator's renderer. 9B cannot be fully GPU-resident; it will split across CPU and GPU. Use 4B as the development model for now.
3. **Option B (pyrender/trimesh) is rejected**. It has no navmesh, so no geodesic distances (success and SPL cannot be computed the same way), no collision or sliding behaviour, and different camera and depth conventions. It cannot give parity with the cluster. Only Habitat-Sim is acceptable.
4. **WSL2 parity is unproven**. The old plan says '100% identical'. That has not been demonstrated. Habitat-Sim on WSL2 has a history of EGL failures (unable to find CUDA device 0 among N EGL devices; unable to create windowless context) and mixed success reports. Phase 3 is therefore a go/no-go gate with defined fallbacks.
5. **'Verified' claims are mock-only**. The occupancy grid, tool harness and metrics were verified against `MockGroundEmbodiment`, whose depth and geometry do not exercise real sensor conventions (depth normalisation, z-depth versus ray length, axis handedness, mesh holes). They must be re-verified on real simulator data (Phases 4 to 6).
6. **Unconfirmed parameters**. 4 views at 640x480 and 90 degrees HFOV, 15 degree turns, 0.25 m steps, 0.30 m standoff, 0.96 m x 1.15 m corridor, E <= 4, R <= 5, K = 6, 256x256 grid at 0.15 m/px, height band 0.10 to 1.25 m, success radius. The public project page describes the three tools and the memory design but does not state these numbers. Each must be confirmed from the paper PDF or appendix, or labelled an assumption (Phase 0).
7. **Success radius**. R2R-CE success is normally defined as stopping within 3.0 m of the goal. The line '(and Habitat standard <= 2.0 m)' mixes definitions. Use one definition, confirmed in Phase 0, and report it.
8. **Headline numbers are not your target**. The 76% SR and 66.50% SPL are reported with Gemini backbones (the project page labels its ablations as Gemini-3.7-Flash). A local 4B to 9B open model should not be expected to reproduce them. Success for this project is (a) pipeline correctness proven by the gates, and (b) results comparable to the paper's open-weight backbone rows, once read from the paper.
9. **Missing silent-failure controls**. Silent fallbacks (mock sim, images disabled), Ollama context truncation, wrong image order or count, thinking-mode token burn, and model-specific pixel coordinate conventions. All are covered below.
10. **Statistics**. With 100 episodes, a 95% confidence interval on a 76% success rate is roughly plus or minus 8 to 9 percentage points. Smaller differences are not evidence. Report intervals, and tune prompts and harness only on a dev split, never on the locked test episodes.

---

## 2. Ground rules
- **2.1 Spec ledger**. Create `spec/spec_ledger.yaml` with one entry per parameter: name, value, source (paper section, VLN-CE config file, or 'assumption'), status (verified or assumed), owner. A CI check fails if a constant in the code has no ledger entry. Assumed values on the critical path need a planned ablation.
- **2.2 Fail fast, no silent fallback**.
  - `BACKEND` must be `habitat` for real runs. If `habitat_sim` fails to import or a scene is missing, the run aborts. The mock only runs with an explicit `--allow-mock` flag, stamps `MOCK` on every output and every BEV image, and is refused by the benchmark script.
  - If the model lacks vision, or an image fails to attach, abort. Never continue text-only.
  - Any Ollama HTTP error aborts the step after a bounded, logged retry.
- **2.3 Run manifest**. Every run writes `manifest.json` before step 1: git commit and dirty flag, spec ledger hash, config hash, Ollama version, model tag and digest, model capabilities, inference options (`num_ctx`, temperature, seed, think), `habitat_sim` and `habitat-lab` versions, GPU and driver, scene id and scene file sha256, episode id, episode file sha256, Python lockfile hash.
- **2.4 Evidence**. Each gate stores outputs in `audit/phaseN/`. Image evidence is saved as PNG contact sheets with the check ID in the filename.
- **2.5 Test integrity**. Three episode pools: `dev` (train or val_seen episodes; prompts and harness may be tuned here), `shakedown` (a few val_unseen episodes used only for end-to-end smoke tests) and `locked test` (the 100 val_unseen episodes; run only after the Phase 9 config freeze). No prompt edits after the freeze.

---

## Phase 0. Source of truth
**Brief**. Freeze exactly what is being replicated before writing more code.  
**Do**:
- Read the paper PDF, appendix and any released code. Extract: the exact tool names and argument schemas (the project page shows Depth, Action and Recall tools; your code uses `query_depth`, `move_to`, `recall` and `stop`, so confirm the real schemas), the system and user prompt text, the map image spec, the memory spec (reasoning text history plus compact map image plus recall), every numeric parameter in section 1 item 6, the episode subset and how the 100 episodes were chosen, the success radius, the step or time budget, and which open-weight backbone rows exist.
- Fill the spec ledger.  
**Checks**:
- **G0.1**: Every numeric constant and prompt string in the code maps to a ledger line (automated script).
- **G0.2**: Every assumed item has a reason and an ablation plan, or is removed.
- **G0.3**: The episode list file is identified and checksummed, and its scene set equals your 11 scenes (derived from the file, not typed by hand).
- **G0.4**: Target definition written: which paper rows your local result will be compared with.  
**Gate**: ledger committed; zero unmapped constants.

---

## Phase 1. Host baseline
**Brief**. Pin the machine so every later result is attributable.  
**Do**:
- Record the Windows NVIDIA driver version, Ollama version, WSL2 kernel and distro. Keep the project inside the WSL ext4 filesystem, not under `/mnt/c` (I/O speed and permissions).
- Decide the Ollama topology. Recommended: Ollama stays on Windows (uses the GPU directly) and the code in WSL calls it over the network, either through mirrored networking in `.wslconfig` or the Windows host IP. Record which.
- Create a locked Python environment (conda env export plus pip freeze).  
**Checks**:
- **G1.1**: `nvidia-smi` works in Windows and in WSL and reports the same GPU and driver.
- **G1.2**: `ollama --version` recorded. Update Ollama if a pull of the chosen model fails with a version error (Qwen 3.5 and Qwen3-VL need recent Ollama builds).
- **G1.3**: From WSL, a GET on `/api/tags` returns the model list; latency recorded.
- **G1.4**: Idle VRAM (Windows desktop included) recorded; free disk space exceeds scenes + models + audit output with margin.
- **G1.5**: Environment lock files saved in `audit/phase1`.

---

## Phase 2. Vision model layer
**Brief**. Choose the model with measured evidence, prove it really sees, and pin down its quirks before it touches the simulator.  
**Do**:
- Candidates, in order: `qwen3.5:4b` (fits the GPU; development), `qwen3.5:9b` (partial offload; fidelity), `qwen3-vl:8b` (same family as the cluster run; partial offload). Pull official tags only, for example `ollama pull qwen3.5:4b`. Never use a tag with a user namespace.
- Write the test battery as `scripts/vlm_battery.py` so it can be re-run on any candidate. Generate the images in code (known content) so ground truth is exact.  
**Checks (run on every candidate)**:
- **G2.1**: Provenance: `ollama show` lists the vision capability (and tools, if native tool-calling is used); digest recorded; tag has no namespace.
- **G2.2**: Single image: a generated image (red square on the left, blue circle on the right, the number 73 as text). Temperature 0. 5/5 correct.
- **G2.3**: Multi-image order and count: 5 images, each with a different large numeral and colour. Ask for the content of image $k$, for each $k$, in shuffled order. 20/20 correct. This proves count and order are preserved.
- **G2.4**: Blindness control: repeat G2.3 with all-black images. Answers must change to 'cannot see' or be wrong. If answers do not change, images are not being used; stop.
- **G2.5**: Context truncation: build the worst-case real payload (5 images at final resolution, maximum history, maximum prompt). Set `num_ctx` explicitly. Put a random sentinel string at the very start of the prompt and ask the model to repeat it; it must succeed. Record `prompt_eval_count` and derive tokens per image. Check `done_reason` is not a length stop.
- **G2.6**: Pixel-grounding convention: show 50 synthetic 640x480 images, each with one coloured dot; ask for its $(u, v)$. Qwen generations have used different coordinate conventions (absolute pixels of a resized image versus normalised 0 to 1000 style), so measure it. Fit the convention, state it explicitly in the prompt, and convert in the harness. Pass (proposed): median error <= 3% of image width and 90th percentile <= 8%. Repeat at the candidate input resolutions (native and downscaled) and choose.
- **G2.7**: Thinking mode: run G2.2 to G2.6 with thinking on and off. Record accuracy, tokens and latency. Choose one setting and put it in the ledger (a thinking model can burn the context and time budget but may ground better).
- **G2.8**: Latency and VRAM: worst-case payload, 10 runs. Record time to first token, total time, `ollama ps` (GPU/CPU split) and `nvidia-smi` peak. Estimate episode time as mean steps x model turns per step x latency and confirm 100 episodes are feasible.
- **G2.9**: Determinism: temperature 0 with a fixed seed, same payload, 5 runs. Record whether outputs are identical.
- **G2.10**: Protocol decision only (full tests are in Phase 7): native Ollama tool calling versus JSON in the reply text. Follow what the paper specifies.  
**Gate**: one primary model with G2.1 to G2.9 green, one fallback model, and the settings (`num_ctx`, `think`, `temperature`, `seed`, `image_resolution`, `coordinate_convention`) written to the ledger. Newer Qwen releases (3.6 and later) exist on Ollama, but their sizes (roughly 18 GB and up) do not fit this machine, so they are out of scope.

---

## Phase 3. Simulator (go/no-go)
**Brief**. Real Habitat-Sim rendering real Matterport3D scenes, matched to the cluster.  
**Do**:
1. Match versions. On the cluster record `habitat_sim`, `habitat-lab`, Python and the Habitat-Sim build (headless or not). Locally run `conda search habitat-sim -c aihabitat -c conda-forge` to see which versions and Python builds exist (0.3.3 is the latest conda release listed); do not assume Python 3.10. Pin to the cluster's versions where the channel offers them; otherwise record the difference as a risk and test its effect in G3.4.
2. Install in a fresh conda environment in WSL2 (`conda install habitat-sim withbullet headless -c conda-forge -c aihabitat`, or the non-headless build; see the order in G3.2).
3. Fetch assets from `/datasets/mp3d` on the cluster: the `.glb` scenes, matching `.navmesh` files, any scene dataset config, and the R2R-CE episode files. Matterport3D is licensed data; keep it private and within your access terms.  
**Checks**:
- **G3.1**: Import and version: `habitat_sim` imports and its version is written to the manifest.
- **G3.2**: GPU rendering gate (the biggest risk). Run the official example on the test scene, then on one MP3D scene. Pass = an observation is returned without 'unable to find CUDA device ... among N EGL devices' or 'unable to create windowless context', and `nvidia-smi` shows the process on the GPU. Try variants in this recorded order and stop at the first pass: (a) the headless build; (b) the non-headless build with WSLg; (c) Docker with the NVIDIA runtime inside WSL2. Log every attempt and its error. If none pass, follow the fallback tree in Appendix C. Do not continue with partial results.
- **G3.3**: Asset integrity: sha256 of every `.glb`, `.navmesh` and episode file matches the cluster copy. The episode file's scene list equals the 11 scenes. Every scene loads and its navmesh has navigable area greater than 0.
- **G3.4**: Cluster parity render: choose 20 fixed (scene, position, rotation) tuples. Render RGB and depth on both machines with identical sensor specs. First measure the cluster's own noise floor (render twice and compare), then compare local to cluster. Pass: differences within the noise floor times a recorded margin. If larger, find the cause (version, driver, shader path) before continuing.
- **G3.5**: Render quality: contact sheet of the four-view start frames for at least 10 episodes across different scenes; verify by eye that they show textured rooms. Automated: unique-colour count, image entropy and edge density far above the mock (the mock's identical 2,627-byte frames are the baseline to beat); start-frame hashes differ across scenes; fraction of zero-depth pixels recorded per scene (Matterport meshes have holes).
- **G3.6**: Mock locked out: with habitat deliberately broken (for example, a bad import), a real run aborts instead of falling back.
- **G3.7**: Concurrency: run the simulator and the chosen model together for 10 minutes and watch `nvidia-smi`. No out-of-memory errors; if there are, reduce model size, `num_ctx` or render resolution and record the change.  
**Gate**: G3.2 and G3.4 green. If not, do not start Phase 4.

---

## Phase 4. Embodiment adapter and geometry
**Brief**. A `HabitatEmbodiment` with the same interface as the mock, with every coordinate convention proven by test.  
**Do**:
- Sensors: four RGB-D cameras at the same position and height, yaw offsets 0, -90, 180 and +90 degrees (per the ledger), identical intrinsics. Actions: `move_forward`, `turn_left`, `turn_right` as in the ledger.
- Handle Habitat conventions explicitly: the world frame is y-up with -z forward at identity rotation; depth may be normalised to a `[min, max]` range (convert to metres with the configured range); decide whether depth is z-depth or ray length.
- Pose source (commanded odometry or simulator state) follows the paper and is written to the ledger.  
**Checks**:
- **G4.1**: Intrinsics: $f_x = (W/2) / \tan(\text{HFOV}/2)$ matches what the sensor really uses (320 px for 640 px and 90 degrees). Verify by projecting a known 3D point.
- **G4.2**: Depth truth: face a flat wall. Measure the true perpendicular distance independently by stepping forward until collision and counting 0.25 m steps. Centre-pixel depth within 2% (proposed). Across a row of wall pixels, depth is constant if z-depth and grows toward the edges if ray length; confirm which, and make the deprojection match.
- **G4.3**: View consistency: the camera named left at heading $\theta$ must match the camera named front at heading $\theta + 90^\circ$ (likewise right and back). Compare with SSIM or pixel difference; near-identical is required. This verifies camera yaw signs, view labels and turn direction at once.
- **G4.4**: Action primitives: an unobstructed `move_forward` displaces 0.25 m within tolerance; a turn changes heading by 15 degrees; 24 consecutive turns return to the original heading; collision and sliding behaviour is recorded in the ledger.
- **G4.5**: Odometry drift: 100 random rollouts of 100 steps. Compare the agent's pose estimate with simulator ground truth. Expect zero drift without collisions; measure and report divergence after collisions.
- **G4.6**: Round trip: deproject pixels to world points, re-project into a second camera pose; error <= 2 px (proposed) wherever depth is valid.
- **G4.7**: Determinism: the same pose and seed give the same image hash, or a recorded tolerance.
- **G4.8**: Simultaneity: the four views are captured at the same pose (no motion between captures).  
**Gate**: G4.1 to G4.8 green on at least three different scenes.

---

## Phase 5. Memory, map and history on real data
**Brief**. Re-verify the occupancy grid, BEV renderer, history buffer and image cache on real depth, since the earlier 'verified' status came from the mock.  
**Checks**:
- **G5.1**: Trajectory consistency: walk a scripted loop in 3 scenes and build the grid from depth. Every cell under the walked trajectory must be free (100%). Separately, build a ground-truth occupancy from the navmesh at the same resolution (test-only code, never imported by agent code) and report IoU for free and obstacle classes. Set IoU thresholds after the first measurement, then freeze them as regression baselines.
- **G5.2**: Height band: on flat floors the floor is never marked as an obstacle and ceilings are excluded. Test one scene with stairs or ramps and record the behaviour.
- **G5.3**: Depth holes: zero or invalid depth becomes unknown. It must not become an obstacle at range 0 and must not carve free space out to the maximum range (follow the ledger).
- **G5.4**: Recentering: walk past the 32 px margin. A fixed landmark must move in the grid by exactly the roll amount, with no content lost inside the canvas.
- **G5.5**: BEV orientation: confirm what 'up' means (north-up or heading-up) and that the renderer and prompt text agree. Turn the agent by 90 degrees and check the arrow rotates the expected way. Decision-node labels on the map must equal the recall node indices and the history text.
- **G5.6**: History buffer: after more than K steps only the last K remain; each entry holds action, commanded view and rationale, in order.
- **G5.7**: Image cache: the RGBs stored for node $t$ hash-match what was sent to the model at step $t$. Memory footprint per episode logged.
- **G5.8**: Per-step map update time is flat across a long episode (no growth).  
**Gate**: all green on 3 scenes.

---

## Phase 6. Tool harness verification (simulator live, scripted calls, no VLM)
**Brief**. Prove every tool, budget and safety rule on real scenes with deterministic pytest tests (marker: `sim`). Each tool gets positive, boundary and negative tests, and every numeric result is checked against an independent reference computed from simulator ground truth, not from the harness's own code.

### 6.1 `query_depth`
- **G6.1a**: Accuracy: 20 wall pixels per scene at independently measured distances; returned median within 2% (proposed).
- **G6.1b**: Window logic: the 5x5 median matches direct numpy indexing; one hole pixel inside the window does not corrupt the result (per ledger policy); pixels at the image border clip the window correctly.
- **G6.1c**: Holes: a window that is entirely invalid returns an explicit 'invalid' result, never 0.0 m.
- **G6.1d**: Units and fields: metres; median, min and mean present as specified; rounding as specified.
- **G6.1e**: Negatives: bad view index; u or v out of range or negative; float versus int; NaN; string; missing key; more points than the allowed maximum. Each returns a structured error, never a crash, and never consumes budget incorrectly.
- **G6.1f**: Budget: one evidence call is counted exactly once against E.

### 6.2 `recall`
- **G6.2a**: Causality: only $1 \le \text{node} < t$ is allowed. Try every $n$ from 0 to $t+5$ plus negative and non-integer values; only the valid range succeeds and future frames are never retrievable.
- **G6.2b**: Fidelity: the returned image bytes hash-match what was stored at that step.
- **G6.2c**: The view argument is validated; the returned image is labelled with node and view so the model cannot confuse it with the live views.
- **G6.2d**: Node indices agree with the BEV labels and the history text.
- **G6.2e**: Reselect attempts do not create new nodes.

### 6.3 `move_to`
- **G6.3a**: Stage-by-stage: for 30 pixels per scene, a reference implementation inside the test computes the deprojected chassis point, polar $(\theta, \rho)$, standoff-subtracted $\rho$, quantised turns and forward steps. The harness must match the reference exactly (integers).
- **G6.3b**: Execution truth: with a clear corridor, the simulator's ground-truth displacement and heading after execution match the plan within one primitive step.
- **G6.3c**: Edge cases, each with a defined and logged outcome: target in the back view (about 12 turns); $\theta$ at the $\pm 180^\circ$ wrap; $\rho$ smaller than the standoff (turn only, or no move: follow the ledger); rounding ties ($7.5^\circ$, $0.125\text{m}$); pixel at the top of the image (ceiling or far); pixel on a depth hole; pixel on glass, mirror or door frame; target beyond maximum depth; floor pixel versus wall pixel in the same direction.
- **G6.3d**: Every outcome is one of `executed`, `reselect` or `error`, and is logged.

### 6.4 Corridor checker
- **G6.4a**: Boundary tests on crafted scenes or depth images: obstacle 0.4 m ahead blocks; lateral obstacle just inside the half-width blocks and just outside passes (for a 0.96 m corridor, test 0.47 m and 0.50 m); a low obstacle below the height band is ignored; an overhead obstacle above the band is ignored; an obstacle beyond the target is ignored.
- **G6.4b**: Reality check: 500 random proposals across scenes. Compare the checker's verdict with what the simulator actually did (collision flag, or executed displacement shorter than planned). Report the confusion matrix. Requirement: zero false-clears in nominal conditions (investigate each one); the false-block rate is reported and recorded in the ledger.
- **G6.4c**: Depth holes inside the corridor are handled per the ledger (conservative by default).

### 6.5 Reselect loop
- **G6.5**: At most $R$ reselects are enforced; the feedback message contains the distance to the blocking obstacle; counters reset at each decision step; after $R$ is exhausted the ledger's fallback is used; an infinite loop is impossible (hard step cap).

### 6.6 `stop`
- **G6.6**: `stop` ends the episode; later actions are rejected; the reason and final pose are logged; metrics are computed only after `stop`.

### 6.7 Turn state machine and budgets
- **G6.7**: $E \le 4$ evidence calls per decision, with a defined forced-commit message at $E+1$. Defined and tested handling of: two tool calls in one assistant message, unknown tool names, malformed arguments, empty responses, text-only responses. Global step cap and wall-clock timeout enforced.

### 6.8 Privilege firewall
- **G6.8a**: Static: an AST or grep test proves the agent package never imports evaluation or ground-truth modules, and the prompt builder never reads goal fields.
- **G6.8b**: Canary: put unique sentinel numbers in the episode's goal coordinates and shortest-path length. Run 3 episodes with the real model. Search every prompt, tool result and log shown to the model for the sentinels; zero hits required.
- **G6.8c**: Navmesh and ground-truth helpers exist only under `tests/` and `eval/`.

### 6.9 Replay
- **G6.9**: Every tool call and state transition is logged. `replay.py` re-executes the logged commands and reproduces the final pose and the BEV hash exactly.  
**Gate**: 100% of deterministic tests green; zero unexplained false-clears; coverage report saved.

---

## Phase 7. VLM tool-call reliability (model in the loop)
**Brief**. Now that the simulator and harness are proven, measure how reliably the model uses the tools. All thresholds below are `(proposed)` and are recorded in the ledger.  

### 7A. Offline states
Sample 60 real states from dev scenes (stratified: corridors, doorways, stairs, open rooms, dead ends). For each, send the real payload and record the model's response.
- **G7.1**: Schema-valid tool call >= 98% on the first attempt, 100% after the ledger's retry policy.
- **G7.2**: Valid view index and in-bounds pixel >= 99%; zero unhandled unknown tool names.
- **G7.3**: Pixel sanity: overlay the chosen pixel and the model's reason on the RGB image; human-review 30 samples; >= 80% land on the thing the reason describes. The coordinate convention must match G2.6.
- **G7.4**: Evidence behaviour: log the distribution of evidence calls per step; `query_depth` arguments valid; `recall` requests obey causality; no budget overruns reach the harness unhandled.
- **G7.5**: Blind controls: repeat 7A with (i) all-black images and (ii) images taken from a different state. Decisions and pixel choices must degrade or change. If they do not, the model is not using vision; stop and investigate.
- **G7.6**: Error recovery: inject a blocked-corridor reselect message; >= 90% of responses pick a different target within 2 tries.
- **G7.7**: Tokens and latency per turn logged and within the G2.8 estimate.

### 7B. Closed loop on 5 dev episodes
Review every step by hand: the four RGBs, the BEV, the prompt, the tool calls and the executed motion.

### 7C. Shakedown on 20 dev episodes
No crashes or aborts; deadlock detector (same pose repeated N times) wired in; aggregate failure taxonomy. Log every prompt or harness change and re-run 7A after each change.  
**Gate**: thresholds met on the primary model. If 4B fails but 9B passes, record that and make 9B primary despite the slower runs.

---

## Phase 8. Single-episode audit and provenance
**Brief**. Prove, on one full episode, that every claim in the manifest is true.
- **G8.1**: Frame-by-frame audit of one shakedown episode. At every step confirm: 4 distinct RGB frames and one BEV were attached (verify by `prompt_eval_count` jump and the saved payload hashes), the tool calls are logged, the executed action matches the committed tool call, and the agent's pose matches simulator ground truth.
- **G8.2**: Cross-episode uniqueness: start frames and file sizes differ across episodes and scenes (the earlier identical 2,627-byte frames must be impossible).
- **G8.3**: `manifest.json` is complete and every field verifiable against the live system.
- **G8.4**: Final metrics for the episode are computed only after `stop` and match a hand calculation from the logged trajectory and the geodesic distance.

---

## Phase 9. Metrics validation and benchmark
### 9.1 Metric implementation validation
- **G9.1**: Your NE, SR, SPL and OSR match the `habitat-lab` measures (distance to goal, success, SPL) on identical trajectories, with geodesic distances from the same navmesh.
- **G9.2**: Sanity baselines on the 100 locked episodes (run with no model): an oracle shortest-path follower gives SR near 100% and SPL near 1 (never above 1); a stop-immediately agent gives SR equal to the fraction of episodes whose start is already within the success radius; a random walker gives low SR. Any mismatch means a metric bug.
- **G9.3**: Episodes with unreachable or degenerate goals are identified and handled per the ledger.

### 9.2 Config freeze
- **G9.4**: Hash the ledger, prompts and config; tag the commit. No edits after this point.

### 9.3 Local-versus-cluster parity
- **G9.5**: Run the same model digest the cluster used (`qwen3-vl:8b`) on the same episodes locally. Compare SR, SPL and NE with Wilson confidence intervals. Trajectories need not match step by step (model nondeterminism), but the metrics must agree within the intervals. If quantisation or tag differs from the cluster, record it.

### 9.4 The benchmark
- **G9.6**: Run the locked 100 episodes with the primary model, with 3 seeds if time allows. Report mean and 95% confidence interval, per-scene breakdown and the failure taxonomy (premature stop, overshoot, collision, deadlock, budget). Manually audit at least 20 failed episodes to confirm the taxonomy labels.
- **G9.7**: Results are resume-safe: one result file per episode, completed episodes skipped on restart.
- **G9.8**: Report against the open-weight rows chosen in G0.4. State plainly that the Gemini headline numbers are not the target.

---

## Phase 10. Freeze and regression
- **G10.1**: Phase 4 to 6 tests run with one command (`pytest -m sim`) and pass.
- **G10.2**: Three golden episodes with stored trajectories; a replay must reproduce them exactly.
- **G10.3**: A smoke run of 3 dev episodes after any dependency, driver, Ollama or model change.
- **G10.4**: Lock files, ledger, manifests and audit folders archived with the release tag.

---

## Appendix A. Risk register
- WSL2 rendering fails (EGL, windowless context): detected by G3.2; mitigation is the fallback tree in Appendix C.
- VRAM contention between Ollama and the simulator: detected by G2.8 and G3.7; mitigate with a smaller model, smaller `num_ctx` or lower render resolution, recorded in the ledger.
- Model silently text-only: G2.1 and G2.4.
- Context truncation drops instructions or images: G2.5.
- Pixel coordinate convention mismatch: G2.6 and G7.3.
- Depth convention (z-depth versus ray length, normalisation): G4.2.
- Camera yaw sign or view-label error: G4.3.
- Mesh holes and zero depth: G3.5 and G5.3.
- Prompt overfitting to the test episodes: pool separation in 2.5 and the freeze in G9.4.
- Model nondeterminism: G2.9, seeds, repeated runs, confidence intervals.
- Laptop GPU thermal throttling in long runs: log clocks and temperature; allow cooldown; compare latency early versus late.
- Ollama unloading the model between steps: set `keep_alive` and check for latency spikes.
- Windows sleep or updates interrupting runs: resume-safe result files (G9.7).

---

## Appendix B. Repository layout
- `spec/spec_ledger.yaml`
- `scripts/vlm_battery.py` and `scripts/replay.py`
- `tests/sim/` (Phase 4 to 6 tests) and `tests/unit/`
- `eval/` (metrics and ground-truth helpers; never imported by the agent)
- `audit/phase0` ... `audit/phase10` (evidence)
- `runs/<run_id>/manifest.json` and per-episode result files

---

## Appendix C. Simulator fallback tree (only if G3.2 fails)
- **F1 (preferred)**: native Ubuntu on this laptop (dual boot or an external SSD). Habitat-Sim is a Linux simulator and this removes the WSL2 graphics layer.
- **F2 (hybrid)**: keep the simulator on the cluster and reach your local Ollama through an SSH reverse tunnel, or run the model on the cluster too. The pipeline is identical; only the host changes.
- **F3**: a rented Linux GPU machine.
- Whichever fallback is used, Phases 3 to 9 apply unchanged, including the parity render in G3.4.
- **Never acceptable**: the 2D mock, or a pyrender/trimesh renderer, as a stand-in for the simulator.

---

## Appendix D. Execution order
Phase 0, then Phase 1, then Phase 2 (model) and Phase 3 (simulator) in parallel, then Phases 4, 5, 6 in order, then Phase 7, then Phase 8, then Phase 9, then Phase 10. A failed gate stops the chain until it is fixed and the phase is re-run in full.

