import os
import sys
import json
import logging
import time
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmark/docker_benchmark_run.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("DockerBenchmarkRunner")

# Add benchmark directory to path
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from benchmark_vln import ValidatedOpenFlyEvaluator
from download_dataset import download_annotations

def wait_for_vlm_server(api_url: str, max_retries: int = 30):
    """Waits until the VLM engine is responsive."""
    logger.info(f"Checking VLM Engine health at {api_url}...")
    base_url = api_url.split("/v1")[0]
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                logger.info("VLM Engine is HEALTHY and ready to receive requests!")
                return True
        except Exception:
            pass
        logger.info(f"Waiting for VLM server to initialize (attempt {attempt+1}/{max_retries})...")
        time.sleep(5)
    raise RuntimeError("VLM server failed to become ready within timeout period.")

def main():
    logger.info("==========================================================================")
    logger.info("STARTING 100% CONTAINERIZED OPENFLY BENCHMARK (ALL 3,000 EPISODES)")
    logger.info("==========================================================================")
    
    # 1. Download OpenFly benchmark annotations if not present
    download_annotations()
    
    # 2. Read VLM endpoint from environment
    vlm_api_url = os.environ.get("VLM_API_URL", "http://vlm-engine:11434/v1/chat/completions")
    model_name = os.environ.get("MODEL_NAME", "qwen3-vl:8b")
    
    # 3. Wait for VLM container
    wait_for_vlm_server(vlm_api_url)
    
    # 4. Initialize Evaluator
    evaluator = ValidatedOpenFlyEvaluator(provider="lmstudio", model_name=model_name)
    
    # Point internal client to the container network URL if needed
    if evaluator.vlm_client:
        evaluator.vlm_client.custom_endpoint = vlm_api_url
        
    # 5. Evaluate Test-Seen Split (All 1,800 trajectories)
    logger.info(">>> PART 1: Evaluating Test-Seen Split (1,800 trajectories)...")
    seen_results = evaluator.evaluate(split_name="seen", max_episodes=1800)
    
    # 6. Evaluate Test-Unseen Split (All 1,200 trajectories)
    logger.info(">>> PART 2: Evaluating Test-Unseen Split (1,200 trajectories)...")
    unseen_results = evaluator.evaluate(split_name="unseen", max_episodes=1200)
    
    # 7. Aggregate & Save Master Results
    summary = {
        "benchmark": "OpenFly (ICLR 2026) - 100% Containerized GPU Run",
        "model": model_name,
        "environment": "Docker (NVIDIA GPU Passthrough)",
        "test_seen": seen_results["metrics"],
        "test_unseen": unseen_results["metrics"]
    }
    
    output_path = os.path.join(BENCHMARK_DIR, "final_docker_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("==========================================================================")
    logger.info("ALL 3,000 CONTAINERIZED BENCHMARK TESTS COMPLETE!")
    logger.info(f"Test-Seen   -> SR: {seen_results['metrics']['SR (%) ↑']}% | NE: {seen_results['metrics']['NE (meters) ↓']}m | SPL: {seen_results['metrics']['SPL (%) ↑']}%")
    logger.info(f"Test-Unseen -> SR: {unseen_results['metrics']['SR (%) ↑']}% | NE: {unseen_results['metrics']['NE (meters) ↓']}m | SPL: {unseen_results['metrics']['SPL (%) ↑']}%")
    logger.info(f"Saved master results to: {output_path}")
    logger.info("==========================================================================")

if __name__ == "__main__":
    main()
