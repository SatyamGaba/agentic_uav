import os
import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_benchmark_run.log"), mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("FullBenchmarkRunner")

from benchmark_vln import ValidatedOpenFlyEvaluator

def run_all():
    provider = "lmstudio"
    model_name = "qwen/qwen3-vl-4b"
    
    logger.info("==========================================================================")
    logger.info("STARTING FULL OPENFLY BENCHMARK EVALUATION (3,000 EPISODES)")
    logger.info(f"Model: {model_name} via {provider}")
    logger.info("==========================================================================")
    
    evaluator = ValidatedOpenFlyEvaluator(provider=provider, model_name=model_name)
    
    # 1. Evaluate Test-Seen Split (All 1,800 episodes)
    logger.info(">>> PART 1: Evaluating Test-Seen Split (1,800 trajectories)...")
    seen_results = evaluator.evaluate(split_name="seen", max_episodes=1800)
    
    # 2. Evaluate Test-Unseen Split (All 1,200 episodes)
    logger.info(">>> PART 2: Evaluating Test-Unseen Split (1,200 trajectories)...")
    unseen_results = evaluator.evaluate(split_name="unseen", max_episodes=1200)
    
    # 3. Aggregate Final Master Report
    summary = {
        "benchmark": "OpenFly (ICLR 2026) - Complete 3,000 Trajectories",
        "model": model_name,
        "provider": provider,
        "test_seen": seen_results["metrics"],
        "test_unseen": unseen_results["metrics"]
    }
    
    master_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_full_benchmark.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("==========================================================================")
    logger.info("ALL 3,000 BENCHMARK TESTS COMPLETE!")
    logger.info(f"Test-Seen   -> SR: {seen_results['metrics']['SR (%) ↑']}% | NE: {seen_results['metrics']['NE (meters) ↓']}m | SPL: {seen_results['metrics']['SPL (%) ↑']}%")
    logger.info(f"Test-Unseen -> SR: {unseen_results['metrics']['SR (%) ↑']}% | NE: {unseen_results['metrics']['NE (meters) ↓']}m | SPL: {unseen_results['metrics']['SPL (%) ↑']}%")
    logger.info(f"Saved master results to: {master_path}")
    logger.info("==========================================================================")

if __name__ == "__main__":
    run_all()
