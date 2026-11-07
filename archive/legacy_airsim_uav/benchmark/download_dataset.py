import os
import json
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkDownloader")

HF_BASE_URL = "https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly/raw/main"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def download_file(rel_path: str, save_path: str):
    url = f"{HF_BASE_URL}/{rel_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        logger.info(f"File already exists: {save_path}")
        return
    logger.info(f"Downloading {url} -> {save_path}...")
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Successfully saved: {save_path}")
    else:
        logger.error(f"Failed to download {url}: HTTP {resp.status_code}")

def download_annotations():
    seen_path = os.path.join(DATA_DIR, "seen.json")
    unseen_path = os.path.join(DATA_DIR, "unseen.json")
    
    download_file("Annotation/seen.json", seen_path)
    download_file("Annotation/unseen.json", unseen_path)
    
    logger.info("Annotations ready.")

def load_test_split(split_name: str = "seen"):
    file_path = os.path.join(DATA_DIR, f"{split_name}.json")
    if not os.path.exists(file_path):
        download_annotations()
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    download_annotations()
