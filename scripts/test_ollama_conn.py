import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
client = OllamaNavClient(model_name="qwen3.5:4b")
print("Base URL:", client.base_url)
print("Configured:", client.is_configured())

