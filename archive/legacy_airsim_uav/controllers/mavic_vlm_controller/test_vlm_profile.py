import sys
sys.path.append('c:\\Users\\Kruti\\Desktop\\PROG\\UAV\\controllers\\mavic_vlm_controller')
from mavic_vlm_controller import VLMClient

vlm = VLMClient()
try:
    p1 = vlm.query_search_profile("windmill", "lmstudio", "", "qwen/qwen3-vl-4b")
    print(p1)
    p2 = vlm.query_search_profile("tesla", "lmstudio", "", "qwen/qwen3-vl-4b")
    print(p2)
except Exception as e:
    print(f"Exception: {e}")
