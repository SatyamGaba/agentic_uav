import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

remote_code = '''import json
from framework.backends.ollama_client import OllamaVLMBackend
from framework.environments.openfly_env import OpenFlyBenchmarkEnv

vlm = OllamaVLMBackend(model_name="qwen3-vl:8b", timeout_s=120.0, num_ctx=8192)
env = OpenFlyBenchmarkEnv("benchmark/seen.json")
goal = env.load_episode_goal(0)
obs = env.reset(goal)

tools = [
    {
        "type": "function",
        "function": {
            "name": "plan_flight",
            "description": "Execute an aerial flight plan composed of discrete OpenFly action integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_codes": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Sequence of OpenFly action integers: 9 (forward 9m), 8 (forward 6m), 1 (forward 3m), 3 (turn left -30 deg), 2 (turn right +30 deg), 4 (climb 3m), 5 (descend 3m), 0 (stop landmark reached)."
                    }
                },
                "required": ["action_codes"]
            }
        }
    }
]

prompt = f'You are an aerial drone navigation brain. Instruction: "{obs.instruction}". Call plan_flight function with action_codes list ending with 0.'
msg = {"role": "user", "content": prompt, "images": [obs.rgb_b64]}
resp = vlm.chat(messages=[msg], tools=tools, max_tokens=4096)
print("DONE REASON:", resp.raw_response.get("done_reason"))
print("TOOL CALLS:", resp.tool_calls)
print("CONTENT:", resp.content)
print("LATENCY:", resp.latency_s)
'''

cmd = f"cat << 'EOF' > /home/jovyan/scratch/test_full_vlm.py\n{remote_code}\nEOF\ncd /home/jovyan/scratch && python /home/jovyan/scratch/test_full_vlm.py"
print(exec_cmd.run_remote(cmd, wait_sec=60.0))
