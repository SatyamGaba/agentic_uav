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
print("Turn 1 - Tool calls:", resp.tool_calls)
print("Turn 1 - Done reason:", resp.raw_response.get("done_reason"))

if not resp.tool_calls:
    print("Turn 1 did not return tool call. Testing Turn 2 follow-up...")
    msg2 = {"role": "user", "content": "Now output the action_codes JSON list ending with 0."}
    resp2 = vlm.chat(messages=[msg, {"role": "assistant", "content": resp.content or "Understood the landmarks."}, msg2], tools=tools, max_tokens=1024)
    print("Turn 2 - Tool calls:", resp2.tool_calls)
    print("Turn 2 - Content:", resp2.content)
'''

cmd = f"cat << 'EOF' > /home/jovyan/scratch/test_followup.py\n{remote_code}\nEOF\ncd /home/jovyan/scratch && python /home/jovyan/scratch/test_followup.py"
print(exec_cmd.run_remote(cmd, wait_sec=70.0))
