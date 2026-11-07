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
goal = env.load_episode_goal(2)
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
print("Ep 2 - Turn 1 Done:", resp.raw_response.get("done_reason"))
print("Ep 2 - Turn 1 Tool calls:", resp.tool_calls)
print("Ep 2 - Turn 1 Thinking len:", len(resp.thinking))
print("Ep 2 - Turn 1 Content len:", len(resp.content))
if not resp.tool_calls:
    print("Ep 2 - Turn 1 Thinking snippet:", resp.thinking[:300])
    print("Ep 2 - Turn 1 Thinking end:", resp.thinking[-300:])
    follow_up_user = {"role": "user", "content": "Now output the action_codes JSON list ending with 0."}
    history = [msg, {"role": "assistant", "content": resp.content or "Reasoning completed."}, follow_up_user]
    resp2 = vlm.chat(messages=history, tools=tools, max_tokens=1024)
    print("Ep 2 - Turn 2 Done:", resp2.raw_response.get("done_reason"))
    print("Ep 2 - Turn 2 Tool calls:", resp2.tool_calls)
    print("Ep 2 - Turn 2 Thinking len:", len(resp2.thinking))
    print("Ep 2 - Turn 2 Content:", resp2.content)
    print("Ep 2 - Turn 2 Thinking snippet:", resp2.thinking[:300])
'''

cmd = f"cat << 'EOF' > /home/jovyan/scratch/debug_ep2.py\n{remote_code}\nEOF\ncd /home/jovyan/scratch && python /home/jovyan/scratch/debug_ep2.py"
print(exec_cmd.run_remote(cmd, wait_sec=70.0))
