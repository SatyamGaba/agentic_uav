import sys, os, json, re
from typing import Dict, Any, List

def normalize_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name") or call.get("function", {}).get("name") or call.get("action")
    args = call.get("arguments") or call.get("function", {}).get("arguments") or call.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not args:
        top_args = {k: v for k, v in call.items() if k not in ("name", "function", "type", "arguments", "action", "parameters")}
        if top_args:
            args = top_args
    return {"name": name, "arguments": args}

def extract_balanced_json(text: str) -> List[Dict[str, Any]]:
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            start = i
            depth = 1
            i += 1
            while i < n and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                candidate = text[start:i]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and any(k in parsed for k in ("name", "action", "function")):
                        results.append(parsed)
                except Exception:
                    pass
        else:
            i += 1
    return results

def enhanced_parse_tool_calls_from_text(raw_text: str) -> List[Dict[str, Any]]:
    if not raw_text:
        return []

    tool_calls = []

    # 1. XML tags: <tool_call> ... </tool_call>
    xml_matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_text, re.DOTALL)
    for m in xml_matches:
        try:
            parsed = json.loads(m.strip())
            if isinstance(parsed, dict):
                tool_calls.append(normalize_call(parsed))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        tool_calls.append(normalize_call(item))
        except Exception:
            pass
    if tool_calls:
        return tool_calls

    # 2. Markdown code fences: ```json ... ``` or ``` ... ```
    fence_matches = re.findall(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", raw_text, re.DOTALL)
    for fm in fence_matches:
        try:
            parsed = json.loads(fm.strip())
            if isinstance(parsed, dict):
                tool_calls.append(normalize_call(parsed))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        tool_calls.append(normalize_call(item))
        except Exception:
            pass
    if tool_calls:
        return tool_calls

    # 3. Balanced JSON blocks
    json_blocks = extract_balanced_json(raw_text)
    for block in json_blocks:
        norm = normalize_call(block)
        if norm.get("name") in ("query_depth", "recall", "move_to", "stop"):
            tool_calls.append(norm)
    if tool_calls:
        return tool_calls

    # 4. Functional syntax fallback (stop or move_to or query_depth)
    stop_match = re.search(r"\bstop\s*\((.*?)\)", raw_text, re.DOTALL | re.IGNORECASE)
    if stop_match:
        reason_str = stop_match.group(1).strip()
        reason = ""
        rm = re.search(r'reason\s*=\s*["\'](.*?)["\']', reason_str, re.DOTALL)
        if rm:
            reason = rm.group(1)
        elif reason_str:
            reason = reason_str.strip('"\'')
        return [{"name": "stop", "arguments": {"reason": reason}}]

    move_match = re.search(r"\bmove_to\s*\((.*?)\)", raw_text, re.DOTALL | re.IGNORECASE)
    if move_match:
        args_str = move_match.group(1)
        args = {}
        # View
        vm = re.search(r'view\s*=\s*([0-3])', args_str, re.IGNORECASE)
        if vm:
            args["view"] = int(vm.group(1))
        # u
        um = re.search(r'u\s*=\s*([0-9.]+)', args_str, re.IGNORECASE)
        if um:
            args["u"] = float(um.group(1))
        # v
        vm2 = re.search(r'v\s*=\s*([0-9.]+)', args_str, re.IGNORECASE)
        if vm2:
            args["v"] = float(vm2.group(1))
        # reason
        rm = re.search(r'reason\s*=\s*["\'](.*?)["\']', args_str, re.DOTALL | re.IGNORECASE)
        if rm:
            args["reason"] = rm.group(1)
        if "view" in args:
            if "u" not in args: args["u"] = 0.5
            if "v" not in args: args["v"] = 0.55
            return [{"name": "move_to", "arguments": args}]

    # 5. Key-Value block fallback:
    # Action: move_to / stop
    act_m = re.search(r'(?:Action|Tool):\s*(move_to|stop|query_depth|recall)', raw_text, re.IGNORECASE)
    if act_m:
        act_name = act_m.group(1).lower()
        if act_name == "stop":
            rm = re.search(r'reason:\s*(.*)', raw_text, re.IGNORECASE)
            return [{"name": "stop", "arguments": {"reason": rm.group(1).strip() if rm else ""}}]
        elif act_name == "move_to":
            args = {}
            vm = re.search(r'view:\s*([0-3])', raw_text, re.IGNORECASE)
            if vm: args["view"] = int(vm.group(1))
            um = re.search(r'u:\s*([0-9.]+)', raw_text, re.IGNORECASE)
            if um: args["u"] = float(um.group(1))
            vm2 = re.search(r'v:\s*([0-9.]+)', raw_text, re.IGNORECASE)
            if vm2: args["v"] = float(vm2.group(1))
            rm = re.search(r'reason:\s*(.*)', raw_text, re.IGNORECASE)
            if rm: args["reason"] = rm.group(1).strip()
            if "view" in args:
                if "u" not in args: args["u"] = 0.5
                if "v" not in args: args["v"] = 0.55
                return [{"name": "move_to", "arguments": args}]

    return []

samples = [
    # 1. Markdown JSON fence
    '''I will move forward.
```json
{
  "name": "move_to",
  "arguments": {
    "view": 0,
    "u": 0.5,
    "v": 0.55,
    "reason": "move along corridor"
  }
}
```''',

    # 2. Raw JSON without fence
    '''Based on the BEV map, I will navigate down the corridor:
{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.55, "reason": "move along corridor"}}''',

    # 3. XML tool call
    '''<tool_call>
{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.55}}
</tool_call>''',

    # 4. Functional syntax with reason
    '''Action: move_to(view=0, u=0.5, v=0.55, reason="Moving forward toward kitchen")''',

    # 5. Stop functional call
    '''All milestones satisfied. stop(reason="Reached goal")''',

    # 6. Stop JSON
    '''{"name": "stop", "arguments": {"reason": "Reached target table"}}''',

    # 7. Action block: Action: move_to\nView: 0\nCoordinates: (0.5, 0.55)
    '''I'll proceed to the kitchen:
Action: move_to
view: 0
u: 0.5
v: 0.55
reason: Proceeding forward''',
]

for i, s in enumerate(samples, 1):
    res = enhanced_parse_tool_calls_from_text(s)
    status = "SUCCESS" if res else "FAILED"
    print(f"Sample {i}: -> {status} | Parsed: {res}")
