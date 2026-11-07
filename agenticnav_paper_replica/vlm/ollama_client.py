"""Local Ollama VLM Client for AgenticNav (Qwen-VL, LLaVA, etc.).
Runs locally on the remote server GPU without API keys or internet access.
"""
import base64
import io
import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image


def encode_image_to_base64(img: Any) -> str:
    """Encodes PIL Image or numpy array to JPEG base64 string."""
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img, mode="L")
        else:
            pil_img = Image.fromarray(img, mode="RGB")
    elif isinstance(img, Image.Image):
        pil_img = img
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _normalize_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name") or call.get("function", {}).get("name")
    args = call.get("arguments") or call.get("function", {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not args:
        top_args = {k: v for k, v in call.items() if k not in ("name", "function", "type", "arguments")}
        if top_args:
            args = top_args
    return {"name": name, "arguments": args}


def _extract_balanced_json(text: str) -> List[Dict[str, Any]]:
    """Extracts all top-level balanced JSON objects from raw text."""
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


def parse_tool_calls_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """Extracts tool calls from XML tags, markdown fences, raw JSON, or functional syntax."""
    if not raw_text:
        return []

    tool_calls = []

    # 1. XML tags: <tool_call> ... </tool_call>
    xml_matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_text, re.DOTALL)
    for m in xml_matches:
        try:
            parsed = json.loads(m.strip())
            if isinstance(parsed, dict):
                tool_calls.append(_normalize_call(parsed))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        tool_calls.append(_normalize_call(item))
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
                tool_calls.append(_normalize_call(parsed))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        tool_calls.append(_normalize_call(item))
        except Exception:
            pass
    if tool_calls:
        return tool_calls

    # 3. Balanced JSON blocks
    json_blocks = _extract_balanced_json(raw_text)
    for block in json_blocks:
        norm = _normalize_call(block)
        if norm.get("name") in ("query_depth", "recall", "move_to", "stop"):
            tool_calls.append(norm)
    if tool_calls:
        return tool_calls

    # 4. Functional syntax fallback (stop, move_to, query_depth)
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
        vm = re.search(r'view\s*=\s*([0-3])', args_str, re.IGNORECASE)
        if vm: args["view"] = int(vm.group(1))
        um = re.search(r'u\s*=\s*([0-9.]+)', args_str, re.IGNORECASE)
        if um: args["u"] = float(um.group(1))
        vm2 = re.search(r'v\s*=\s*([0-9.]+)', args_str, re.IGNORECASE)
        if vm2: args["v"] = float(vm2.group(1))
        rm = re.search(r'reason\s*=\s*["\'](.*?)["\']', args_str, re.DOTALL | re.IGNORECASE)
        if rm: args["reason"] = rm.group(1)
        if "view" in args:
            if "u" not in args: args["u"] = 0.5
            if "v" not in args: args["v"] = 0.55
            return [{"name": "move_to", "arguments": args}]

    # 5. Key-Value block fallback
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


class OllamaNavClient:
    """Client for local VLM inference on NVIDIA GPU via Ollama."""

    def __init__(
        self,
        model_name: str = "qwen3.5:4b",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        seed: int = 42,
        thinking: bool = False,
        timeout_s: float = 60.0,
        num_ctx: int = 8192,
        supports_vision: Optional[bool] = None,
    ):
        self.model_name = model_name
        
        # Auto-detect Ollama URL
        if base_url is None:
            import os
            import subprocess
            env_url = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL")
            if env_url:
                if not env_url.startswith("http"):
                    env_url = f"http://{env_url}"
                base_url = env_url
            else:
                # Check WSL default gateway if running under Linux
                gateway_ip = None
                if os.path.exists("/proc/version"):
                    try:
                        res = subprocess.run(
                            ["ip", "route", "show", "default"],
                            capture_output=True,
                            text=True,
                            timeout=1.0,
                        )
                        parts = res.stdout.strip().split()
                        if "via" in parts:
                            gateway_ip = parts[parts.index("via") + 1]
                    except Exception:
                        pass
                if gateway_ip:
                    base_url = f"http://{gateway_ip}:11434"
                else:
                    base_url = "http://127.0.0.1:11434"

        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.thinking = thinking
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx

        # Verify vision capability on server (FAIL FAST: NO SILENT TEXT-ONLY FALLBACK)
        if supports_vision is None:
            self.supports_vision = self._check_vision_capability()
        else:
            self.supports_vision = supports_vision

        if not self.supports_vision:
            raise RuntimeError(
                f"FATAL: Model '{self.model_name}' lacks verified vision capability on Ollama! "
                f"Silent text-only fallback is strictly prohibited per Constitutional Master Document."
            )

    def _check_vision_capability(self) -> bool:
        """Queries Ollama /api/show to verify native vision support."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/show",
                data=json.dumps({"model": self.model_name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                caps = data.get("capabilities", [])
                families = data.get("details", {}).get("families", [])
                m_info = str(data.get("model_info", {}))
                is_vis = ("vision" in caps) or ("clip" in families) or ("vision" in m_info) or ("qwen35" in str(families))
                return is_vis
        except Exception as e:
            # Fallback to model name heuristic if offline check fails
            m_lower = self.model_name.lower()
            return any(k in m_lower for k in ("qwen3.5", "qwen3-vl", "vl", "vision", "llava", "moondream"))

    def is_configured(self) -> bool:
        """Checks if the local Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/version", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Invoke local Ollama model on conversation messages."""
        ollama_msgs = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            img_b64_list = []

            # Multi-view RGB camera views
            rgb_views = msg.get("rgb_views")
            if rgb_views and isinstance(rgb_views, dict):
                for v_idx in sorted(rgb_views.keys()):
                    if rgb_views[v_idx] is not None:
                        img_b64_list.append(encode_image_to_base64(rgb_views[v_idx]))

            # BEV occupancy map
            bev_image = msg.get("bev_image")
            if bev_image is not None:
                img_b64_list.append(encode_image_to_base64(bev_image))

            # Recalled view from memory
            recalled_img = msg.get("recalled_image")
            if recalled_img is not None:
                img_b64_list.append(encode_image_to_base64(recalled_img))

            # If content is a list of parts (text + images)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image":
                            img_obj = part.get("image")
                            if img_obj is not None:
                                img_b64_list.append(encode_image_to_base64(img_obj))
                text_content = "\n".join(text_parts)
            else:
                text_content = str(content)

            # If role is "tool", convert to user feedback so Ollama never complains of orphaned tool IDs
            if role == "tool":
                tool_name = msg.get("name", "tool")
                text_content = f"[Tool Feedback from {tool_name}]:\n{text_content}"
                role = "user"

            # Attach images only if the model backend supports vision
            attach_imgs = img_b64_list if self.supports_vision else []

            # Merge consecutive messages with the same role to guarantee strict ChatML alternation
            if ollama_msgs and ollama_msgs[-1]["role"] == role:
                ollama_msgs[-1]["content"] += f"\n\n{text_content}"
                if attach_imgs:
                    existing_imgs = ollama_msgs[-1].get("images", [])
                    existing_imgs.extend(attach_imgs)
                    ollama_msgs[-1]["images"] = existing_imgs
            else:
                entry = {"role": role, "content": text_content}
                if attach_imgs:
                    entry["images"] = attach_imgs
                ollama_msgs.append(entry)

        payload = {
            "model": self.model_name,
            "messages": ollama_msgs,
            "stream": False,
            "keep_alive": -1,
            "think": self.thinking,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }

        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        max_retries = 3
        data = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(2.0 * (attempt + 1))

        message = data.get("message", {})
        raw_content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        # If model returned tool calls natively via Ollama API
        if tool_calls:
            parsed_tools = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                parsed_tools.append({"name": fn.get("name"), "arguments": args})
            return {"thought": raw_content, "tool_calls": parsed_tools}

        # Fallback: Extract tool calls from text (XML tags, code fences, raw JSON, functional calls)
        extracted = parse_tool_calls_from_text(raw_content)
        return {"thought": raw_content, "tool_calls": extracted}

