"""OpenAI-compatible / vLLM remote client for AgenticNav with privilege firewall.

Supports remote vLLM, SGLang, and OpenAI-compatible inference servers running larger
models (e.g. Qwen2.5-32B-Instruct, Qwen2.5-VL-72B, Qwen3.6-27B).
"""

import base64
import io
import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from agenticnav_paper_replica.vlm.firewall import assert_firewall_integrity


def encode_image_to_jpeg_data_url(img: Any) -> str:
    """Encodes PIL Image or numpy array to data:image/jpeg;base64,... URL."""
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
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _normalize_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name") or call.get("function", {}).get("name")
    args = call.get("arguments") or call.get("function", {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return {"name": name, "arguments": args}


class OpenAIVLLMNavClient:
    """Client for vLLM / OpenAI-compatible remote endpoints with privilege firewall."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Format request, verify privilege firewall, and dispatch to remote endpoint."""
        openai_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            rgb_views = msg.get("rgb_views")
            bev_image = msg.get("bev_image")

            if role == "system":
                openai_messages.append({"role": "system", "content": content})
            elif role == "user":
                parts: List[Dict[str, Any]] = [{"type": "text", "text": content}]
                if rgb_views:
                    for v_idx in sorted(rgb_views.keys()):
                        data_url = encode_image_to_jpeg_data_url(rgb_views[v_idx])
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        })
                if bev_image is not None:
                    data_url = encode_image_to_jpeg_data_url(bev_image)
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
                openai_messages.append({"role": "user", "content": parts})
            elif role in ("assistant", "model"):
                openai_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id", "call_0"),
                })

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }

        if tools:
            # Format as standard OpenAI function tools
            formatted_tools = []
            for t in tools:
                if "function" in t:
                    formatted_tools.append(t)
                else:
                    formatted_tools.append({"type": "function", "function": t})
            payload["tools"] = formatted_tools
            payload["tool_choice"] = "auto"

        payload_json = json.dumps(payload)

        # STRICT PRIVILEGE FIREWALL CHECK
        assert_firewall_integrity(payload_json)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload_json.encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        raw_text = msg.get("content") or ""
        tool_calls = []

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(_normalize_call(tc))

        return {
            "thought": raw_text,
            "raw_text": raw_text,
            "tool_calls": tool_calls,
        }

