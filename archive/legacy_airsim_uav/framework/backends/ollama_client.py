"""
Robust Ollama VLM Backend for Local Multi-Modal Models (Qwen-VL, LLaVA, etc.).
Features:
- Persistent VRAM residency (keep_alive: -1)
- Context and generation token budgeting
- Native Ollama /api/chat tool-calling parser
- Fallback regex extractor for tool calls embedded in markdown content or <think> tags
"""
import time
import json
import re
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from framework.interfaces.base_vlm import BaseVLMBackend, VLMResponse


class OllamaVLMBackend(BaseVLMBackend):
    """
    Ollama multi-modal backend client with resilient tool-calling parsing.
    """
    def __init__(
        self,
        model_name: str = "qwen3-vl:8b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 120.0,
        keep_alive: Any = -1,
        num_ctx: int = 8192,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name=model_name, config=config)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> VLMResponse:
        t0 = time.time()
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            # Return error response container
            latency = time.time() - t0
            return VLMResponse(
                content="",
                thinking="",
                tool_calls=[],
                raw_response={"error": str(e)},
                latency_s=round(latency, 3),
                model=self.model_name
            )

        latency = time.time() - t0
        msg = result.get("message", {})
        raw_content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        native_tool_calls = msg.get("tool_calls", [])

        # Parse prompt / eval token statistics if returned by Ollama
        prompt_eval_count = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)

        # 1. Check native tool calls
        final_tool_calls = []
        if native_tool_calls:
            for tc in native_tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                final_tool_calls.append({
                    "id": tc.get("id", f"call_{len(final_tool_calls)}"),
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                })

        # 2. Resilient fallback: extract tool call embedded in content text or <think> block
        if not final_tool_calls:
            source = raw_content if raw_content.strip() else thinking
            if source:
                extracted = self._extract_embedded_tool_call(source, tools)
                if extracted:
                    final_tool_calls.append(extracted)

        return VLMResponse(
            content=raw_content,
            thinking=thinking,
            tool_calls=final_tool_calls,
            raw_response=result,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            latency_s=round(latency, 3),
            model=self.model_name
        )

    def _extract_embedded_tool_call(self, text: str, tools: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Extracts JSON tool calls emitted directly inside markdown text or codeblocks."""
        valid_fn_names = [t["function"]["name"] for t in tools] if tools else ["move_to", "query_depth", "stop_arrival"]
        for fn_name in valid_fn_names:
            # Pattern: {"name": "move_to", "arguments": {...}} or `move_to(u=..., v=...)`
            pat_json = rf'\{{\s*"name"\s*:\s*"{fn_name}"\s*,\s*"arguments"\s*:\s*(\{{.*?\}})\s*\}}'
            match = re.search(pat_json, text, re.DOTALL)
            if match:
                try:
                    args = json.loads(match.group(1))
                    return {
                        "id": f"extracted_{fn_name}",
                        "type": "function",
                        "function": {"name": fn_name, "arguments": args}
                    }
                except Exception:
                    pass

            # Pattern: function call inside markdown block
            pat_direct = rf'"{fn_name}"\s*:\s*(\{{.*?\}})'
            match2 = re.search(pat_direct, text, re.DOTALL)
            if match2:
                try:
                    args = json.loads(match2.group(1))
                    return {
                        "id": f"extracted_{fn_name}",
                        "type": "function",
                        "function": {"name": fn_name, "arguments": args}
                    }
                except Exception:
                    pass

        return None

    def health_check(self) -> bool:
        """Pings Ollama endpoint to verify connectivity."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False


Union_KeepAlive = Any
