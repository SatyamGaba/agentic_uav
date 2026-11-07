"""
OpenAI-Compatible Multi-Modal VLM Backend.
Connects with LM Studio, vLLM, DeepSeek, and official OpenAI endpoints.
"""
import time
import json
import urllib.request
from typing import List, Dict, Any, Optional
from framework.interfaces.base_vlm import BaseVLMBackend, VLMResponse


class OpenAIVLMBackend(BaseVLMBackend):
    """
    Standard OpenAI-compatible API client for vision-language models.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name=model_name, config=config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> VLMResponse:
        t0 = time.time()
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
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
        choices = result.get("choices", [])
        if not choices:
            return VLMResponse(
                content="",
                tool_calls=[],
                raw_response=result,
                latency_s=round(latency, 3),
                model=self.model_name
            )

        msg = choices[0].get("message", {})
        raw_content = msg.get("content", "") or ""
        native_tool_calls = msg.get("tool_calls", [])

        final_tool_calls = []
        if native_tool_calls:
            for tc in native_tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                final_tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                })

        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return VLMResponse(
            content=raw_content,
            thinking="",
            tool_calls=final_tool_calls,
            raw_response=result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=round(latency, 3),
            model=self.model_name
        )
