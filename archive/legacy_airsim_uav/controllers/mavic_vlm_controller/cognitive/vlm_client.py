"""
VLM Client & Self-Consistency Wrapper
=====================================
Multi-backend VLM interface supporting structured schema generation,
robust markdown-json extraction, and concurrent self-consistency voting.
"""

from __future__ import annotations
import abc
import asyncio
import base64
import json
import logging
import re
import urllib.request
import urllib.error
from typing import List, Type, TypeVar, Optional, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("MavicVLM")

T = TypeVar("T", bound=BaseModel)


def extract_json_from_llm_output(text: str) -> Dict[str, Any]:
    """Robustly extracts JSON object or array from VLM markdown response, stripping thinking tokens."""
    # Strip <think> ... </think> blocks emitted by reasoning models (e.g. Qwen 2.5 / DeepSeek / Qwen 3.5)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    
    # Try 1: Markdown code fence ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        snippet = match.group(1).strip()
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # Try 2: Find outermost matching curly brackets { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = cleaned[first_brace:last_brace + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # Try 3: Direct parse
    return json.loads(cleaned)


class VLMClient(abc.ABC):
    """Abstract interface for Vision-Language Model inference."""

    @abc.abstractmethod
    async def query_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[np.ndarray],
        schema: Type[T],
        temperature: float = 0.2,
    ) -> T:
        """Queries the VLM and validates response against Pydantic schema T."""
        pass


class MockVLMClient(VLMClient):
    """Deterministic Mock VLM client for unit testing with zero external dependencies."""

    def __init__(self, response_queue: Optional[List[BaseModel]] = None):
        self.response_queue = list(response_queue) if response_queue else []
        self.call_history: List[Dict[str, Any]] = []

    def queue_response(self, response: BaseModel) -> None:
        self.response_queue.append(response)

    async def query_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[np.ndarray],
        schema: Type[T],
        temperature: float = 0.2,
    ) -> T:
        self.call_history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_count": len(images),
            "schema": schema,
            "temperature": temperature,
        })
        if self.response_queue:
            resp = self.response_queue.pop(0)
            if isinstance(resp, schema):
                return resp
            # If queued as dict or different model, convert via model_validate
            return schema.model_validate(resp)
        
        # Default fallback instance
        raise ValueError(f"MockVLMClient response queue is empty. Expected instance of {schema.__name__}")


class OpenAICompatibleVLMClient(VLMClient):
    """Client for OpenAI-compatible HTTP endpoints (LM Studio, vLLM, Ollama, OpenAI)."""

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:1234/v1/chat/completions",
        api_key: str = "",
        model_name: str = "qwen3-vl",
        timeout_s: float = 20.0,
        max_retries: int = 2,
        keep_alive: str = "-1",
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.keep_alive = keep_alive

    def _encode_image_b64(self, img_bgr: np.ndarray) -> str:
        import cv2
        success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode image to JPEG.")
        return base64.b64encode(buffer).decode("utf-8")

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[np.ndarray],
        temperature: float,
    ) -> Dict[str, Any]:
        content_items: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for img in images:
            b64_str = self._encode_image_b64(img)
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_items},
        ]
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    async def warmup(self) -> bool:
        """Pre-warms the local VLM endpoint to load vision encoder and weights into VRAM before flight."""
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        dummy_payload = self._build_payload(
            system_prompt="You are an autonomous assistant. Respond with empty JSON.",
            user_prompt="Warmup test.",
            images=[dummy_img],
            temperature=0.0
        )
        dummy_payload["max_tokens"] = 10
        payload_bytes = json.dumps(dummy_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            def _call():
                req = urllib.request.Request(self.endpoint_url, data=payload_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    return resp.read()
            await asyncio.to_thread(_call)
            logger.info(f"[OpenAICompatibleVLMClient] Pre-flight warmup successful for model '{self.model_name}'.")
            return True
        except Exception as e:
            logger.warning(f"[OpenAICompatibleVLMClient] Pre-flight warmup failed (non-fatal): {e}")
            return False

    async def query_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[np.ndarray],
        schema: Type[T],
        temperature: float = 0.2,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            f"You MUST output valid JSON conforming exactly to this JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Return ONLY the JSON object, with no conversational fluff."
        )

        payload = self._build_payload(augmented_system, user_prompt, images, temperature)
        payload_bytes = json.dumps(payload).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries + 1):
            try:
                def _do_http():
                    req = urllib.request.Request(self.endpoint_url, data=payload_bytes, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                        return response.read().decode("utf-8")

                raw_resp = await asyncio.to_thread(_do_http)
                resp_json = json.loads(raw_resp)
                raw_text = resp_json["choices"][0]["message"]["content"]
                parsed_dict = extract_json_from_llm_output(raw_text)
                return schema.model_validate(parsed_dict)

            except (urllib.error.URLError, json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"[OpenAICompatibleVLMClient] Attempt {attempt+1}/{self.max_retries+1} failed: {e}")
                if attempt == self.max_retries:
                    raise


class SelfConsistentVLMWrapper:
    """Dispatches N concurrent VLM queries with temperature > 0 and performs consensus voting.
    
    - Majority-votes on discrete fields (intent, selected_mark_id)
    - Averages continuous scalar fields (confidence)
    - Fast cutoff: if quorum is reached before all complete, proceeds without waiting.
    """

    def __init__(self, backend: VLMClient, n_samples: int = 1, temperature: float = 0.3, timeout_s: float = 12.0):
        self.backend = backend
        self.n_samples = max(1, n_samples)
        self.temperature = temperature
        self.timeout_s = timeout_s

    async def query_with_voting(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[np.ndarray],
        schema: Type[T],
    ) -> T:
        if self.n_samples == 1:
            return await self.backend.query_structured(system_prompt, user_prompt, images, schema, temperature=0.1)

        # Dispatch N concurrent tasks
        tasks = [
            asyncio.create_task(
                self.backend.query_structured(system_prompt, user_prompt, images, schema, temperature=self.temperature)
            )
            for _ in range(self.n_samples)
        ]

        done, pending = await asyncio.wait(tasks, timeout=self.timeout_s, return_when=asyncio.ALL_COMPLETED)
        for p in pending:
            p.cancel()

        successful: List[T] = []
        for t in done:
            try:
                successful.append(t.result())
            except Exception as e:
                logger.warning(f"[SelfConsistentVLMWrapper] Sample failed: {e}")

        if not successful:
            raise RuntimeError(f"All {self.n_samples} self-consistency samples failed.")

        if len(successful) == 1:
            return successful[0]

        # Consolidate samples via majority voting on discrete attributes
        best_candidate = successful[0]

        # 1. Consensus voting for Set-of-Mark discrete candidate selection
        if hasattr(best_candidate, "selected_mark_id"):
            votes: Dict[Optional[int], int] = {}
            for item in successful:
                val = getattr(item, "selected_mark_id")
                votes[val] = votes.get(val, 0) + 1

            # Check for ties or split decisions
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            top_mark, top_count = sorted_votes[0]

            # If there is a tie for 1st place, or no majority agreement
            if len(sorted_votes) > 1 and sorted_votes[1][1] == top_count:
                logger.warning(
                    f"[SelfConsistentVLMWrapper] Set-of-Mark consensus tie detected: {votes}. "
                    "Rejecting ambiguous selection -> Safe non-selection fallback."
                )
                return best_candidate.model_copy(update={
                    "selected_mark_id": None,
                    "confidence": 0.0,
                    "reasoning": "Consensus tie detected across self-consistency samples. Defaulting to safe re-observation.",
                })

            # Clear winner exists
            matching = [s for s in successful if getattr(s, "selected_mark_id") == top_mark]
            avg_conf = float(np.mean([getattr(s, "confidence", 0.5) for s in matching]))
            return matching[0].model_copy(update={"confidence": avg_conf})

        # 2. Consensus voting for Subgoals
        if hasattr(best_candidate, "intent"):
            intent_votes: Dict[Any, int] = {}
            for item in successful:
                intent_val = getattr(item, "intent")
                intent_votes[intent_val] = intent_votes.get(intent_val, 0) + 1

            sorted_intents = sorted(intent_votes.items(), key=lambda x: x[1], reverse=True)
            top_intent, top_count = sorted_intents[0]

            # If tie on intent or ambiguous, choose the highest-caution action
            is_tie = len(sorted_intents) > 1 and sorted_intents[1][1] == top_count
            if is_tie:
                logger.warning(f"[SelfConsistentVLMWrapper] Subgoal intent tie detected: {intent_votes}. Enforcing highest caution.")
                # Pick the safest sample and enforce high caution
                safe_choice = successful[0].model_copy(update={
                    "caution_level": "high",
                    "v_target": min(1.0, getattr(successful[0], "v_target", 2.0)),
                })
                return safe_choice

            # Select sample matching top intent with highest caution
            matching_subgoals = [s for s in successful if getattr(s, "intent") == top_intent]
            # If any sample requested high caution, maintain high caution
            has_high_caution = any(getattr(s, "caution_level", "low") == "high" for s in matching_subgoals)
            chosen = matching_subgoals[0]
            if has_high_caution:
                chosen = chosen.model_copy(update={"caution_level": "high"})
            return chosen

        return best_candidate
