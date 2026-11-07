"""Gemini-3.7-Flash API client with function calling and multi-modal image packaging."""

import io
import json
import os
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from PIL import Image

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def encode_numpy_to_pil(img_array: np.ndarray) -> Image.Image:
    """Convert (H, W, 3) or (H, W) uint8 numpy array to PIL Image."""
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    if len(img_array.shape) == 2:
        return Image.fromarray(img_array, mode="L")
    return Image.fromarray(img_array, mode="RGB")


def convert_tools_to_gemini_format(tool_schemas: List[Dict[str, Any]]) -> List[Any]:
    """Converts OpenAI-style tool schema definitions into Google GenAI FunctionDeclarations."""
    declarations = []
    for tool in tool_schemas:
        fn = tool.get("function", {})
        declarations.append({
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters"),
        })
    return declarations


class GeminiNavClient:
    """Client for querying Gemini-3.7-Flash with strict greedy decoding (temp=0.0)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = os.environ.get("GEMINI_MODEL", model_name)
        self.temperature = temperature
        self.client = None

        if self.api_key and GENAI_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)

    def is_configured(self) -> bool:
        return bool(self.client is not None)

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Invoke Gemini model on the multi-turn conversation messages."""
        if not self.is_configured():
            raise RuntimeError(
                "GeminiNavClient is not configured. Set GEMINI_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

        # 1. Format contents for GenAI API
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_instruction = msg.get("content", "")
                continue

            parts = []

            # Text content
            content_text = msg.get("content")
            if content_text:
                parts.append(types.Part.from_text(text=content_text))

            # Multi-view RGB images
            rgb_views = msg.get("rgb_views")
            if rgb_views:
                for v_idx in sorted(rgb_views.keys()):
                    pil_img = encode_numpy_to_pil(rgb_views[v_idx])
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=90)
                    parts.append(
                        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
                    )

            # BEV map
            bev_img = msg.get("bev_image")
            if bev_img is not None:
                pil_bev = encode_numpy_to_pil(bev_img)
                buf_bev = io.BytesIO()
                pil_bev.save(buf_bev, format="PNG")
                parts.append(
                    types.Part.from_bytes(data=buf_bev.getvalue(), mime_type="image/png")
                )

            # Recalled image from recall tool
            recalled_img = msg.get("recalled_image")
            if recalled_img is not None:
                pil_rec = encode_numpy_to_pil(recalled_img)
                buf_rec = io.BytesIO()
                pil_rec.save(buf_rec, format="JPEG", quality=90)
                parts.append(
                    types.Part.from_bytes(data=buf_rec.getvalue(), mime_type="image/jpeg")
                )

            # Assistant tool calls
            if role == "assistant":
                resp_data = msg.get("response", {})
                for tc in resp_data.get("tool_calls", []):
                    parts.append(
                        types.Part.from_function_call(
                            name=tc["name"],
                            args=tc.get("arguments", {}),
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))

            # Tool responses
            elif role == "tool":
                t_name = msg.get("name")
                t_content = msg.get("content")
                try:
                    t_response = json.loads(t_content) if isinstance(t_content, str) else t_content
                except Exception:
                    t_response = {"raw_output": str(t_content)}

                parts.append(
                    types.Part.from_function_response(
                        name=t_name,
                        response={"result": t_response},
                    )
                )
                contents.append(types.Content(role="user", parts=parts))

            else:
                contents.append(types.Content(role="user", parts=parts))

        # 2. Convert function schemas
        gemini_tools = convert_tools_to_gemini_format(tools)

        # Privilege Firewall Check
        from agenticnav_paper_replica.vlm.firewall import assert_firewall_integrity
        full_text = (system_instruction or "") + "\n"
        for c in contents:
            for p in getattr(c, "parts", []):
                t = getattr(p, "text", None)
                if t:
                    full_text += t + "\n"
        assert_firewall_integrity(full_text)

        # 3. Call API
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
            tools=[{"function_declarations": gemini_tools}] if gemini_tools else None,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        # 4. Parse response into canonical dictionary
        tool_calls = []
        thought = ""

        if response.candidates:
            cand = response.candidates[0]
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if part.text:
                        thought += part.text
                    if part.function_call:
                        fc = part.function_call
                        tool_calls.append({
                            "name": fc.name,
                            "arguments": dict(fc.args) if fc.args else {},
                        })

        return {
            "thought": thought.strip(),
            "tool_calls": tool_calls,
        }

