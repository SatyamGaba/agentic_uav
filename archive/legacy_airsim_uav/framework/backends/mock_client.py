"""
Mock Vision-Language Model Backend for deterministic testing and validation.
Simulates tool calling, JSON outputs, and discrete action predictions without GPU or network dependencies.
"""
import time
import json
from typing import List, Dict, Any, Optional
from framework.interfaces.base_vlm import BaseVLMBackend, VLMResponse


class MockVLMBackend(BaseVLMBackend):
    """
    Configurable Mock Backend for end-to-end framework testing.
    """
    def __init__(
        self,
        model_name: str = "mock-vlm-v1",
        mode: str = "tool_call",  # 'tool_call', 'direct_pixel', 'discrete_actions'
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name=model_name, config=config)
        self.mode = mode
        self.call_history: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> VLMResponse:
        t0 = time.time()
        self.call_history.append({"messages": messages, "tools": tools})

        # Inspect latest user message text
        user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_content = str(m.get("content", ""))
                break

        # Simulate response based on mode
        if self.mode == "tool_call" and tools:
            # Check tool available
            tool_names = [t["function"]["name"] for t in tools if "function" in t]
            if "stop_arrival" in tool_names and ("arrived" in user_content.lower() or "final" in user_content.lower()):
                tool_call = {
                    "id": "mock_call_stop",
                    "type": "function",
                    "function": {
                        "name": "stop_arrival",
                        "arguments": json.dumps({"u": 0.5, "v": 0.5, "reason": "Reached target landmark"})
                    }
                }
            elif "move_to" in tool_names:
                # Steer slightly based on prompt
                u_val = 0.7 if "right" in user_content.lower() else (0.3 if "left" in user_content.lower() else 0.5)
                tool_call = {
                    "id": "mock_call_move",
                    "type": "function",
                    "function": {
                        "name": "move_to",
                        "arguments": json.dumps({"u": u_val, "v": 0.45, "reason": "Moving toward visible landmark corridor"})
                    }
                }
            else:
                tool_call = {
                    "id": "mock_call_fallback",
                    "type": "function",
                    "function": {
                        "name": tool_names[0],
                        "arguments": json.dumps({"u": 0.5, "v": 0.5, "reason": "Forward step"})
                    }
                }

            return VLMResponse(
                content="",
                thinking="Mock reasoning: analyzed frame and identified waypoint corridor.",
                tool_calls=[tool_call],
                raw_response={"mock": True},
                prompt_tokens=150,
                completion_tokens=45,
                latency_s=round(time.time() - t0, 3),
                model=self.model_name
            )

        elif self.mode == "direct_pixel":
            u_val = 0.7 if "right" in user_content.lower() else (0.3 if "left" in user_content.lower() else 0.5)
            content = json.dumps({
                "u": u_val,
                "v": 0.45,
                "is_stop": "stop" in user_content.lower() or "final" in user_content.lower(),
                "reason": "Direct visual grounding on the primary path"
            })
            return VLMResponse(
                content=content,
                thinking="Direct pixel identification.",
                tool_calls=[],
                raw_response={"mock": True},
                prompt_tokens=120,
                completion_tokens=30,
                latency_s=round(time.time() - t0, 3),
                model=self.model_name
            )

        elif self.mode == "discrete_actions":
            # Return OpenFly discrete action codes, e.g., [9, 9, 2, 8, 0]
            codes = [9, 8]
            if "right" in user_content.lower():
                codes.extend([2, 9])
            elif "left" in user_content.lower():
                codes.extend([3, 9])
            codes.append(0)
            return VLMResponse(
                content=json.dumps(codes),
                thinking="Generated OpenFly discrete plan.",
                tool_calls=[],
                raw_response={"mock": True},
                prompt_tokens=100,
                completion_tokens=20,
                latency_s=round(time.time() - t0, 3),
                model=self.model_name
            )

        # Default fallback text
        return VLMResponse(
            content='{"action": "forward", "distance": 6.0}',
            thinking="Default mock response.",
            tool_calls=[],
            raw_response={"mock": True},
            latency_s=round(time.time() - t0, 3),
            model=self.model_name
        )
