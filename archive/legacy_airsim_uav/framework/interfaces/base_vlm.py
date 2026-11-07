"""
Abstract Base Class for Vision-Language Model Backends.
Provides swappable integration across Ollama, OpenAI-compatible APIs, Google Gemini, and Mock clients.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class VLMResponse:
    """Standardized response container across all VLM providers."""
    content: str = ""
    thinking: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""


class BaseVLMBackend(ABC):
    """
    Abstract interface for multi-modal Vision-Language Model execution.
    """
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> VLMResponse:
        """
        Sends multi-modal chat messages with optional tool declarations to the backend.
        Returns a standardized VLMResponse.
        """
        pass

    def health_check(self) -> bool:
        """Checks if the backend is reachable and ready to serve."""
        return True
