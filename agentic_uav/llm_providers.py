"""Real LLM backends behind the provider-agnostic :class:`AgentDecider` seam.

Two drop-in deciders live here:

* :class:`BedrockAgentDecider` — AWS Bedrock via the boto3 ``bedrock-runtime``
  Converse API with *forced tool use*, so the model is constrained to return a
  schema-valid action object.
* :class:`QGenieAgentDecider` — QGenie over plain HTTP (stdlib ``urllib``),
  assuming an OpenAI-compatible chat-completions surface.

Design rules this module obeys:

* ``boto3``/``urllib`` are imported **lazily inside the call path**, never at
  module top level, so ``import agentic_uav.llm_providers`` works without the
  optional ``llm`` extra installed and so :mod:`agentic_uav.llm_agent` stays
  dependency-free (it imports providers only lazily, via ``build_decider``).
* Providers do NOT re-implement validation. They return a raw action dict;
  :func:`agentic_uav.llm_agent.validate_action` remains the single source of
  truth, and :func:`agentic_uav.llm_agent.decide_proposal` falls back to the
  greedy heuristic on any exception or invalid output.
* The prompt constrains target choice to the cells the UAV can actually see
  (open ``nearby`` sectors plus believed-urgent cells). Letting the model
  invent coordinates would mean mass validation failure and a silent collapse
  of ``agentic`` into ``greedy`` — see :func:`build_prompt`.

QGenie speaks an OpenAI-compatible chat-completions API (confirmed against the
pi-qgenie-provider source): ``POST https://qgenie-api.qualcomm.com/v1/chat/completions``
with ``Authorization: Bearer $QGENIE_API_KEY`` and the usual ``model`` /
``messages`` / ``temperature`` / ``max_tokens`` fields, parsing
``choices[0].message``. Note ``/v2/`` is only the models-list path; the chat
endpoint is under ``/v1``. Model ids use a ``provider::model`` scheme (e.g.
``anthropic::claude-4-6-sonnet``, ``azure::gpt-4.1``); ``/v1/chat/completions``
serves every family (it strips reasoning output, which is irrelevant for our
tiny JSON action). Tool-calling support varies per model (``supports_tool_call``
in /v2/models), so the robust default parses a JSON object out of the message
text and OpenAI-style tool calling is opt-in (``QGENIE_USE_TOOLS``).
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentic_uav.llm_agent import ALLOWED_ROLES

Cell = tuple[int, int]

TOOL_NAME = "choose_action"
TOOL_DESCRIPTION = (
    "Choose this UAV's next target cell, role, and ranked fallback targets. "
    "target_cell and every ranked_targets entry MUST be one of the listed "
    "candidate cells, or use null target_cell to hold/patrol."
)

# Mirrors the contract enforced by agentic_uav.llm_agent.validate_action:
# role in ALLOWED_ROLES; target_cell a nullable [x, y] integer pair (null =>
# patrol/hold); ranked_targets a list of [x, y] pairs used by in-tick
# deconfliction. ``reason`` is accepted for free-form rationale but ignored by
# the validator.
ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_cell": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Chosen cell [x, y] from the candidate list, or null to hold/patrol.",
        },
        "role": {
            "type": "string",
            "enum": list(ALLOWED_ROLES),
            "description": "The UAV's role for this plan.",
        },
        "ranked_targets": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "description": "Fallback cells [x, y] in preference order, each from the candidate list.",
        },
        "reason": {
            "type": "string",
            "description": "Short rationale (optional; ignored by the simulator).",
        },
    },
    "required": ["role", "ranked_targets"],
}


def candidate_cells(observation: dict[str, Any]) -> list[Cell]:
    """Return the cells a UAV may legally target, in a stable order.

    This is the union of open (non-blocked) ``nearby`` sectors and the UAV's
    believed-urgent cells — exactly the cells :func:`validate_action` will
    accept. ``nearby`` order is preserved first, then any extra urgent cells.
    """

    cells: list[Cell] = []
    seen: set[Cell] = set()
    for sector in observation.get("nearby", []):
        if not isinstance(sector, dict) or sector.get("blocked"):
            continue
        cell = _as_cell(sector.get("cell"))
        if cell is not None and cell not in seen:
            seen.add(cell)
            cells.append(cell)
    for raw in observation.get("known_urgent", []):
        cell = _as_cell(raw)
        if cell is not None and cell not in seen:
            seen.add(cell)
            cells.append(cell)
    return cells


def build_prompt(observation: dict[str, Any]) -> tuple[str, str]:
    """Build the (system, user) prompt from a UAV's LOCAL serialized observation.

    The user text enumerates the candidate cells (with coverage/priority tags)
    and the model is told to choose ONLY from them. Built purely from the
    serialized dict — no access to global world state (decentralization /
    observation symmetry).
    """

    self_info = observation.get("self", {}) or {}
    self_cell = self_info.get("cell")
    self_role = self_info.get("role")

    candidates = candidate_cells(observation)
    nearby_by_cell = {
        _as_cell(sector.get("cell")): sector
        for sector in observation.get("nearby", [])
        if isinstance(sector, dict) and _as_cell(sector.get("cell")) is not None
    }
    urgent = {cell for cell in (_as_cell(c) for c in observation.get("known_urgent", [])) if cell is not None}

    lines: list[str] = []
    for cell in candidates:
        tags: list[str] = []
        if cell in urgent:
            tags.append("URGENT")
        sector = nearby_by_cell.get(cell)
        if sector is not None:
            tags.append(f"coverage={sector.get('coverage')}")
            priority = sector.get("priority")
            if priority and priority != "normal":
                tags.append(f"priority={priority}")
        else:
            tags.append("believed-urgent (beyond sensor range)")
        lines.append(f"  - {list(cell)}: {', '.join(tags) if tags else 'sensed, open'}")
    candidate_block = "\n".join(lines) if lines else "  (none — no open sector in range; hold/patrol)"

    peer_intents = observation.get("peer_intents", []) or []

    system_text = (
        "You are the onboard decision agent for a single UAV in a decentralized "
        "swarm performing disaster-area mapping. Each UAV decides on its own from "
        "its LOCAL view only; there is no global map. Maximize area coverage and "
        "respond quickly to urgent sectors, while spreading the swarm out (avoid "
        "duplicating cells peers have already claimed).\n\n"
        "Decide three things: target_cell (where this UAV should head next), role "
        f"(one of {', '.join(ALLOWED_ROLES)}), and ranked_targets (fallback cells "
        "in preference order, used when another UAV contests your target).\n\n"
        "HARD CONSTRAINTS:\n"
        "- target_cell and every ranked_targets entry MUST be chosen from the "
        "Candidate Cells list. Do NOT invent coordinates.\n"
        "- Use target_cell = null only when no candidate is worth moving to "
        "(hold/patrol).\n"
        "- Prefer URGENT cells for the priority_responder role; prefer "
        "low-coverage cells for the coverage role."
    )

    user_text = (
        f"Your UAV id: {self_info.get('uav_id')}\n"
        f"Your current cell: {self_cell}\n"
        f"Your current role: {self_role}\n\n"
        f"Candidate Cells (choose target_cell and ranked_targets from these only):\n"
        f"{candidate_block}\n\n"
        f"Peer intents (cells/roles peers announced this tick): {json.dumps(peer_intents)}\n\n"
        "Choose the action now."
    )
    return system_text, user_text


def parse_bedrock_tool_response(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``choose_action`` tool input from a Bedrock Converse response."""

    content = response["output"]["message"]["content"]
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict) and tool_use.get("name") == TOOL_NAME:
            return tool_use["input"]
    raise ValueError("Bedrock Converse response contained no choose_action toolUse block")


def parse_qgenie_response(resp_json: dict[str, Any]) -> dict[str, Any]:
    """Parse a QGenie OpenAI-compatible chat-completions response.

    Tries the ``tool_calls`` path first (when tool calling is enabled and the
    model supports it), then falls back to extracting a JSON object from the
    message ``content`` text.
    """

    message = resp_json["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        arguments = tool_calls[0]["function"]["arguments"]
        return json.loads(arguments) if isinstance(arguments, str) else arguments
    content = message.get("content") or ""
    return _extract_json_object(content)


# --- internal helpers -------------------------------------------------------


_QGENIE_JSON_DIRECTIVE = (
    "Respond with ONLY a single JSON object and nothing else (no prose, no "
    "markdown fences). Shape: "
    '{"target_cell": [x, y] or null, "role": one of '
    f"{list(ALLOWED_ROLES)}, "
    '"ranked_targets": [[x, y], ...]}.'
)


class _BaseLLMDecider:
    """Shared cache + call accounting for the real LLM deciders.

    Subclasses implement :meth:`_invoke`. ``calls``/``successes`` let callers
    report the fallback rate (a high silent-fallback rate would invisibly
    collapse ``agentic`` toward ``greedy``). Caching is opt-in
    (``AGENTIC_UAV_LLM_CACHE``) and keyed by the serialized observation, which
    is JSON by construction.
    """

    def __init__(self, *, temperature: float | None, max_tokens: int, cache: bool) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.successes = 0
        self._cache_enabled = cache
        self._cache: dict[str, dict[str, Any]] = {}

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        key: str | None = None
        if self._cache_enabled:
            key = json.dumps(observation, sort_keys=True, default=str)
            cached = self._cache.get(key)
            if cached is not None:
                self.successes += 1
                return cached
        result = self._invoke(observation)
        if key is not None:
            self._cache[key] = result
        self.successes += 1
        return result

    def _invoke(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class BedrockAgentDecider(_BaseLLMDecider):
    """AWS Bedrock decider using the Converse API with forced tool use.

    AWS credentials and region come from the standard boto3 chain (e.g.
    ``AWS_PROFILE``/``AWS_REGION`` after ``breeze aws auth``); no auth code
    here. The model id is an inference-profile id such as
    ``us.anthropic.claude-opus-4-7`` from ``BEDROCK_MODEL_ID``. boto3 is
    imported and the client built lazily on first call, so constructing this
    object (e.g. in ``build_method``) never requires boto3 or touches network.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        region: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 512,
        read_timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_attempts: int = 2,
        cache: bool | None = None,
    ) -> None:
        # Temperature is OMITTED by default: reasoning models (e.g.
        # us.anthropic.claude-opus-4-7) reject the `temperature` field outright
        # ("temperature is deprecated"), and sending it would make every call
        # fail and silently fall back to greedy. Opt in via BEDROCK_TEMPERATURE
        # only for models that accept it.
        super().__init__(
            temperature=temperature if temperature is not None else _env_float("BEDROCK_TEMPERATURE"),
            max_tokens=max_tokens,
            cache=_env_flag("AGENTIC_UAV_LLM_CACHE") if cache is None else cache,
        )
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
        self.region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout
        self.max_attempts = max_attempts
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    read_timeout=self.read_timeout,
                    connect_timeout=self.connect_timeout,
                    retries={"max_attempts": self.max_attempts, "mode": "standard"},
                ),
            )
        return self._client

    def _invoke(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.model_id:
            raise ValueError("BEDROCK_MODEL_ID is not set; cannot call Bedrock")
        system_text, user_text = build_prompt(observation)
        client = self._get_client()
        inference_config: dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            inference_config["temperature"] = self.temperature
        response = client.converse(
            modelId=self.model_id,
            system=[{"text": system_text}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": TOOL_NAME,
                            "description": TOOL_DESCRIPTION,
                            "inputSchema": {"json": ACTION_SCHEMA},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": TOOL_NAME}},
            },
            inferenceConfig=inference_config,
        )
        return parse_bedrock_tool_response(response)


class QGenieAgentDecider(_BaseLLMDecider):
    """QGenie decider over plain HTTP (stdlib urllib).

    Speaks the OpenAI-compatible ``POST {base_url}/chat/completions`` surface
    (``base_url`` defaults to ``https://qgenie-api.qualcomm.com/v1``) with
    ``Authorization: Bearer $QGENIE_API_KEY`` and ``model`` / ``messages`` /
    ``temperature`` / ``max_tokens`` fields, parsing ``choices[0].message``.
    ``QGENIE_MODEL`` uses QGenie's ``provider::model`` id scheme, e.g.
    ``anthropic::claude-4-6-sonnet`` or ``azure::gpt-4.1``. The robust default
    parses a JSON object from the message content; OpenAI tool calling is
    opt-in via ``QGENIE_USE_TOOLS`` (not all models advertise tool support).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 512,
        timeout: float = 30.0,
        use_tools: bool | None = None,
        ca_bundle: str | None = None,
        cache: bool | None = None,
    ) -> None:
        # Temperature omitted by default (opt in via QGENIE_TEMPERATURE):
        # reasoning models routed through QGenie (gpt-5*/o-series) reject a
        # non-default temperature, which would fail every call.
        super().__init__(
            temperature=temperature if temperature is not None else _env_float("QGENIE_TEMPERATURE"),
            max_tokens=max_tokens,
            cache=_env_flag("AGENTIC_UAV_LLM_CACHE") if cache is None else cache,
        )
        self.model = model or os.environ.get("QGENIE_MODEL")
        self.api_key = api_key or os.environ.get("QGENIE_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("QGENIE_BASE_URL")
            or "https://qgenie-api.qualcomm.com/v1"
        ).rstrip("/")
        self.timeout = timeout
        self.use_tools = _env_flag("QGENIE_USE_TOOLS") if use_tools is None else use_tools
        # stdlib urllib does NOT honor AWS_CA_BUNDLE/REQUESTS_CA_BUNDLE the way
        # boto3/requests do; behind the corporate TLS-inspecting proxy we must
        # pass the CA bundle explicitly or HTTPS verification fails.
        self.ca_bundle = ca_bundle or _resolve_ca_bundle()

    def _invoke(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("QGENIE_API_KEY is not set; cannot call QGenie")
        if not self.model:
            raise ValueError("QGENIE_MODEL is not set; cannot call QGenie")
        import ssl
        import urllib.request

        system_text, user_text = build_prompt(observation)
        # JSON-in-text is the robust default; nudge the model to emit only JSON.
        user_text = f"{user_text}\n\n{_QGENIE_JSON_DIRECTIVE}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.use_tools:
            # OpenAI-style function calling; only some QGenie models advertise
            # supports_tool_call, so this stays opt-in (JSON-in-text otherwise).
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "parameters": ACTION_SCHEMA,
                    },
                }
            ]
            body["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}

        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        context = ssl.create_default_context(cafile=self.ca_bundle) if self.ca_bundle else None
        with urllib.request.urlopen(request, timeout=self.timeout, context=context) as http_response:
            payload = json.loads(http_response.read().decode("utf-8"))
        return parse_qgenie_response(payload)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Find and parse the first balanced ``{...}`` object in ``text``.

    Tolerates surrounding prose and ```` ```json ```` fences by scanning for the
    first ``{`` and matching braces (respecting string literals).
    """

    if not text or not text.strip():
        raise ValueError("empty content; no JSON object to parse")
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in content")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("unbalanced JSON object in content")


def _as_cell(value: Any) -> Cell | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _resolve_ca_bundle() -> str | None:
    """Find a CA bundle for stdlib TLS behind a corporate proxy.

    urllib has no built-in env support like requests/boto3, so we consult the
    common bundle env vars (including the breeze ``AWS_CA_BUNDLE``) and return
    the first that points at an existing file. ``None`` => system defaults.
    """

    for name in (
        "QGENIE_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "AWS_CA_BUNDLE",
    ):
        path = os.environ.get(name)
        if path and os.path.isfile(path):
            return path
    return None


__all__ = [
    "ACTION_SCHEMA",
    "BedrockAgentDecider",
    "QGenieAgentDecider",
    "build_prompt",
    "candidate_cells",
    "parse_bedrock_tool_response",
    "parse_qgenie_response",
]
