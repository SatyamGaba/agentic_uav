"""OpenAI / Gemini function-calling JSON schemas for AgenticNav tools."""

from typing import Any, Dict, List

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_depth",
            "description": (
                "Query metric depth at one or more normalized pixel coordinates across current camera views. "
                "Returns center depth (depth_m), 5x5 spatial statistics (min, mean, median), and an action preview "
                "(preview_theta_deg, preview_rho_m) indicating the exact turn angle and forward displacement move_to would execute. "
                "Use this to inspect candidate path clearance, doorway openings, or obstacle distances before moving."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "description": "List of pixel points to query across views.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "view": {
                                    "type": "integer",
                                    "description": "Camera view index: 0=Front (0°), 1=Right (-90°), 2=Back (180°), 3=Left (+90°).",
                                    "enum": [0, 1, 2, 3],
                                },
                                "u": {
                                    "type": "number",
                                    "description": "Normalized horizontal image coordinate in [0.0, 1.0] (0.0=left edge, 1.0=right edge).",
                                },
                                "v": {
                                    "type": "number",
                                    "description": "Normalized vertical image coordinate in [0.0, 1.0] (0.0=top edge, 1.0=bottom edge).",
                                },
                            },
                            "required": ["view", "u", "v"],
                        },
                    }
                },
                "required": ["points"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Retrieve a past RGB observation from a previously visited decision node. "
                "Allows inspecting landmarks or verifying return paths. "
                "node must be between 1 and t-1 (where t is the current decision node shown on the BEV map)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "integer",
                        "description": "Past decision node index to recall (1 <= node <= current_node - 1).",
                    },
                    "view": {
                        "type": "integer",
                        "description": "Camera view index to retrieve: 0=Front, 1=Right, 2=Back, 3=Left.",
                        "enum": [0, 1, 2, 3],
                    },
                },
                "required": ["node", "view"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": (
                "Command the robot to navigate towards a visual target specified by normalized coordinates (u, v) in view. "
                "The safety engine deprojects the target, estimates surface slope, quantizes the displacement and heading, "
                "and verifies that the safety corridor is free of obstacles. "
                "Returns 'executed' if safe, or 'reselect' with an explanation if blocked or out of range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "integer",
                        "description": "Camera view index containing the target: 0=Front, 1=Right, 2=Back, 3=Left.",
                        "enum": [0, 1, 2, 3],
                    },
                    "u": {
                        "type": "number",
                        "description": "Normalized horizontal image coordinate in [0.0, 1.0].",
                    },
                    "v": {
                        "type": "number",
                        "description": "Normalized vertical image coordinate in [0.0, 1.0].",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Navigation rationale explaining why this target advances towards the instruction goal.",
                    },
                },
                "required": ["view", "u", "v", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": (
                "Declare that the goal described in the instruction has been reached and terminate navigation. "
                "Only call this when confident that the agent is within 3 meters of the target location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Rationale describing the observed visual landmarks matching the goal instruction.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]


def get_available_tools(enable_depth: bool = True, enable_recall: bool = True) -> List[Dict[str, Any]]:
    """Return tool schemas with optional ablation stripping."""
    tools = []
    for tool in TOOL_SCHEMAS:
        name = tool["function"]["name"]
        if name == "query_depth" and not enable_depth:
            continue
        if name == "recall" and not enable_recall:
            continue
        tools.append(tool)
    return tools

