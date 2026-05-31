from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_uav.models import UavState, manhattan


@dataclass
class Message:
    sender_id: str
    message_type: str
    payload: dict[str, Any]
    ttl: int
    urgency: str = "routine"
    recipient_id: str | None = None


class NetworkModel:
    def __init__(self, communication_range: int) -> None:
        self.communication_range = communication_range
        self.pending: list[Message] = []

    def enqueue(self, messages: list[Message]) -> None:
        self.pending.extend(messages)

    def deliver(self, uavs: dict[str, UavState]) -> None:
        current = self.pending
        self.pending = []
        forwarded: list[Message] = []

        for message in current:
            delivered_to = self._neighbors(message.sender_id, uavs, message.recipient_id)
            for uav_id in delivered_to:
                received = Message(
                    sender_id=message.sender_id,
                    message_type=message.message_type,
                    payload=dict(message.payload),
                    ttl=message.ttl,
                    urgency=message.urgency,
                    recipient_id=uav_id,
                )
                uavs[uav_id].inbox.append(received)
                if message.urgency == "urgent" and message.ttl > 1:
                    forwarded.append(
                        Message(
                            sender_id=uav_id,
                            message_type=message.message_type,
                            payload=dict(message.payload),
                            ttl=message.ttl - 1,
                            urgency=message.urgency,
                        )
                    )
        self.pending.extend(forwarded)

    def _neighbors(
        self,
        sender_id: str,
        uavs: dict[str, UavState],
        recipient_id: str | None = None,
    ) -> list[str]:
        if sender_id not in uavs:
            return []
        sender = uavs[sender_id]
        if not sender.active:
            return []

        neighbors: list[str] = []
        for uav_id, uav in uavs.items():
            if uav_id == sender_id or not uav.active:
                continue
            if recipient_id is not None and uav_id != recipient_id:
                continue
            if manhattan(sender.cell, uav.cell) <= self.communication_range:
                neighbors.append(uav_id)
        return neighbors
