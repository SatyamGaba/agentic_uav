from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from random import Random

from agentic_uav.models import UavState, manhattan

if TYPE_CHECKING:
    from agentic_uav.environment import BlackoutZone
    from agentic_uav.models import WorldState


MSG_HEARTBEAT = "heartbeat"
MSG_TASK_BID = "task_bid"
MSG_TASK_COMMITMENT = "task_commitment"
MSG_INTENT_SUMMARY = "intent_summary"
MSG_HAZARD_ALERT = "hazard_alert"
MSG_FAILURE_NOTICE = "failure_notice"
MSG_COVERAGE_UPDATE = "coverage_update"


@dataclass
class Message:
    sender_id: str
    message_type: str
    payload: dict[str, Any]
    ttl: int
    urgency: str = "routine"
    recipient_id: str | None = None


class NetworkModel:
    def __init__(self, communication_range: int, packet_loss_rate: float = 0.0, random: Random | None = None) -> None:
        self.communication_range = communication_range
        self.packet_loss_rate = packet_loss_rate
        self.random = random if random is not None else Random()
        self.pending: list[Message] = []
        self.blackout_zones: list[BlackoutZone] = []

    def enqueue(self, messages: list[Message]) -> None:
        self.pending.extend(messages)

    def _in_blackout_zone(self, cell: tuple[int, int], tick: int) -> bool:
        for bz in self.blackout_zones:
            if bz.is_active(tick) and cell in bz.cells:
                return True
        return False

    def deliver(self, uavs: dict[str, UavState], world: WorldState, tick: int) -> None:
        current = self.pending
        self.pending = []
        forwarded: list[Message] = []

        for message in current:
            sender_id = message.sender_id
            if sender_id not in uavs:
                continue
            sender_cell = uavs[sender_id].cell
            
            if self._in_blackout_zone(sender_cell, tick):
                continue

            sender_quality = world.sectors[sender_cell].comm_quality if sender_cell in world.sectors else 1.0

            delivered_to = self._neighbors(sender_id, uavs, message.recipient_id)
            for uav_id in delivered_to:
                recipient_cell = uavs[uav_id].cell
                if self._in_blackout_zone(recipient_cell, tick):
                    continue

                recipient_quality = world.sectors[recipient_cell].comm_quality if recipient_cell in world.sectors else 1.0
                delivery_prob = (1 - self.packet_loss_rate) * sender_quality * recipient_quality
                
                if self.random.random() > delivery_prob:
                    continue  # Packet lost

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
