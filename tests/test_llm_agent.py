import unittest

from agentic_uav.llm_agent import (
    MockAgentDecider,
    decide_action,
    serialize_observation,
    validate_action,
)
from agentic_uav.planning import MethodState
from agentic_uav.simulation import (
    ScenarioConfig,
    Sector,
    Simulation,
    UavConfig,
    WorldConfig,
)


def build_simulation(*, sectors=None, urgent_cell=None) -> Simulation:
    sectors = list(sectors or [])
    if urgent_cell is not None:
        sectors.append(Sector(cell=urgent_cell, priority="urgent"))
    scenario = ScenarioConfig(
        method_name="agentic",
        ticks=1,
        communication_range=2,
        sensing_radius=1,
        heartbeat_interval=3,
        urgent_message_ttl=2,
        world=WorldConfig(width=4, height=4, sectors=sectors),
        uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
        events=[],
        seed=1,
    )
    return Simulation.from_config(scenario)


def build_observation(simulation: Simulation) -> dict:
    return simulation.observations.build(
        simulation.world, simulation.uavs, simulation.method_state.beliefs
    )["u0"]


class SerializeObservationTest(unittest.TestCase):
    def test_serialized_observation_omits_global_state(self) -> None:
        simulation = build_simulation(urgent_cell=(0, 1))
        serialized = serialize_observation(build_observation(simulation))

        self.assertEqual(set(serialized), {"self", "nearby", "known_urgent", "messages", "peer_intents"})
        self.assertNotIn("urgent_cells", serialized)
        self.assertNotIn("coverage", serialized)
        self.assertEqual(serialized["self"]["cell"], [0, 0])


class ValidateActionTest(unittest.TestCase):
    def test_rejects_out_of_bounds_target(self) -> None:
        simulation = build_simulation()
        self.assertIsNone(
            validate_action({"target_cell": [9, 9], "role": "coverage"}, simulation)
        )

    def test_rejects_blocked_target(self) -> None:
        simulation = build_simulation(sectors=[Sector(cell=(1, 0), blocked=True)])
        self.assertIsNone(
            validate_action({"target_cell": [1, 0], "role": "coverage"}, simulation)
        )

    def test_rejects_unknown_role(self) -> None:
        simulation = build_simulation()
        self.assertIsNone(
            validate_action({"target_cell": [1, 0], "role": "wander"}, simulation)
        )

    def test_accepts_valid_action_and_filters_ranked_targets(self) -> None:
        simulation = build_simulation(sectors=[Sector(cell=(2, 0), blocked=True)])
        result = validate_action(
            {
                "target_cell": [1, 0],
                "role": "priority_responder",
                "ranked_targets": [[1, 0], [2, 0], [9, 9], [0, 1]],
            },
            simulation,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["target_cell"], (1, 0))
        self.assertEqual(result["role"], "priority_responder")
        # (2, 0) blocked and (9, 9) out of bounds are dropped.
        self.assertEqual(result["ranked_targets"], [(1, 0), (0, 1)])

    def test_allows_null_target(self) -> None:
        simulation = build_simulation()
        result = validate_action({"target_cell": None, "role": "relay"}, simulation)

        self.assertIsNotNone(result)
        self.assertIsNone(result["target_cell"])


class DecideActionFallbackTest(unittest.TestCase):
    def test_falls_back_to_greedy_when_decider_raises(self) -> None:
        class Raising:
            def decide(self, observation):
                raise RuntimeError("boom")

        simulation = build_simulation(urgent_cell=(0, 1))
        observation = build_observation(simulation)

        action = decide_action(Raising(), simulation, observation, MethodState())

        # Greedy fallback targets the locally-sensed urgent cell.
        self.assertEqual(action.target_cell, (0, 1))
        self.assertEqual(action.new_role, "priority_responder")

    def test_falls_back_to_greedy_when_decider_returns_junk(self) -> None:
        class Junk:
            def decide(self, observation):
                return {"not": "an action", "role": "nonsense"}

        simulation = build_simulation(urgent_cell=(0, 1))
        observation = build_observation(simulation)

        action = decide_action(Junk(), simulation, observation, MethodState())

        self.assertEqual(action.target_cell, (0, 1))

    def test_falls_back_when_decider_returns_non_dict(self) -> None:
        class NotJson:
            def decide(self, observation):
                return "definitely not json"

        simulation = build_simulation()
        observation = build_observation(simulation)

        action = decide_action(NotJson(), simulation, observation, MethodState())

        self.assertIsNotNone(action)
        self.assertEqual(action.uav_id, "u0")


class MockAgentDeciderTest(unittest.TestCase):
    def test_returns_valid_in_bounds_action(self) -> None:
        simulation = build_simulation(urgent_cell=(0, 1))
        observation = build_observation(simulation)
        decider = MockAgentDecider()

        raw = decider.decide(serialize_observation(observation))
        validated = validate_action(raw, simulation)

        self.assertIsNotNone(validated)
        self.assertEqual(validated["target_cell"], (0, 1))
        self.assertEqual(validated["role"], "priority_responder")

    def test_mirrors_greedy_choosing_uncovered_when_no_urgent(self) -> None:
        simulation = build_simulation()
        observation = build_observation(simulation)
        decider = MockAgentDecider()

        action = decide_action(decider, simulation, observation, MethodState())

        self.assertEqual(action.new_role, "coverage")
        self.assertIsNotNone(action.target_cell)
        self.assertNotEqual(action.target_cell, (0, 0))


class PeerIntentPriorTest(unittest.TestCase):
    """The decider must actually CONSUME peer_intents (gap 12b): a peer's
    announced target should steer the decision away from that cell when an
    equally-good alternative exists."""

    def _serialized(self, peer_target):
        # u0 at (1, 1) with two equidistant uncovered neighbours (0, 1) and (2, 1).
        nearby = [
            {"cell": cell, "coverage": 0.0, "priority": "normal", "blocked": False}
            for cell in ([0, 1], [2, 1], [1, 1])
        ]
        peer_intents = (
            [{"peer_id": "u1", "target_cell": list(peer_target)}] if peer_target else []
        )
        return {
            "self": {"cell": [1, 1]},
            "nearby": nearby,
            "known_urgent": [],
            "messages": [],
            "peer_intents": peer_intents,
        }

    def test_decider_avoids_peer_claimed_cell_when_alternative_exists(self) -> None:
        decider = MockAgentDecider()
        baseline = decider.decide(self._serialized(None))["target_cell"]
        avoided = decider.decide(self._serialized(tuple(baseline)))["target_cell"]

        # With its first choice claimed by a peer, it picks the other candidate.
        self.assertNotEqual(avoided, baseline)
        self.assertEqual(set(map(tuple, [baseline, avoided])), {(0, 1), (2, 1)})

    def test_peer_intent_reaches_serialized_observation(self) -> None:
        # End-to-end: an intent_summary in the inbox surfaces as peer_intents on
        # the observation (so a real LLM would see it), not the empty default.
        from agentic_uav.communication import Message
        from agentic_uav.policy import _ingest_messages

        simulation = build_simulation()
        observation = build_observation(simulation)
        observation["messages"] = [
            Message(
                sender_id="u1",
                message_type="intent_summary",
                payload={"target_cell": (1, 0), "role": "coverage"},
                ttl=1,
            )
        ]
        _ingest_messages("u0", observation, simulation.method_state)

        serialized = serialize_observation(observation)
        self.assertEqual(
            serialized["peer_intents"], [{"peer_id": "u1", "target_cell": [1, 0]}]
        )


if __name__ == "__main__":
    unittest.main()
