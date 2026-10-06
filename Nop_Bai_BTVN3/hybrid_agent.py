"""Hybrid design: follow a plan, choose actions from observations, replan on failure.

Run: python hybrid_agent.py --scenario all
"""
from __future__ import annotations

import json

from flight_core import Session, _request
from strategy_runtime import run_cli, run_strategy


class HybridPolicy:
    """Follow a plan while rethinking its remaining steps after a failed candidate."""

    def prepare(self, session: Session) -> None:
        session.plan = json.loads(str(session.ask_model({"task": "plan", "phase": "search"}).content))

    def decide_action(self, session: Session) -> dict | None:
        response = session.ask_model(_request(session))
        if not response.tool_calls:
            session.stop("HANDOFF", "NO_ACTION", "Cần mở rộng tiêu chí tìm kiếm không?")
            return None
        action = response.tool_calls[0]
        if session.plan_index >= len(session.plan) or action["name"] != session.plan[session.plan_index]:
            session.stop("HANDOFF", "PLAN_MISMATCH", "Có cần lập lại kế hoạch không?")
            return None
        session.plan_index += 1
        return action

    def _replan_after_rejection(self, session: Session, flight_id: str) -> None:
        session.excluded.append(flight_id)
        session.selected = None
        session.seat = None
        session.phase = "check"
        session.plan = json.loads(
            str(session.ask_model({"task": "plan", "phase": "check", "excluded": session.excluded}).content)
        )
        session.plan_index = 0
        session.replans += 1

    def on_invalid_flight(self, session: Session, args: dict, reason: str) -> None:
        self._replan_after_rejection(session, args["flight_id"])

    def on_no_seats(self, session: Session, args: dict) -> None:
        self._replan_after_rejection(session, args["flight_id"])

    def on_no_flights(self, session: Session) -> None:
        session.stop("HANDOFF", "NO_FLIGHTS", "Có chặng bay hoặc ngày khác không?")


def run(scenario: int, approved_override: bool | None = None) -> dict:
    return run_strategy("hybrid", scenario, HybridPolicy(), approved_override=approved_override)


def main() -> None:
    run_cli("hybrid", run)


if __name__ == "__main__":
    main()
