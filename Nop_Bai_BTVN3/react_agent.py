"""ReAct design: decide again after each tool observation.

Run: python react_agent.py --scenario all
"""
from __future__ import annotations

from flight_core import Session, _request
from strategy_runtime import run_cli, run_strategy


class ReActPolicy:
    """Choose a fresh action after each observation and try another flight on failure."""

    def prepare(self, session: Session) -> None:
        pass

    def decide_action(self, session: Session) -> dict | None:
        response = session.ask_model(_request(session))
        if response.tool_calls:
            return response.tool_calls[0]
        session.stop("HANDOFF", "NO_ACTION", "Cần mở rộng tiêu chí tìm kiếm không?")
        return None

    def on_invalid_flight(self, session: Session, args: dict, reason: str) -> None:
        session.excluded.append(args["flight_id"])
        session.selected = None
        session.seat = None
        session.phase = "check"

    def on_no_seats(self, session: Session, args: dict) -> None:
        session.excluded.append(args["flight_id"])
        session.selected = None
        session.seat = None
        session.phase = "check"

    def on_no_flights(self, session: Session) -> None:
        # A repeat search is caught by the shared LoopDetector.
        pass


def run(scenario: int, approved_override: bool | None = None) -> dict:
    return run_strategy("react", scenario, ReActPolicy(), approved_override=approved_override)


def main() -> None:
    run_cli("react", run)


if __name__ == "__main__":
    main()
