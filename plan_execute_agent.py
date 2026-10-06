"""Plan-then-Execute design: plan once, then execute fixed steps.

Run: python plan_execute_agent.py --scenario all
"""
from __future__ import annotations

import json

from flight_core import PASSENGER, Session
from strategy_runtime import run_cli, run_strategy


class PlanExecutePolicy:
    """Keep the initial five-step plan; hand off if it cannot be executed."""

    def prepare(self, session: Session) -> None:
        session.plan = json.loads(str(session.ask_model({"task": "plan", "phase": "search"}).content))

    def decide_action(self, session: Session) -> dict | None:
        if session.plan_index >= len(session.plan):
            session.stop("HANDOFF", "PLAN_EXHAUSTED", "Cần lập kế hoạch mới hay dừng yêu cầu?")
            return None

        name = session.plan[session.plan_index]
        session.plan_index += 1
        if name == "search_flights":
            args = {
                "origin": session.constraints.origin,
                "destination": session.constraints.destination,
                "date": session.constraints.date,
            }
        elif name == "check_seat":
            if not session.candidates:
                session.stop("HANDOFF", "NO_FLIGHTS", "Có chặng bay hoặc ngày khác không?")
                return None
            args = {"flight_id": session.candidates[0]}
        elif name == "book_seat":
            args = {
                "flight_id": session.selected,
                "seat_number": session.seat,
                "passenger_name": PASSENGER,
            }
        else:
            args = {"booking_code": session.booking_code}
        return {"name": name, "args": args}

    def on_invalid_flight(self, session: Session, args: dict, reason: str) -> None:
        session.stop("HANDOFF", f"CONSTRAINT_VIOLATION: {reason}", "Lập lại kế hoạch với chuyến hợp lệ?")

    def on_no_seats(self, session: Session, args: dict) -> None:
        session.stop("HANDOFF", "NO_SEATS", "Có cần tìm chuyến khác không?")

    def on_no_flights(self, session: Session) -> None:
        session.stop("HANDOFF", "NO_FLIGHTS", "Có chặng bay hoặc ngày khác không?")


def run(scenario: int, approved_override: bool | None = None) -> dict:
    return run_strategy("plan-execute", scenario, PlanExecutePolicy(), approved_override=approved_override)


def main() -> None:
    run_cli("plan-execute", run)


if __name__ == "__main__":
    main()
