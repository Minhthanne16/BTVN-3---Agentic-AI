"""Shared mock store, harness, LangGraph loop, and CLI for strategy modules."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from flight_core import (
    DATE,
    SCENARIOS,
    FlightConstraints,
    FlightStore,
    GroundingChecker,
    Session,
    execute,
    flight_tools,
)

Run = Callable[..., dict]


def reject_unsupported_claim(session: Session) -> bool:
    """Use the same grounding gate for the deliberate hallucination case."""
    if session.scenario != 5:
        return False
    answer = session.ask_model({"task": "hallucinate"}).content
    _, missing = GroundingChecker().check(str(answer), session.observations)
    session.safety_catches += 1
    session.stop(
        "UNSUPPORTED_CLAIM",
        f"Không có căn cứ: {', '.join(missing)}",
        "Có cần tra cứu dữ liệu đặt vé thực tế không?",
    )
    return True


def run_strategy(
    strategy: str,
    scenario: int,
    policy: Any,
    approved_override: bool | None = None,
) -> dict:
    """Run one design with the shared mock tools and code-based harness."""
    if scenario not in SCENARIOS:
        raise ValueError("Kịch bản không hợp lệ")

    store = FlightStore()
    rules = FlightConstraints(
        "VTG" if scenario == 4 else "SGN",
        "VCS" if scenario == 4 else "DAD",
        DATE,
    )
    approved = SCENARIOS[scenario]["approved"] if approved_override is None else approved_override
    session = Session(strategy, scenario, store, rules, approved=approved)
    tools = flight_tools(store, SCENARIOS[scenario]["order"])
    if scenario != 5:
        policy.prepare(session)
    started = time.perf_counter()

    def decide(state: dict) -> dict:
        if session.model_calls >= 15 or session.tool_calls >= 15:
            session.stop("HANDOFF", "BUDGET_EXCEEDED", "Có cần tăng giới hạn lượt gọi không?")
            return {**state, "action": None}
        if reject_unsupported_claim(session):
            return {**state, "action": None}
        return {**state, "action": policy.decide_action(session)}

    def act(state: dict) -> dict:
        execute(session, state["action"], tools, policy)
        return {**state, "tick": state["tick"] + 1}

    graph = StateGraph(dict)
    graph.add_node("decide", decide)
    graph.add_node("act", act)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide", lambda state: "act" if session.status == "RUNNING" and state.get("action") else END
    )
    graph.add_conditional_edges("act", lambda state: "decide" if session.status == "RUNNING" else END)
    graph.compile().invoke({"tick": 0}, {"recursion_limit": 100})
    elapsed = time.perf_counter() - started
    return {
        "agent": strategy,
        "scenario": scenario,
        "scenario_name": SCENARIOS[scenario]["name"],
        "status": session.status,
        "answer": session.answer,
        "booking": session.verified_booking,
        "handoff": session.handoff_report,
        "model_calls": session.model_calls,
        "tool_calls": session.tool_calls,
        "estimated_tokens": session.estimated_tokens,
        "latency_seconds": round(elapsed, 6),
        "safety_catches": session.safety_catches,
        "replans": session.replans,
        "steps": len(session.attempts),
        "mock_cost_usd": 0.0,
        "attempts": session.attempts,
    }


def run_cli(strategy: str, run: Run) -> None:
    """Offer a standalone command for one strategy."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=f"SE373 mock flight agent: {strategy}")
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--approve", action="store_true", help="Approve mock booking/payment in this run")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per scenario")
    args = parser.parse_args()
    scenarios = list(SCENARIOS) if args.scenario == "all" else [int(args.scenario)]
    results = [run(number, approved_override=True if args.approve else None) for number in scenarios]
    if args.json:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
    else:
        print("Agent          KB  Trạng thái          Model  Tool  Token~  Thời gian(s)  Chặn")
        for result in results:
            print(
                f"{result['agent']:<14} {result['scenario']:<3} {result['status']:<20} "
                f"{result['model_calls']:<6} {result['tool_calls']:<5} "
                f"{result['estimated_tokens']:<7} {result['latency_seconds']:<13.6f} "
                f"{result['safety_catches']}"
            )
