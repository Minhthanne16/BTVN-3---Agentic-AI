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
    initial_plan = session.plan.copy()
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
        "trace": session.trace,
        "initial_plan": initial_plan,
        "approved": approved,
    }


def print_demo(result: dict) -> None:
    """Render recorded execution, including blocked calls, for terminal screenshots."""
    print(f"\n=== {result['agent']} | Scenario {result['scenario']}: {result['scenario_name']} ===")
    route = "VTG -> VCS" if result["scenario"] == 4 else "SGN -> DAD"
    print(f"Goal: {route}, {DATE}, depart before 12:00, price <= 2,000,000 VND")
    print(f"Mock booking/payment approval: {'yes' if result['approved'] else 'no'}")
    print("\n--- plan ---")
    if result["initial_plan"]:
        for index, name in enumerate(result["initial_plan"], 1):
            print(f"Step {index}: {name}")
    elif result["scenario"] == 5:
        print("Grounding test: model claims VN999, seat 5C, price 1.200.000đ without observations.")
    else:
        print("ReAct: choose the next action after each observation.")
    recoveries = [event["recovery"] for event in result["trace"] if "recovery" in event]
    if recoveries:
        print("\n--- recovery ---")
        for recovery in recoveries:
            print(recovery)
    print("\n--- trace ---")
    if not result["trace"]:
        print("No tool executed.")
    for index, event in enumerate(result["trace"], 1):
        arguments = json.dumps(event["args"], ensure_ascii=False)
        print(f"[{index}] {event['tool']}({arguments}) -> {event['outcome']}")
        if "reason" in event:
            print(f"    Reason: {event['reason']}")
        observation = event.get("result", {})
        if event["tool"] == "search_flights" and "count" in observation:
            print(f"    Found {observation['count']} flight(s)")
        elif event["tool"] == "check_seat" and "available_seats" in observation:
            print(f"    Available seats: {', '.join(observation['available_seats']) or 'none'}")
        elif "booking_code" in observation:
            print(f"    {observation['booking_code']} | {observation['flight_id']} | "
                  f"{observation['status']} | paid={observation['is_paid']}")
        if "error" in observation:
            print(f"    Error: {observation['error']}")
    print("\n--- result ---")
    payload = {key: result[key] for key in ("status", "answer")}
    if result["booking"]:
        payload["booking"] = {key: result["booking"][key] for key in (
            "booking_code", "flight_id", "seat_number", "depart_time", "price", "status", "is_paid")}
    if result["handoff"]:
        # The full attempts appear above; the four original handoff fields remain in JSON mode.
        payload["handoff"] = {**result["handoff"], "da_thu": [
            f"{event['tool']} -> {event['outcome']}" for event in result["trace"]]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nModel calls: {result['model_calls']} | Tool calls: {result['tool_calls']} | "
          f"Token~: {result['estimated_tokens']} | Catches: {result['safety_catches']} | "
          f"Replans: {result['replans']}")


def choose_interactively(parser: argparse.ArgumentParser, approved: bool) -> tuple[int, bool]:
    print("Choose scenario:")
    for number, config in SCENARIOS.items():
        print(f"  {number}. {config['name']}")
    try:
        while True:
            choice = input("Your choice (1-5): ").strip()
            if choice in {str(number) for number in SCENARIOS}:
                break
            print("Please enter a number from 1 to 5.")
        if not approved:
            while True:
                approval = input("Approve mock booking/payment (y/n): ").strip().lower()
                if approval in {"y", "n"}:
                    approved = approval == "y"
                    break
                print("Please enter y/n.")
        return int(choice), approved
    except (EOFError, KeyboardInterrupt):
        parser.exit(0, "\nDemo cancelled.\n")


def run_cli(strategy: str, run: Run) -> None:
    """Offer a standalone command for one strategy."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=f"SE373 mock flight agent: {strategy}")
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "5", "all"], default=None)
    parser.add_argument("--approve", action="store_true", help="Approve mock booking/payment in this run")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per scenario")
    parser.add_argument("--summary", action="store_true", help="Print the compact benchmark table")
    parser.add_argument("--interactive", action="store_true", help="Choose a scenario and mock approval interactively")
    args = parser.parse_args()
    if args.interactive and (args.scenario or args.json or args.summary):
        parser.error("--interactive cannot be combined with --scenario, --json, or --summary")
    interactive = args.interactive or (
        len(sys.argv) == 1 and sys.stdin.isatty() and sys.stdout.isatty())
    if interactive:
        scenario, approved = choose_interactively(parser, args.approve)
        results = [run(scenario, approved_override=approved)]
    else:
        scenarios = list(SCENARIOS) if args.scenario in {None, "all"} else [int(args.scenario)]
        results = [run(number, approved_override=True if args.approve else None) for number in scenarios]
    if args.json:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
    elif args.summary:
        print("Agent          KB  Trạng thái          Model  Tool  Token~  Thời gian(s)  Chặn")
        for result in results:
            print(
                f"{result['agent']:<14} {result['scenario']:<3} {result['status']:<20} "
                f"{result['model_calls']:<6} {result['tool_calls']:<5} "
                f"{result['estimated_tokens']:<7} {result['latency_seconds']:<13.6f} "
                f"{result['safety_catches']}"
            )
    else:
        for result in results:
            print_demo(result)
