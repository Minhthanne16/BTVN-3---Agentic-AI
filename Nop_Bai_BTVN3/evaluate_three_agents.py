"""Evaluate the three separate agent modules over the same five mock cases."""
from __future__ import annotations

import json
from pathlib import Path
import platform

import hybrid_agent
import plan_execute_agent
import react_agent
from flight_core import SCENARIOS


AGENTS = {
    "react": react_agent,
    "plan-execute": plan_execute_agent,
    "hybrid": hybrid_agent,
}


def evaluate() -> dict:
    cases = [
        module.run(scenario)
        for module in AGENTS.values()
        for scenario in SCENARIOS
    ]
    summary = {}
    for name in AGENTS:
        rows = [row for row in cases if row["agent"] == name]
        completed = sum(row["status"] == "COMPLETED" for row in rows)
        summary[name] = {
            "completed": completed,
            "cases": len(rows),
            "success_rate": round(completed / len(rows), 2),
            "model_calls": sum(row["model_calls"] for row in rows),
            "tool_calls": sum(row["tool_calls"] for row in rows),
            "estimated_tokens": sum(row["estimated_tokens"] for row in rows),
            "mock_cost_usd": 0.0,
            "latency_seconds": round(sum(row["latency_seconds"] for row in rows), 6),
            "safety_catches": sum(row["safety_catches"] for row in rows),
        }
    return {
        "environment": {
            "python": platform.python_version(),
            "model": "deterministic BaseChatModel",
            "token_method": "(prompt characters + answer characters) // 4; approximation",
        },
        "summary": summary,
        "cases": cases,
    }


if __name__ == "__main__":
    output = Path(__file__).with_name("benchmark_three_agents.json")
    results = evaluate()
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results['cases'])} cases to {output}")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
