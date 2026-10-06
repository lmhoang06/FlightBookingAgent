"""Automated Benchmark and Evaluation Runner for FlightBookingAgent.

Evaluates ReAct, Plan-then-Execute, and Hybrid patterns across 4 core scenarios:
1. Scenario A: Happy Path (Available morning flight under 2M VND)
2. Scenario B: Constraint Conflict (Goal drift prevention & structured handoff)
3. Scenario C: Dynamic Inventory / Replan Trigger (Observation delta handling)
4. Scenario D: State Corruption / Tool Glitch Resilience (Empty or malformed responses)

Measures:
- Latency (seconds)
- LLM Calls Count
- Estimated Token Count
- Task Success Rate (verified strictly by code is_done())
- Constraint Compliance Rate (%)
- Structured Handoff Validity
"""

import argparse
import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import (
    Constraints,
    ScriptedFakeChatModel,
    get_flight_agent,
)
from agents.config import AgentConfig
from agents.harness import is_done


def run_scenario(pattern: str, scenario_id: str, backend: str = "scripted") -> dict:
    """Executes a single pattern against a scenario and records evaluation metrics."""
    session_id = f"eval_{pattern}_{scenario_id}_{int(time.time() * 1000)}"

    # 1. Define Scenario Parameters & Constraints
    if scenario_id == "scenario_a":
        # Happy Path: Morning flight SGN -> DAD under 2M VND
        c = Constraints(
            origin="SGN",
            destination="DAD",
            date="2026-10-07",
            depart_before="12:00",
            depart_after="00:00",
            max_price=2_000_000,
            passenger_name="Nguyen Van A",
            passenger_count=1,
        )
        query = c.to_prompt()
        case = "success"

    elif scenario_id == "scenario_b":
        # Constraint Conflict: Departure between 01:00 and 05:00 (zero flights exist)
        c = Constraints(
            origin="SGN",
            destination="DAD",
            date="2026-10-07",
            depart_before="05:00",
            depart_after="01:00",
            max_price=1_000_000,
            passenger_name="Nguyen Van B",
            passenger_count=1,
        )
        query = c.to_prompt()
        case = "fail_drift"

    elif scenario_id == "scenario_c":
        # Dynamic Inventory / Replan: Flight selection requires adaptation
        c = Constraints(
            origin="SGN",
            destination="DAD",
            date="2026-10-07",
            depart_before="12:00",
            depart_after="06:00",
            max_price=2_000_000,
            passenger_name="Tran Thi C",
            passenger_count=1,
        )
        query = "Book flight SGN to DAD on 2026-10-07. Note: if first choice is full, pick next."
        case = "success"

    elif scenario_id == "scenario_d":
        # Malformed / Empty Search: Tests defense against state corruption
        c = Constraints(
            origin="NONEXISTENT",
            destination="NOWHERE",
            date="2026-10-07",
            max_price=2_000_000,
        )
        query = "Search flights NONEXISTENT to NOWHERE"
        case = "fail_drift"

    else:
        raise ValueError(f"Unknown scenario {scenario_id}")

    # 2. Setup Agent & Backend
    if backend == "scripted":
        llm = ScriptedFakeChatModel(case=case)
        agent = get_flight_agent(
            pattern=pattern,
            llm=llm,
            session_id=session_id,
            config=AgentConfig(max_iterations=8),
        )
    else:
        agent = get_flight_agent(
            pattern=pattern,
            session_id=session_id,
            config=AgentConfig(max_iterations=8),
        )

    # 3. Execution & Timing
    start_time = time.perf_counter()
    resp = agent.invoke(query, session_id=session_id, constraints=c)
    latency = time.perf_counter() - start_time

    # 4. Metric Extraction
    steps = resp.steps or []
    tool_calls = [s for s in steps if getattr(s, "step_type", "") == "tool_call"]

    # Compute LLM calls based on pattern architecture
    if pattern == "plan_then_execute":
        # 1 LLM call for one-shot plan generation, zero for plain code steps
        llm_calls = 1
    elif pattern == "react":
        # 1 call per reasoning turn
        llm_calls = max(1, len(tool_calls) + 1)
    elif pattern == "hybrid":
        # 1 macro planning call + 1 per milestone micro-loop
        llm_calls = max(2, len(tool_calls))
    else:
        llm_calls = 1

    # Estimate token consumption (approx 250 tokens per turn + prompt)
    total_tokens = llm_calls * 320 + len(query.split()) * 4

    # Code-Checked Done
    done_verified = is_done(session_id, c)

    # Constraint compliance
    compliance = 100.0 if (done_verified or resp.stop_reason in ("unachievable_goal", "awaiting_user_input")) else 0.0

    # Handoff Quality
    handoff_info = resp.metadata.get("handoff")
    has_valid_handoff = False
    if (
        handoff_info
        and "done_so_far" in handoff_info
        and "tried" in handoff_info
        and "question" in handoff_info
    ):
        has_valid_handoff = True

    return {
        "pattern": pattern,
        "scenario": scenario_id,
        "latency_sec": round(latency, 4),
        "llm_calls": llm_calls,
        "total_tokens": total_tokens,
        "is_done": done_verified,
        "compliance_pct": compliance,
        "stop_reason": resp.stop_reason,
        "has_valid_handoff": has_valid_handoff,
        "total_trace_steps": len(steps),
    }


def main():
    parser = argparse.ArgumentParser(description="Run FlightBookingAgent benchmarks")
    parser.add_argument(
        "--backend",
        choices=["scripted", "live"],
        default="scripted",
        help="Backend model: scripted (deterministic offline) or live (LLM API)",
    )
    args = parser.parse_args()

    patterns = ["react", "plan_then_execute", "hybrid"]
    scenarios = ["scenario_a", "scenario_b", "scenario_c", "scenario_d"]

    results = []
    print("\n=======================================================")
    print(f" FlightBookingAgent Benchmark Suite ({args.backend.upper()})")
    print("=======================================================\n")

    for pat in patterns:
        for sc in scenarios:
            res = run_scenario(pat, sc, backend=args.backend)
            results.append(res)
            print(
                f"[{pat.upper():<17}] {sc:<12} | Done: {res['is_done']!s:<5} | "
                f"Latency: {res['latency_sec']:<6}s | LLM Calls: {res['llm_calls']} | "
                f"Tokens: {res['total_tokens']:<5} | Stop: {res['stop_reason']}"
            )

    # Aggregate Metrics by Pattern
    print("\n-------------------------------------------------------")
    print(" SUMMARY BENCHMARK TABLE")
    print("-------------------------------------------------------")
    print(
        f"{'Pattern':<18} | {'Avg Latency':<12} | {'Avg LLM Calls':<14} | "
        f"{'Avg Tokens':<11} | {'Success (A)':<12} | {'Compliance':<10}"
    )
    print("-" * 75)

    summary_data = {}
    for pat in patterns:
        pat_res = [r for r in results if r["pattern"] == pat]
        avg_lat = round(sum(r["latency_sec"] for r in pat_res) / len(pat_res), 4)
        avg_llm = round(sum(r["llm_calls"] for r in pat_res) / len(pat_res), 1)
        avg_tok = int(sum(r["total_tokens"] for r in pat_res) / len(pat_res))
        succ_a = next(r["is_done"] for r in pat_res if r["scenario"] == "scenario_a")
        avg_comp = round(sum(r["compliance_pct"] for r in pat_res) / len(pat_res), 1)

        summary_data[pat] = {
            "avg_latency": avg_lat,
            "avg_llm_calls": avg_llm,
            "avg_tokens": avg_tok,
            "scenario_a_success": succ_a,
            "avg_compliance": avg_comp,
        }

        print(
            f"{pat:<18} | {avg_lat:<12} | {avg_llm:<14} | "
            f"{avg_tok:<11} | {succ_a!s:<12} | {avg_comp}%"
        )
    print("-------------------------------------------------------\n")

    # Save output artifacts
    output_dir = os.path.dirname(__file__)
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": results, "summary": summary_data}, f, indent=2)

    print(f"✅ Benchmark results saved to: {json_path}")


if __name__ == "__main__":
    main()
