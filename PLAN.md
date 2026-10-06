# FlightBookingAgent: Master Implementation Plan
## Harness Architecture & Course Pattern Realignment (BTVN#3)

---

## 1. Executive Summary & Context

### 1.1. Assignment Requirements (BTVN#3 - SE373)
The assignment objective is to build a robust flight booking agent using **LangChain** and **LangGraph**, comprising:
1. **Full Harness Architecture (4 Layers)**:
   - **Constraints are DATA** (*Ràng buộc là dữ liệu*)
   - **Permission Check** (*Kiểm quyền*)
   - **Done checked by CODE** (*Tiêu chí hoàn thành kiểm bằng code*)
   - **Handoff to Human** (*Bàn giao*)
2. **Three Agent Design Patterns matching Lecture Slides**:
   - **ReAct** (Reasoning + Acting with Harness interceptor)
   - **Plan-then-Execute** (One-shot plan generation + Human reviewer gate + Plain-code step execution)
   - **Hybrid** (Plan -> Execute $k$ steps -> Check if observation changed significantly -> Dynamic Replan)
3. **Comprehensive Evaluation & Report**:
   - Multi-metric benchmark comparing ReAct, Plan-then-Execute, and Hybrid across diverse scenarios.
   - Full technical report (`REPORT.md`).
4. **Resilience to 4 Classic Failure Modes** (from slide & demo):
   - Infinite Loop, Tool Hallucination, Goal Drift, State Corruption.

### 1.2. Current Repository State & Gap Analysis
- **Current Strengths**: Production-grade mock flight database (`tools/db.py`), 8 well-defined tools (`tools/`), rich Streamlit UI (`app.py`), and 40 passing pytest unit tests.
- **Identified Gaps**:
  1. *Missing Harness*: No centralized `harness.py`, no structured `Constraints` class acting as data, no `check_permission()` running before tools, no `is_done()` querying the DB directly, and no structured `handoff()` (`done_so_far`, `tried`, `question`).
  2. *Plan-then-Execute Mismatch*: Current implementation acts like autonomous BabyAGI (calls LLM sub-agent on every step + calls LLM replanner after every step). It lacks the slide's core tenets: **calling the model ONCE for the whole plan, Human Reviewer approval gate, and zero-LLM-cost plain-code step execution with placeholder replacement**.
  3. *Hybrid Pattern Mismatch*: Current implementation uses fixed macro-milestones rather than the slide's pattern: **Plan -> Execute $k$ steps -> Evaluate observation delta -> Dynamic Replan**.
  4. *Missing Evaluation & Report*: No quantitative benchmark script and no comparative `REPORT.md`.

---

## 2. Target Architecture Overview

```
                                  [ USER REQUEST ]
                                         │
                         ┌───────────────▼───────────────┐
                         │      1. CONSTRAINTS AS DATA   │
                         │   origin, dest, date, budget, │
                         │   depart_before/after, seats  │
                         └───────────────┬───────────────┘
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
   ┌─────────▼─────────┐       ┌─────────▼─────────┐       ┌─────────▼─────────┐
   │    PATTERN 1:     │       │    PATTERN 2:     │       │    PATTERN 3:     │
   │      ReAct        │       │ Plan-then-Execute │       │      Hybrid       │
   │  (Slide Pattern)  │       │  (Slide Pattern)  │       │  (Slide Pattern)  │
   │                   │       │                   │       │                   │
   │ 01 Dựng ngữ cảnh  │       │ Model sinh trọn   │       │ Model lập plan    │
   │ 02 Model đề xuất  │       │ kế hoạch (1 lần)  │       │ ban đầu (k bước)  │
   │ 03 Harness gọi    │       │        │          │       │        │          │
   │    tool (kiểm     │       │        ▼          │       │        ▼          │
   │    quyền)         │       │  NGƯỜI DUYỆT      │       │ Thực thi k bước   │
   │ 04 Ghi kết quả    │       │ (Approve/Reject)  │       │ (Harness kiểm)    │
   │ 05 Xét dừng bằng  │       │        │          │       │        │          │
   │    code           │       │        ▼          │       │        ▼          │
   │                   │       │ Chạy Step 1, 2, 3 │       │ Observation đổi   │
   │                   │       │ bằng Plain Code   │       │ đáng kể?          │
   │                   │       │ ($booking_code)   │       │ (Có: Replan,      │
   │                   │       │                   │       │  Không: đi tiếp)  │
   └─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                               ┌─────────▼─────────┐
                               │ 2. PERMISSION CHK │
                               │ Runs BEFORE tool: │
                               │ Blocks if violates│
                               │ data constraints  │
                               └─────────┬─────────┘
                                         │
                               ┌─────────▼─────────┐
                               │ 3. DONE CHK (CODE)│
                               │ is_done() queries │
                               │ DB: paid & valid  │
                               └─────────┬─────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ SUCCESS: DONE ]              [ FAILED: HANDOFF ]
                                                - done_so_far
                                                - tried
                                                - question
```

---

## 3. Detailed Component Implementation Plan

### Phase 1: Core Harness Module (`agents/harness.py`)
Create a single source of truth for safety, constraint validation, failure prevention, and human handoff.

#### 1.1. Constraints as Data (`Constraints`)
```python
@dataclass
class Constraints:
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"          # Departure time threshold
    depart_after: str = "00:00"
    max_price: int = 2_000_000            # Max allowable price (VND/USD)
    passenger_name: str = "Nguyen Van A"
    passenger_count: int = 1

    def to_prompt(self) -> str:
        """Deterministically generates prompt from structured data."""
        return (
            f"Book {self.passenger_count} ticket(s) {self.origin} -> {self.destination} on {self.date}, "
            f"departing between {self.depart_after} and {self.depart_before}, "
            f"price at most {self.max_price:,}. Passenger: {self.passenger_name}."
        )

    def is_ok(self, flight_or_booking: dict) -> bool:
        """Validates if a flight offer or booking strictly satisfies all data constraints."""
        # Check departure date, time range, price threshold, and route
        ...
```

#### 1.2. Permission Check (`check_permission`)
- Intercepts calls **before** tools execute.
- Inspects args and referenced flight offers in `MockFlightDatabase`.
- If the flight exceeds `max_price`, violates departure window, or exceeds capacity:
  - Returns a denial reason string: `f"{offer_id} breaks the constraints: {constraints.to_prompt()}"`.
  - The tool execution is blocked, preventing unintended state changes or payments.

#### 1.3. Code-Checked Done (`is_done`)
- Queries the underlying database directly (`db.get_active_bookings(session_id)`).
- Verifies:
  1. A booking record actually exists in the database.
  2. The booking status is `"CONFIRMED"` / paid.
  3. The booked flight satisfies all constraints defined in `Constraints`.
- **Golden Rule**: Never trust the LLM's text output stating "I have completed your booking".

#### 1.4. Structured Handoff (`handoff`)
When the agent reaches max turns, tool denial, or unachievable constraints, return:
```python
def handoff(session_id: str, log: list, constraints: Constraints) -> dict:
    return {
        "done_so_far": [...],  # Current bookings held / paid
        "tried": [...],        # Tools called, arguments, and outcomes
        "question": "No flight meets all constraints. Which one can we relax: departure time or maximum price?",
    }
```

#### 1.5. Four Classic Failure Mode Mitigations
- **Infinite Loop Detection (`is_looping`)**: Tracks `(tool_name, json_args)` in execution log. If called $\ge 3$ times identically, immediately raises `StopAgentException("LOOP: Repeated call detected")`.
- **Tool Hallucination Detection (`check_hallucination`)**: Regex-checks flight numbers and prices mentioned in agent's final text against the tool result log.
- **Goal Drift Prevention**: Enforced by `check_permission` on every booking/pricing step.
- **State Corruption Detection (`is_empty_tool_error`)**: Ensures empty query responses `{}` from mock network failures are treated as errors rather than "no flights exist".

---

### Phase 2: Refactoring & Realigning the 3 Agent Patterns

#### 2.1. ReAct Agent (`agents/react/agent.py`)
- **Slide Alignment**: `01 Dựng ngữ cảnh` -> `02 Model đề xuất tool` -> `03 Harness gọi tool (kiểm quyền)` -> `04 Ghi kết quả` -> `05 Xét điều kiện dừng (is_done)`.
- **Implementation**:
  - Accept `constraints: Constraints` in `invoke()`.
  - Wrap tool execution with `check_permission` and log recording.
  - After agent execution, invoke `is_done()`:
    - If `True`: set `stop_reason = "completed"`.
    - If `False`: set `stop_reason = "failed"` and attach `handoff()`.
  - Maintain compatibility with both LangGraph prebuilt loop and pure tool middleware.

#### 2.2. Plan-then-Execute Agent (`agents/plan_then_execute/`)
- **Slide Alignment**: One model call generates the entire plan -> Human Reviewer approves/rejects -> Plain code executes steps in order with `$placeholder` replacement -> Result verified by `is_done()`.
- **Implementation**:
  - **Planner Chain**: Uses structured output `Plan(steps=[PlanStep(tool=..., args=...)])`.
  - **Human-in-the-Loop Review Gate (`approve_plan`)**:
    - Supports programmatic auto-approval (`auto_approve=True` for tests/evals) and interactive approval (`callback/input` for CLI/Streamlit).
    - If rejected, prompts model to regenerate plan up to `MAX_PLANS = 2`.
  - **Deterministic Plain-Code Step Executor**:
    - Eliminates the need to spawn nested LLMs for each step.
    - Resolves dynamic variables between steps (e.g., passing `$booking_reference` from `create_flight_booking` to `get_booking_order`).
    - Runs `check_permission` before each tool execution.
  - **Zero LLM Step Overhead**: Predictable upfront cost and token count.

#### 2.3. Hybrid Agent (`agents/hybrid/`)
- **Slide Alignment**: `Plan -> Execute k steps -> Check if observation changed significantly -> Dynamic Replan`.
- **Implementation**:
  - State includes `active_plan: list[str]`, `k_steps_budget: int = 2`, `last_observation_delta: bool`.
  - Model creates a short initial sequence of steps (e.g. Discovery + Search).
  - Agent executes up to $k$ steps with tool tracking.
  - **Observation Delta Evaluator (`should_replan`)**:
    - Checks if tool outputs brought unexpected states (e.g., flight fully booked, price changed, zero matching flights).
    - If observation changed significantly -> routes to `replanner` node.
    - If observation is normal -> continues executing remaining steps.
    - If plan completed -> verifies via `is_done()`.

---

### Phase 3: Offline / Scripted Demo Mode (`agents/fake_model.py`)
- Implement a deterministic `ScriptedFakeChatModel` mimicking the baseline `fake_model` from `DemoFlightAgent`.
- Supports scripted paths:
  1. `case="success"`: Valid flight found -> booked -> paid -> `is_done() == True`.
  2. `case="fail_drift"`: Model attempts to book an out-of-budget flight -> Harness blocks it -> `handoff()`.
  3. `case="fail_loop"`: Model retries the same query 3 times -> Harness interrupts loop.
- Ensures the entire project can be evaluated and demonstrated offline without an active OpenAI API key or network connection.

---

### Phase 4: Streamlit UI Integration (`app.py`)
Enhance `app.py` with the lecture harness features:
1. **Constraints Control Panel**:
   - Sidebar UI to configure origin, destination, date, depart before/after, max price, and passenger details.
   - Button: *"Sync Constraints as DATA"*.
2. **Human-in-the-Loop Approval Modal/Section**:
   - For Plan-then-Execute: Displays the generated plan before execution, offering "Approve (Execute)" and "Reject (Regenerate)" buttons.
3. **Live Harness Inspector**:
   - Displays real-time Permission Check decisions (`ALLOWED` / `DENIED`).
   - Code-Checked Done Status badge (`is_done: TRUE / FALSE`).
   - Structured Handoff Card (`done_so_far`, `tried`, `question`) when a request fails.
4. **Offline Mock Demo Switch**:
   - Toggle to test cases 1 & 2 without requiring external LLM API calls.

---

### Phase 5: Empirical Benchmark & Evaluation Suite (`scripts/evaluate.py`)
Build an automated evaluation runner executing all 3 patterns across standard evaluation scenarios:

#### 5.1. Evaluation Scenarios
1. **Scenario A (Happy Path)**: Available morning flight under 2M VND.
2. **Scenario B (Constraint Conflict)**: Only afternoon or expensive flights exist (tests Goal Drift prevention and Handoff).
3. **Scenario C (Dynamic Inventory Change)**: Initial selected flight runs out of seats (tests Hybrid replanning).
4. **Scenario D (Malformed/Empty Search)**: Network failure simulation (tests State Corruption defense).

#### 5.2. Metrics Recorded
- **LLM Calls Count**: Total invocations of the language model.
- **Total Token Consumption**: Input + output tokens (or estimated equivalents).
- **Execution Latency**: Wall-clock time (seconds).
- **Task Success Rate**: Verified exclusively by code `is_done()`.
- **Constraint Compliance Rate**: % of booked flights strictly satisfying data constraints.
- **Handoff Quality**: Presence and validity of `done_so_far`, `tried`, and actionable `question`.

---

### Phase 6: Formal Comprehensive Report (`REPORT.md`)
Draft a publication-grade markdown report structured as follows:

1. **Giới thiệu & Bài toán (Introduction)**:
   - Bài toán đặt vé máy bay và vai trò của Harness Architecture.
2. **Kiến trúc Lớp Harness (The 4 Harness Layers)**:
   - Ràng buộc là dữ liệu (Constraints as DATA).
   - Kiểm quyền trước tool (Permission check).
   - Tiêu chí hoàn thành kiểm bằng code (Done checked by code).
   - Bàn giao có cấu trúc (Handoff to human).
   - Phòng chống 4 failure modes (Loop, Hallucination, Goal Drift, State Corruption).
3. **Hiện thực 3 Mẫu thiết kế theo Slide (Architectural Realization)**:
   - ReAct Pattern: Vòng lặp lý luận và can thiệp Harness.
   - Plan-then-Execute Pattern: Kế hoạch một lần, Người duyệt (Human-in-the-loop), thực thi code thuần.
   - Mẫu Lai (Hybrid Pattern): Lập kế hoạch, thực thi $k$ bước, phát hiện biến động observation, tái lập kế hoạch.
4. **Kết quả Đánh giá Thực nghiệm (Empirical Evaluation)**:
   - Bảng tổng hợp số liệu Benchmark giữa 3 mẫu.
   - Phân tích Trade-offs: Chi phí (Token) vs Độ trễ vs Tính linh hoạt vs Khả năng kiểm soát.
5. **Kết luận & Bài học kinh nghiệm (Conclusions & Key Takeaways)**.

---

## 4. Work Breakdown Structure & Implementation Checklist

- [x] **Step 1: Implement `agents/harness.py`**
  - [x] Implement `Constraints` dataclass with `to_prompt()` and `is_ok()`.
  - [x] Implement `check_permission(tool_name, args, constraints)`.
  - [x] Implement `is_done(session_id, constraints)`.
  - [x] Implement `handoff(session_id, log, constraints)`.
  - [x] Implement failure mitigations (`is_looping`, `check_hallucination`, `is_empty_or_error`).
- [x] **Step 2: Implement Scripted Fake Model (`agents/fake_model.py`)**
  - [x] Scripted responses for success, drift, and loop scenarios.
- [x] **Step 3: Update `agents/react/agent.py`**
  - [x] Integrate Harness tool wrapper and code-checked done verification.
- [x] **Step 4: Update `agents/plan_then_execute/`**
  - [x] Implement slide-compliant one-shot plan generation.
  - [x] Implement Human Review approval gate (`approve_plan`).
  - [x] Implement plain-code deterministic execution with placeholder substitution.
- [x] **Step 5: Update `agents/hybrid/`**
  - [x] Implement $k$-step bounded execution with observation delta replanning.
- [x] **Step 6: Update Streamlit Application (`app.py`)**
  - [x] Add Constraints manager, Human Reviewer UI, Harness badge, and Offline Demo switcher.
- [x] **Step 7: Build Evaluation Engine (`scripts/evaluate.py`)**
  - [x] Execute automated benchmark across the 4 scenarios and output metrics table.
- [x] **Step 8: Write Comprehensive Report (`REPORT.md`)**
  - [x] Document architectures, formulas, benchmark results, and slide alignments.
- [x] **Step 9: Run Test Suite & Final QA**
  - [x] Run `pytest` across all tool and agent tests.
  - [x] Verify clean code, ruff lints, and typing rule adherence.
