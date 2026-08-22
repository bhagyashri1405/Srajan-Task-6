# 🛰️ Research & Competitor Intelligence Agent

## Team Members
- [Your name]
- [Teammate name, if any]

## Problem Statement
Organizations, startups, and research institutions operate in highly competitive
and rapidly evolving environments where staying updated on research trends,
patent developments, competitor strategies, and industry news is critical.
However, manually monitoring scientific publications, patent databases, news
platforms, and social media sources is time-consuming, inefficient, and prone
to missing important updates. The lack of timely insights can result in lost
opportunities, delayed innovation, and weakened competitive positioning.
Therefore, there is a need for an autonomous AI agent capable of continuously
tracking research and competitor activities, analyzing vast information
sources, and delivering concise, actionable insights in real time.

## Project Description
This project is an autonomous AI agent that takes a topic, technology area, or
competitor name and produces a concise intelligence briefing on it. Instead of
a single fixed search, a **LangGraph pipeline** plans what to investigate,
gathers findings using two tools (run in parallel where useful), checks its
own findings for gaps and conflicts, replans if it needs more, and hands off
to an Analyst step that synthesizes the final report. Every step is visible
in the UI's reasoning trace, so the decision-making is transparent, not a
black box.

## Agent Framework: LangGraph

This system is built on **LangGraph**, a graph-based orchestration framework
for stateful, multi-step agent systems. It was chosen because it natively
provides a shared mutable state across nodes, real conditional routing
between nodes based on that state, and built-in checkpointing — all used
directly here, not simulated.

### Pipeline

```
Planner → Execute Tools (parallel) → Evaluator → (loop back to Planner, or) → Analyst
```

- **Planner** — decides, each round, whether more research is needed and
  which tool call(s) to run next (dynamic planning + adaptive task
  decomposition). It runs again on every replanning loop, so its decisions
  change based on what's already been found.
- **Execute Tools** — runs the planner's tool calls: `search_web`
  (DuckDuckGo, for competitor/news/patent/general info) and `search_arxiv`
  (the arXiv API, for academic research papers). When two are planned in the
  same round, they run concurrently via a thread pool (parallel execution).
  Each call has try/except fallback to the other tool on failure (tool
  fallback + failure recovery).
- **Evaluator** — checks whether findings are sufficient and flags
  conflicting evidence between sources (self-evaluation, conflicting-evidence
  resolution, uncertainty awareness). It also enforces two **deterministic**
  safety checks, independent of what the LLM decides: a hard cap on
  replanning rounds (resource-aware execution) and a check for a repeated
  identical action (loop/deadlock detection).
- **Conditional routing** — a real LangGraph conditional edge sends control
  back to the Planner (autonomous replanning) or forward to the Analyst,
  based on the Evaluator's output.
- **Analyst** — synthesizes the final briefing, assigns a confidence note per
  section, and explicitly addresses any conflicts the Evaluator flagged.
- **Checkpointing** — the graph is compiled with LangGraph's `MemorySaver`
  checkpointer, keyed by a per-search `thread_id`. Every run is checkpointed;
  this is shown in the UI trace as a "💾 Checkpoint" step.

### Adversarial live-test mode

The sidebar has a **"Force tool failures this run"** toggle. Turning it on
makes every tool call's primary attempt fail on purpose, forcing the
fallback path to run — a manual switch for demonstrating failure recovery
live and on command, not a claim that the system withstands arbitrary
real-world failures.

### Scope notes (what's real vs. simplified here)

Everything above is implemented in actual LangGraph code, not just
described — verified with a mocked API client covering the replanning loop,
the failure/fallback path, conflict flagging, and the deadlock guard. A few
things are intentionally scoped down for a hackathon timeline rather than
faked:
- "Uncertainty-aware decisions" = a confidence label (High/Medium/Low) per
  section from the Analyst, based on source agreement — a simple heuristic,
  not a calibrated probability model.
- "Self-evaluation / hypothesis verification" is one Evaluator reasoning
  pass per round, not a separate formal verification subsystem.
- Parallel execution uses Python's `ThreadPoolExecutor` inside the tool node,
  rather than LangGraph's own fan-out (`Send`) API — the effect (concurrent
  tool calls) is the same, the mechanism is simpler and more testable under
  time pressure.

## Context & Memory Management

- **Short-term (task-level) context** — the graph's shared state accumulates
  every plan, tool call, observation, and evaluation across the run, so each
  node is fully aware of everything before it. This full state is what the
  Analyst synthesizes from, not just a summary.
- **Long-term (cross-session) memory** — every search is persisted to a
  local SQLite database, tied to the logged-in user. Before starting a new
  search, the app pulls a short summary of that user's most recent past
  sessions and passes it to the Planner as known context, so it can note
  connections to prior research where relevant. Shown in the UI trace as a
  "📚 Long-Term Memory Loaded" step whenever past sessions exist.

## Evaluation

Two complementary pieces:

- **`eval_harness.py`** — automated evaluation across normal, ambiguous,
  contradictory, incomplete, adversarial, and tool-failure scenarios.
  Measures task completion, latency, planner rounds/tool calls (resource
  efficiency), tool failure→fallback→recovery, conflicting-evidence
  detection, and consistency across repeated runs. Groundedness and
  hallucination are scored by an LLM-as-judge (a second Gemini call that
  checks the final answer's claims against the raw findings it had access
  to). Also runs a **baseline comparison** — the same queries answered by a
  single plain Gemini call with no framework or tools — so the pipeline's
  groundedness score can be compared directly against it. Outputs
  `evaluation_results.json` (raw data) and `evaluation_report.md`
  (summary table).
  - ⚠️ **Cost note:** each test case costs several Gemini API calls (the
    pipeline + a judge call + a baseline call). Check remaining quota before
    running.
- **`human_eval_rubric.md`** — a scoring sheet for the criteria that need a
  person's judgment (accuracy, task completion quality, evidence quality),
  filled in by reviewing actual outputs rather than automated.

## Technologies Used
- **Framework:** LangGraph (`StateGraph`, conditional edges, `MemorySaver` checkpointing)
- **Model:** Google Gemini API (`gemini-3.5-flash-lite`) for reasoning and decision-making
- **Tool 1 — Web Search:** DuckDuckGo (via the `ddgs` Python library) — free, keyless
- **Tool 2 — arXiv API:** the public arXiv REST API — free, keyless, for academic research papers
- **UI:** Streamlit
- **Storage:** SQLite (users, search history)
- **Language:** Python

## Features
- Real LangGraph pipeline with dynamic planning, conditional routing, autonomous replanning, and checkpointing
- Two external tools, dynamically selected, with parallel execution and fallback-on-failure
- Self-evaluation each round: checks for sufficiency and conflicting evidence between sources, with deterministic loop/deadlock and resource-budget guards
- Confidence-labeled, conflict-aware final briefing from a dedicated Analyst step
- Short-term context shared across the whole pipeline; long-term memory of past sessions per user
- Username/password login (salted + hashed) and persistent, per-user search history in the sidebar
- Adversarial demo toggle to force and show live failure recovery
- Full visible reasoning trace in the UI for every step above
- Graceful error handling if a tool or the API is unavailable or rate-limited

## Installation / Setup
1. Clone this repo and `cd` into it
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
5. Copy `.env.example` to `.env` and add your Gemini API key:
   ```
   cp .env.example .env
   ```

## How to Run
```
streamlit run app.py
```
This opens the app in your browser (usually at `http://localhost:8501`). Log
in, enter a topic, technology area, or competitor name, and the pipeline will
plan, research, evaluate, and return a briefing.

## Screenshots / Demo Link
- **Live demo:** [add your deployed Streamlit link here]
- **Screenshots:** [add screenshots here, if included]
