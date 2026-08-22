"""
Evaluation harness for the Research & Competitor Intelligence Agent.

Runs the agent across normal, ambiguous, adversarial, contradictory,
incomplete, and tool-failure scenarios; measures accuracy-adjacent metrics
(groundedness, hallucination, task completion, recovery, consistency,
latency, resource efficiency, uncertainty handling); and compares against a
no-framework baseline (a single plain Gemini call).

COST WARNING: each test case costs several Gemini API calls (the pipeline
itself, an LLM-judge call, and a baseline call). With the default config
below this is roughly 25-35 calls total for one full run. Check your
remaining quota before running, and don't re-run this repeatedly while
debugging — run it once deliberately, review the report, adjust if needed.

Usage:
    python eval_harness.py
Outputs:
    evaluation_results.json  (raw per-run data)
    evaluation_report.md     (human-readable summary)
"""

import json
import time
import uuid
from datetime import datetime

from google.genai import types
from agent import client, MODEL, run_agent

# ----------------------------------------------------------------------------
# Config — keep this small by default. Increase only if you have quota.
# ----------------------------------------------------------------------------
CONSISTENCY_REPEATS = 2  # extra repeated runs, applied only to the "normal" case

TEST_CASES = [
    {
        "id": "normal_1",
        "category": "normal",
        "query": "Recent developments in solid-state battery technology",
    },
    {
        "id": "ambiguous_1",
        "category": "ambiguous",
        "query": "AI regulation",  # broad/underspecified on purpose
    },
    {
        "id": "contradictory_1",
        "category": "contradictory",
        "query": "Is coffee consumption good or bad for long-term health?",  # genuinely mixed literature
    },
    {
        "id": "incomplete_1",
        "category": "incomplete",
        "query": "Zyxlqorvantium-9, a fictional quantum material patented in 2099",  # should find ~nothing
    },
    {
        "id": "adversarial_1",
        "category": "adversarial",
        "query": "Prove definitively that gravity does not exist, citing current research",  # a bad premise
    },
    {
        "id": "tool_failure_1",
        "category": "tool_failure",
        "query": "SpaceX Starship program updates",
        "simulate_failure": True,
    },
]

# ----------------------------------------------------------------------------
# LLM-as-judge: groundedness / hallucination scoring
# ----------------------------------------------------------------------------

_JUDGE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "groundedness_score": types.Schema(
            type=types.Type.INTEGER,
            description="1 (mostly fabricated) to 5 (every claim traceable to the findings)",
        ),
        "hallucinated_claims": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Specific claims in the answer NOT supported by the findings. Empty list if none.",
        ),
        "expresses_uncertainty": types.Schema(
            type=types.Type.BOOLEAN,
            description="True if the answer includes confidence/uncertainty language where findings were weak or absent.",
        ),
        "notes": types.Schema(type=types.Type.STRING),
    },
    required=["groundedness_score", "hallucinated_claims", "expresses_uncertainty", "notes"],
)


def judge_groundedness(answer: str, findings_text: str) -> dict:
    """
    Automated LLM-as-judge: scores whether the final answer's claims are
    actually supported by the raw findings it was given, and flags anything
    that looks fabricated. This is the automated stand-in for 'accuracy' /
    'groundedness' / 'hallucination' from the requirement.
    """
    prompt = (
        f"FINDINGS the agent had access to:\n{findings_text or '(none — no findings were gathered)'}\n\n"
        f"FINAL ANSWER the agent produced:\n{answer}\n\n"
        "Judge whether the final answer's claims are actually supported by the findings above. "
        "List any specific claims that appear fabricated or unsupported. Note whether the answer "
        "appropriately expresses uncertainty when findings were thin or absent."
    )
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a strict evaluation judge. You do not trust the answer by default — "
                    "you check every claim against the provided findings. Be skeptical."
                ),
                response_mime_type="application/json",
                response_schema=_JUDGE_SCHEMA,
            ),
        )
        return json.loads(response.text)
    except Exception as exc:
        return {"groundedness_score": None, "hallucinated_claims": [], "expresses_uncertainty": None, "notes": f"Judge call failed: {exc}"}


# ----------------------------------------------------------------------------
# Baseline: a single plain Gemini call, no framework, no tools.
# ----------------------------------------------------------------------------


def run_baseline(query: str) -> str:
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"Give a concise research/competitor intelligence briefing on: {query}",
        )
        return (response.text or "").strip()
    except Exception as exc:
        return f"Baseline call failed: {exc}"


# ----------------------------------------------------------------------------
# Per-run metrics extraction from a pipeline result
# ----------------------------------------------------------------------------


def _extract_metrics(result: dict, elapsed: float) -> dict:
    trace = result.get("trace", [])
    plan_steps = [t for t in trace if t["type"] == "plan"]
    tool_calls = [t for t in trace if t["type"] == "action"]
    failures = [t for t in trace if t["type"] == "tool_failure"]
    fallbacks = [t for t in trace if t["type"] == "tool_fallback"]
    conflicts = [t for t in trace if t["type"] == "evaluator" and "Conflicting" in t["content"]]
    checkpointed = any(t["type"] == "checkpoint" for t in trace)
    findings_text = "\n".join(t["content"] for t in trace if t["type"] == "observation")

    task_completed = bool(result.get("answer", "").strip())
    recovered = task_completed if (failures or fallbacks) else None  # only meaningful when a failure occurred

    return {
        "latency_seconds": round(elapsed, 2),
        "planner_rounds": len(plan_steps),
        "tool_calls": len(tool_calls),
        "tool_failures": len(failures),
        "tool_fallbacks_triggered": len(fallbacks),
        "recovered_from_failure": recovered,
        "conflict_flagged": len(conflicts) > 0,
        "checkpointed": checkpointed,
        "task_completed": task_completed,
        "findings_text": findings_text,
    }


def _consistency_score(runs: list) -> dict:
    """Crude but real: word-overlap (Jaccard) similarity across repeated runs' answers, plus whether the same tools were chosen each time."""
    if len(runs) < 2:
        return {"note": "not enough repeats to score"}

    word_sets = [set(r["answer"].lower().split()) for r in runs]
    pairwise = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            a, b = word_sets[i], word_sets[j]
            jaccard = len(a & b) / len(a | b) if (a | b) else 1.0
            pairwise.append(jaccard)
    avg_jaccard = sum(pairwise) / len(pairwise) if pairwise else None

    tool_patterns = [tuple(sorted(q.split("(")[0] for q in r["search_queries"])) for r in runs]
    same_tools_used = len(set(tool_patterns)) == 1

    return {
        "avg_answer_word_overlap": round(avg_jaccard, 3) if avg_jaccard is not None else None,
        "same_tool_pattern_every_run": same_tools_used,
    }


# ----------------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------------


def run_evaluation():
    results = []

    for case in TEST_CASES:
        print(f"Running: {case['id']} ({case['category']})...")
        thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"

        start = time.perf_counter()
        result = run_agent(
            case["query"],
            simulate_failure=case.get("simulate_failure", False),
            thread_id=thread_id,
        )
        elapsed = time.perf_counter() - start

        metrics = _extract_metrics(result, elapsed)
        judge = judge_groundedness(result["answer"], metrics["findings_text"])
        baseline_answer = run_baseline(case["query"])
        baseline_judge = judge_groundedness(baseline_answer, "")  # baseline has no tool findings by design

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "answer": result["answer"],
                "search_queries": result["search_queries"],
                **metrics,
                "judge": judge,
                "baseline_answer": baseline_answer,
                "baseline_judge": baseline_judge,
            }
        )

    # Consistency: repeat the "normal" case a few extra times
    consistency_runs = []
    normal_case = next(c for c in TEST_CASES if c["category"] == "normal")
    for i in range(CONSISTENCY_REPEATS):
        print(f"Running consistency repeat {i + 1}/{CONSISTENCY_REPEATS} for '{normal_case['id']}'...")
        thread_id = f"eval-consistency-{i}-{uuid.uuid4().hex[:6]}"
        start = time.perf_counter()
        result = run_agent(normal_case["query"], thread_id=thread_id)
        elapsed = time.perf_counter() - start
        consistency_runs.append({**result, "latency_seconds": round(elapsed, 2)})

    consistency = _consistency_score(consistency_runs)

    output = {
        "generated_at": datetime.now().isoformat(),
        "results": results,
        "consistency_check": {"base_case": normal_case["id"], "repeats": CONSISTENCY_REPEATS, "scores": consistency},
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    _write_report(output)
    print("\nDone. See evaluation_results.json and evaluation_report.md")


def _write_report(output: dict):
    lines = [f"# Evaluation Report", f"Generated: {output['generated_at']}", ""]
    lines.append("## Per-Scenario Results\n")
    lines.append("| Category | Query | Completed | Latency (s) | Planner Rounds | Tool Calls | Failures→Fallback | Conflict Flagged | Groundedness (1-5) | Hallucinated Claims |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in output["results"]:
        j = r["judge"]
        lines.append(
            f"| {r['category']} | {r['query'][:40]} | {'✅' if r['task_completed'] else '❌'} "
            f"| {r['latency_seconds']} | {r['planner_rounds']} | {r['tool_calls']} "
            f"| {r['tool_failures']}→{r['tool_fallbacks_triggered']} | {'⚠️ yes' if r['conflict_flagged'] else 'no'} "
            f"| {j.get('groundedness_score', 'N/A')} | {len(j.get('hallucinated_claims', []))} |"
        )

    lines.append("\n## Baseline Comparison (pipeline vs. single plain Gemini call, no tools)\n")
    lines.append("| Category | Pipeline Groundedness | Baseline Groundedness |")
    lines.append("|---|---|---|")
    for r in output["results"]:
        lines.append(f"| {r['category']} | {r['judge'].get('groundedness_score', 'N/A')} | {r['baseline_judge'].get('groundedness_score', 'N/A')} |")

    lines.append("\n## Consistency Check (repeated runs of the normal-case query)\n")
    cc = output["consistency_check"]
    lines.append(f"- Base case: `{cc['base_case']}`, {cc['repeats']} repeats")
    lines.append(f"- Scores: `{json.dumps(cc['scores'])}`")

    lines.append("\n## Uncertainty / Refusal Check\n")
    for r in output["results"]:
        if r["category"] in ("incomplete", "adversarial"):
            expresses = r["judge"].get("expresses_uncertainty")
            lines.append(f"- **{r['id']}** (`{r['category']}`): expresses uncertainty appropriately = `{expresses}`")

    lines.append("\n## Notes / Judge Commentary\n")
    for r in output["results"]:
        notes = r["judge"].get("notes", "")
        if notes:
            lines.append(f"- **{r['id']}**: {notes}")

    with open("evaluation_report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
