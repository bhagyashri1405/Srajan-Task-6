import os
import json
import operator
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, Annotated, List, Dict, Any

from google import genai
from google.genai import types
from ddgs import DDGS

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    import streamlit as st
    _API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
except Exception:
    _API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL = "gemini-3.5-flash-lite"
MAX_ROUNDS = 3  # resource-aware execution: caps planner<->evaluator replanning loops
FALLBACK_TOOL = {"search_web": "search_arxiv", "search_arxiv": "search_web"}

client = genai.Client(api_key=_API_KEY)

# ============================================================================
# TOOLS (unchanged from before): DuckDuckGo web search + arXiv API
# ============================================================================


def _execute_web_search(query: str, max_results: int = 5) -> str:
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No web results found for this query."
    lines = []
    for r in results:
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        lines.append(f"- {title}: {body} (source: {href})")
    return "\n".join(lines)


_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _execute_arxiv_search(query: str, max_results: int = 5) -> str:
    params = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": max_results})
    url = f"http://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    entries = root.findall("atom:entry", _ARXIV_NS)
    if not entries:
        return "No arXiv papers found for this query."
    lines = []
    for entry in entries:
        title = (entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or "").strip().replace("\n", " ")
        link = (entry.findtext("atom:id", default="", namespaces=_ARXIV_NS) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=_ARXIV_NS) or "").strip()[:10]
        snippet = summary[:280] + ("..." if len(summary) > 280 else "")
        lines.append(f"- {title} ({published}): {snippet} (source: {link})")
    return "\n".join(lines)


_TOOL_EXECUTORS = {"search_web": _execute_web_search, "search_arxiv": _execute_arxiv_search}


def _run_tool_with_fallback(tool: str, query: str, simulate_failure: bool) -> tuple[str, list]:
    """
    Runs one tool call. If it fails (or simulate_failure forces a synthetic
    failure for demo purposes), falls back to the other tool for the same
    query. Returns (observation_text, list_of_trace_entries).
    """
    entries = []
    try:
        if simulate_failure:
            raise RuntimeError("Simulated failure (adversarial demo mode is ON)")
        result = _TOOL_EXECUTORS[tool](query)
        entries.append({"type": "action", "content": f'{tool}("{query}")'})
        entries.append({"type": "observation", "content": result})
        return result, entries
    except Exception as exc:
        entries.append({"type": "action", "content": f'{tool}("{query}")'})
        entries.append({"type": "tool_failure", "content": f"{tool} failed: {exc}"})
        fallback = FALLBACK_TOOL[tool]
        try:
            result = _TOOL_EXECUTORS[fallback](query)
            entries.append({"type": "tool_fallback", "content": f"Falling back to {fallback}(\"{query}\")"})
            entries.append({"type": "observation", "content": result})
            return result, entries
        except Exception as exc2:
            entries.append({"type": "tool_fallback", "content": f"Fallback {fallback} also failed: {exc2}"})
            failure_note = f"Both {tool} and its fallback {fallback} failed for query '{query}'."
            entries.append({"type": "observation", "content": failure_note})
            return failure_note, entries


# ============================================================================
# SHARED STATE (LangGraph)
# ============================================================================


class ToolCallSpec(TypedDict):
    tool: str
    query: str


class AgentState(TypedDict):
    user_input: str
    long_term_context: str
    findings: Annotated[List[str], operator.add]
    trace: Annotated[List[Dict[str, Any]], operator.add]
    tried_actions: Annotated[List[str], operator.add]
    round_count: int
    plan: List[ToolCallSpec]
    need_more_research: bool
    conflict_notes: str
    final_report: str
    simulate_failure: bool


# ============================================================================
# NODE 1: Planner — dynamic planning + adaptive task decomposition.
# Also the entry point for autonomous replanning (this node runs again if
# the Evaluator decides more research is needed).
# ============================================================================

_PLAN_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "need_more_research": types.Schema(type=types.Type.BOOLEAN),
        "reasoning": types.Schema(type=types.Type.STRING),
        "tool_calls": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "tool": types.Schema(type=types.Type.STRING, enum=["search_web", "search_arxiv"]),
                    "query": types.Schema(type=types.Type.STRING),
                },
                required=["tool", "query"],
            ),
            max_items=2,
        ),
    },
    required=["need_more_research", "reasoning", "tool_calls"],
)


def planner_node(state: AgentState) -> dict:
    findings_so_far = "\n".join(state.get("findings", [])) or "(none yet)"
    tried = "\n".join(state.get("tried_actions", [])) or "(none yet)"
    context_block = f"\nKnown long-term context from past sessions:\n{state['long_term_context']}\n" if state.get("long_term_context") else ""

    prompt = (
        f"User's request: {state['user_input']}\n{context_block}\n"
        f"Findings gathered so far:\n{findings_so_far}\n\n"
        f"Actions already tried (avoid exact repeats):\n{tried}\n\n"
        "Decide: do you need more research, and if so, which 1-2 tool calls should run next "
        "(search_web for competitor/news/patent/general info, search_arxiv for academic papers)? "
        "Only mark need_more_research=false once you genuinely have enough to answer well."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are the Planning module of a research agent. You decide what to investigate "
                "next, dynamically, based on what's already been found. Skip categories that don't "
                "apply to this topic. Be specific and targeted with queries."
            ),
            response_mime_type="application/json",
            response_schema=_PLAN_SCHEMA,
        ),
    )

    try:
        plan_data = json.loads(response.text)
    except Exception:
        plan_data = {"need_more_research": False, "reasoning": "Planner output could not be parsed.", "tool_calls": []}

    trace_entry = {"type": "plan", "content": f"Round {state.get('round_count', 0) + 1} plan: {plan_data.get('reasoning', '')}"}

    return {
        "plan": plan_data.get("tool_calls", [])[:2],
        "need_more_research": plan_data.get("need_more_research", False) and bool(plan_data.get("tool_calls")),
        "round_count": state.get("round_count", 0) + 1,
        "trace": [trace_entry],
    }


# ============================================================================
# NODE 2: Execute tools — runs the planner's tool calls. When the plan has
# two calls, they run concurrently via ThreadPoolExecutor (parallel
# execution). Each call has try/except fallback to the other tool
# (tool fallback + failure recovery), with an optional forced failure for
# the adversarial live-test demo.
# ============================================================================


def execute_tools_node(state: AgentState) -> dict:
    plan = state.get("plan", [])
    simulate_failure = state.get("simulate_failure", False)

    all_trace = []
    all_findings = []
    tried = []

    if not plan:
        return {"findings": [], "trace": [], "tried_actions": []}

    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        futures = [pool.submit(_run_tool_with_fallback, call["tool"], call["query"], simulate_failure) for call in plan]
        for call, future in zip(plan, futures):
            observation, entries = future.result()
            all_findings.append(observation)
            all_trace.extend(entries)
            tried.append(f"{call['tool']}(\"{call['query']}\")")

    return {"findings": all_findings, "trace": all_trace, "tried_actions": tried}


# ============================================================================
# NODE 3: Evaluator — self-evaluation + conflicting-evidence detection +
# loop/deadlock detection + resource-aware execution (hard caps below are
# deterministic code, not left to the LLM, so they can't be talked out of).
# Routes back to the Planner (replanning) or forward to the Analyst.
# ============================================================================

_EVAL_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sufficient": types.Schema(type=types.Type.BOOLEAN),
        "reasoning": types.Schema(type=types.Type.STRING),
        "conflicts_found": types.Schema(type=types.Type.BOOLEAN),
        "conflict_notes": types.Schema(type=types.Type.STRING),
    },
    required=["sufficient", "reasoning", "conflicts_found", "conflict_notes"],
)


def evaluator_node(state: AgentState) -> dict:
    round_count = state.get("round_count", 0)
    tried_actions = state.get("tried_actions", [])

    # Deterministic deadlock check: same exact action requested twice.
    deadlock = len(tried_actions) != len(set(tried_actions))

    if round_count >= MAX_ROUNDS or deadlock:
        reason = "resource budget (max rounds) reached" if round_count >= MAX_ROUNDS else "repeated identical action detected (deadlock guard)"
        return {
            "need_more_research": False,
            "trace": [{"type": "evaluator", "content": f"Stopping further research: {reason}."}],
        }

    findings_so_far = "\n".join(state.get("findings", [])) or "(none yet)"
    prompt = (
        f"User's request: {state['user_input']}\n\n"
        f"Findings gathered so far:\n{findings_so_far}\n\n"
        "Is this sufficient to write a good briefing? Do any findings conflict with each other "
        "(e.g. web and academic sources disagree)? Note any conflicts explicitly."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are the Evaluator module. You judge whether gathered research is sufficient "
                "and flag any conflicting evidence between sources. You do not do research yourself."
            ),
            response_mime_type="application/json",
            response_schema=_EVAL_SCHEMA,
        ),
    )

    try:
        eval_data = json.loads(response.text)
    except Exception:
        eval_data = {"sufficient": True, "reasoning": "Evaluator output could not be parsed; proceeding.", "conflicts_found": False, "conflict_notes": ""}

    trace_entry = {"type": "evaluator", "content": eval_data.get("reasoning", "")}
    if eval_data.get("conflicts_found"):
        trace_entry["content"] += f" ⚠️ Conflicting evidence noted: {eval_data.get('conflict_notes', '')}"

    return {
        "need_more_research": not eval_data.get("sufficient", True),
        "conflict_notes": eval_data.get("conflict_notes", "") if eval_data.get("conflicts_found") else state.get("conflict_notes", ""),
        "trace": [trace_entry],
    }


def route_after_evaluator(state: AgentState) -> str:
    """Conditional edge: autonomous replanning loop vs. moving to synthesis."""
    return "planner" if state.get("need_more_research") else "analyst"


# ============================================================================
# NODE 4: Analyst — synthesis only, no tools. Produces the final briefing,
# incorporating uncertainty/confidence and any flagged conflicts.
# ============================================================================

ANALYST_SYSTEM_PROMPT = (
    "You are the Analyst Agent. You have no search tools and cannot look anything up — your "
    "only job is to synthesize what the Research pipeline already found. You will be given the "
    "user's original request, all gathered findings, and any noted conflicts between sources. "
    "Organize the findings into a concise report using markdown headings only for the categories "
    "genuinely supported by the findings (e.g. '## Research Trends', '## Patent Activity', "
    "'## Competitor Activity', '## Industry News') — omit a heading entirely if findings don't "
    "support it. Under each heading, give tight bullet points on what changed and why it matters. "
    "For each section, end with a one-line confidence note (High/Medium/Low confidence) based on "
    "how well-supported and consistent the findings were. If conflicts were noted, address them "
    "explicitly — state what disagrees and, if possible, which source seems more reliable and why. "
    "Be concise: this is a briefing, not an essay."
)


def analyst_node(state: AgentState) -> dict:
    findings_text = "\n".join(state.get("findings", [])) or "(no findings were gathered)"
    conflict_block = f"\nNoted conflicts between sources:\n{state['conflict_notes']}\n" if state.get("conflict_notes") else ""
    prompt = (
        f"User's original request: {state['user_input']}\n\n"
        f"All gathered findings:\n{findings_text}\n{conflict_block}\n"
        "Synthesize the final briefing now."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=ANALYST_SYSTEM_PROMPT),
    )
    final_report = (response.text or "").strip()
    return {"final_report": final_report, "trace": [{"type": "analyst_final", "content": final_report}]}


# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

_graph = StateGraph(AgentState)
_graph.add_node("planner", planner_node)
_graph.add_node("execute_tools", execute_tools_node)
_graph.add_node("evaluator", evaluator_node)
_graph.add_node("analyst", analyst_node)

_graph.set_entry_point("planner")
_graph.add_edge("planner", "execute_tools")
_graph.add_edge("execute_tools", "evaluator")
_graph.add_conditional_edges("evaluator", route_after_evaluator, {"planner": "planner", "analyst": "analyst"})
_graph.add_edge("analyst", END)

_checkpointer = MemorySaver()
_compiled_graph = _graph.compile(checkpointer=_checkpointer)


# ============================================================================
# PUBLIC ENTRY POINT — same interface app.py already calls.
# ============================================================================


def run_agent(user_input: str, long_term_context: str = "", simulate_failure: bool = False, thread_id: str = "default") -> dict:
    """
    Runs the full LangGraph pipeline: Planner -> (parallel) tool execution ->
    Evaluator -> [loop back to Planner, or proceed] -> Analyst. Checkpointed
    via LangGraph's MemorySaver, keyed by thread_id (pass a per-user/session
    id from the caller so each conversation has its own checkpoint history).
    """
    initial_state: AgentState = {
        "user_input": user_input,
        "long_term_context": long_term_context,
        "findings": [],
        "trace": [],
        "tried_actions": [],
        "round_count": 0,
        "plan": [],
        "need_more_research": True,
        "conflict_notes": "",
        "final_report": "",
        "simulate_failure": simulate_failure,
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = _compiled_graph.invoke(initial_state, config=config)
    except Exception as exc:
        return {
            "answer": f"The agent pipeline stopped due to an error: {exc}",
            "used_search": False,
            "search_queries": [],
            "trace": [{"type": "analyst_final", "content": f"Pipeline error: {exc}"}],
        }

    trace = list(final_state.get("trace", []))
    trace.append({"type": "checkpoint", "content": f"Checkpoint saved (thread_id={thread_id})."})

    search_queries = final_state.get("tried_actions", [])

    return {
        "answer": final_state.get("final_report", ""),
        "used_search": len(search_queries) > 0,
        "search_queries": search_queries,
        "trace": trace,
    }
