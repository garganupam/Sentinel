import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from indexing.retriever import retrieve_context
from indexing.store import VectorStore

from .aggregator import aggregate_issues
from .agents import run_agent
from .github_client import ChangedFile
from .llm_client import LLMClient
from .models import AgentType, ReviewIssue


class ReviewState(TypedDict):
    """Shared state threaded through the review graph.

    `context` is written once by the retrieval node, before the fan-out —
    no reducer needed, same as `aggregated`. `issues` uses `operator.add`
    as its reducer: all 4 agent branches run in parallel and each
    contributes its own list. Without a reducer here, LangGraph's parallel
    writes to the same key would overwrite each other instead of combining.
    """
    files: list[ChangedFile]
    context: str
    issues: Annotated[list[ReviewIssue], operator.add]
    aggregated: list[ReviewIssue]


class _AgentTask(TypedDict):
    """Per-branch payload Send hands to one agent node.

    Deliberately NOT the full ReviewState — each parallel branch only
    needs to know which agent it is, which files to look at, and the
    shared retrieved context (same for every branch).
    """
    agent_type: AgentType
    files: list[ChangedFile]
    context: str


def build_review_graph(llm: LLMClient, store: VectorStore, rag_debug: bool = False):
    """Build and compile the M2+M3 review graph.

    Flow: START → retrieve shared project context (M3) → fan out to all 4
    agents in parallel (Send API) → once every branch finishes, fan in to
    `aggregate` → END.

    Args:
        llm:   Shared Gemini client. Captured by closure into the agent
               node — kept out of graph state since it's a live
               connection, not data to be threaded or checkpointed.
        store: VectorStore for the repo being reviewed. Also captured by
               closure, same reasoning as llm. Locked design: retrieval
               failure here must never block the review — retrieve_context
               already degrades to a "no context" message on its own, so
               this node has no extra error handling of its own to add.

    Returns:
        A compiled graph. Call `.invoke({"files": [...], "context": "",
        "issues": [], "aggregated": []})` to run it; the result's
        `"aggregated"` key holds the final deduped issue list.
    """

    def retrieve_node(state: ReviewState) -> dict:
        """Entry node — fetches shared project context once, before the
        4 agents fan out. One retrieval, not four — locked design."""
        context = retrieve_context(state["files"], store, rag_debug)
        return {"context": context}

    def fan_out(state: ReviewState) -> list[Send]:
        """Routing function after retrieval: one Send per agent type.

        Same `files` and `context` go to every branch — only `agent_type`
        differs.
        """
        return [
            Send("run_agent", {
                "agent_type": agent_type,
                "files": state["files"],
                "context": state["context"],
            })
            for agent_type in AgentType
        ]

    def run_agent_node(task: _AgentTask) -> dict:
        """One parallel branch — runs a single specialist agent."""
        issues = run_agent(task["agent_type"], task["files"], llm, context=task["context"])
        return {"issues": issues}

    def aggregate_node(state: ReviewState) -> dict:
        """Fan-in node — runs once, after all 4 agent branches complete."""
        return {"aggregated": aggregate_issues(state["issues"])}

    graph = StateGraph(ReviewState)
    graph.add_node("retrieve_context", retrieve_node)
    graph.add_node("run_agent", run_agent_node)
    graph.add_node("aggregate", aggregate_node)

    graph.add_edge(START, "retrieve_context")
    graph.add_conditional_edges("retrieve_context", fan_out, ["run_agent"])
    graph.add_edge("run_agent", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()