from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from app.retriever import retrieve
from app.config import *

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
)

class AgentState(TypedDict):
    question: str
    context: list
    answer: str
    sources: list
    retrieval_passes: int
    sufficient_context: bool

def retrieve_node(state: AgentState) -> AgentState:
    query = state["question"]
    passes = state.get("retrieval_passes", 0)

    if passes > 0:
        rewrite = llm.invoke(
            f"Rephrase this to find more specific information: {query}"
        ).content
        query = rewrite

    docs = retrieve(query, top_k=TOP_K_RESULTS)
    return {
        **state,
        "context": state.get("context", []) + docs,
        "retrieval_passes": passes + 1
    }

def check_context_node(state: AgentState) -> AgentState:
    if not state["context"]:
        return {**state, "sufficient_context": False}

    context_text = " ".join([d["text"] for d in state["context"][:3]])
    check_prompt = f"""
    Question: {state["question"]}
    Retrieved context: {context_text[:600]}
    Is this context sufficient to answer the question well?
    Reply with only YES or NO.
    """
    response = llm.invoke(check_prompt).content.strip().upper()
    return {**state, "sufficient_context": "YES" in response}

def generate_node(state: AgentState) -> AgentState:
    context_parts = [
        f"[Source: {d['source']}]\n{d['text']}"
        for d in state["context"]
    ]
    context_str = "\n\n".join(context_parts)

    prompt = f"""You are a research assistant. Answer the question using ONLY
the provided context. For every claim, cite the source in brackets.
If the context does not contain the answer, say so clearly.

Context:
{context_str}

Question: {state["question"]}

Answer (with citations):"""

    answer = llm.invoke(prompt).content
    sources = list(set(d["source"] for d in state["context"]))
    return {**state, "answer": answer, "sources": sources}

def should_retrieve_again(state: AgentState) -> str:
    if state.get("sufficient_context") or state.get("retrieval_passes", 0) >= 2:
        return "generate"
    return "retrieve"

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("check_context", check_context_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "check_context")
graph.add_conditional_edges(
    "check_context",
    should_retrieve_again,
    {"retrieve": "retrieve", "generate": "generate"}
)
graph.add_edge("generate", END)

app_graph = graph.compile()

async def run_agent(question: str) -> dict:
    result = app_graph.invoke({
        "question": question,
        "context": [],
        "retrieval_passes": 0,
        "sufficient_context": False,
        "answer": "",
        "sources": [],
    })
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieval_passes": result["retrieval_passes"]
    }