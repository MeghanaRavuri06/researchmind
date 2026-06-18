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

def retrieve_node(state: AgentState) -> AgentState:
    docs = retrieve(state["question"], top_k=TOP_K_RESULTS)
    return {
        **state,
        "context": docs,
        "retrieval_passes": 1
    }

def generate_node(state: AgentState) -> AgentState:
    if not state["context"]:
        return {
            **state,
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": []
        }

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

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app_graph = graph.compile()

async def run_agent(question: str) -> dict:
    result = app_graph.invoke({
        "question": question,
        "context": [],
        "retrieval_passes": 0,
        "answer": "",
        "sources": [],
    })
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieval_passes": result["retrieval_passes"]
    }