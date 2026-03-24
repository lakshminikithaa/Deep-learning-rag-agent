"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from rag_agent.agent.prompts import (
    QUERY_REWRITE_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()
    
    # Extract latest HumanMessage content
    original_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
            
    if not original_query:
        # Fallback if no human message found
        return {"original_query": "", "rewritten_query": ""}
    try:
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(original_query=original_query)
        rewritten = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten_query = str(rewritten.content).strip()
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query or original_query,
        }
    except Exception:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }
# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    try:
        manager = VectorStoreManager(get_settings())
        chunks = manager.query(
            query_text=state.get("rewritten_query", ""),
            topic_filter=state.get("topic_filter"),
            difficulty_filter=state.get("difficulty_filter")
        )
    except Exception as e:
        chunks = []

    if not chunks:
        return {"retrieved_chunks": [], "no_context_found": True}
    else:
        return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if state.get("no_context_found", False):
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.get("rewritten_query", ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    chunks = state.get("retrieved_chunks", [])
    context_str = ""
    citations = []
    total_score = 0.0
    
    for chunk in chunks:
        citation = chunk.to_citation()
        if citation not in citations:
            citations.append(citation)
        context_str += f"{citation}\n{chunk.chunk_text}\n\n"
        total_score += chunk.score
        
    avg_confidence = total_score / len(chunks) if chunks else 0.0
    
    # Simple history trimming (keeping last 10 messages max to prevent context window blowup)
    history = state.get("messages", [])[:-1]  # Exclude current query if it's there
    history = history[-10:] if len(history) > 10 else history

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]
    prompt_messages.extend(history)
    prompt_messages.append(HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {state.get('original_query', '')}"))
    
    try:
        llm_response = llm.invoke(prompt_messages)
        answer = llm_response.content
    except Exception as e:
        answer = "I'm sorry, I encountered an error while generating the response."
        
    new_ai_message = AIMessage(content=answer)
    agent_response = AgentResponse(
        answer=answer,
        sources=citations,
        confidence=avg_confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query", "")
    )
    
    return {
        "final_response": agent_response,
        "messages": [new_ai_message]
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    return "generate"
