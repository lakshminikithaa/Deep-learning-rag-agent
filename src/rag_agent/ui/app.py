"""
app.py
======
Upgraded Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>

        /* -------------------- APP BACKGROUND -------------------- */
        .stApp {
            background:
                radial-gradient(circle at top left, #172554 0%, transparent 28%),
                radial-gradient(circle at top right, #312e81 0%, transparent 24%),
                linear-gradient(135deg, #0f172a 0%, #111827 50%, #020617 100%);
            color: #e5e7eb;
        }

        html, body, [class*="css"] {
            color: #e5e7eb !important;
        }

        /* -------------------- TEXT -------------------- */
        .hero-subtitle,
        .section-subtitle,
        .muted,
        .footer-note {
            color: #cbd5e1 !important;
        }

        .section-title,
        .doc-name {
            color: #f8fafc !important;
            font-weight: 700;
        }

        /* -------------------- SIDEBAR -------------------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        /* -------------------- FILE UPLOADER -------------------- */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(30, 41, 59, 0.92) !important;
            border: 1px dashed rgba(125, 211, 252, 0.35) !important;
            border-radius: 16px !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #ffffff !important;
        }

        /* Browse button fix */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] button * {
            background: #334155 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #475569 !important;
        }

        /* Uploaded file */
        [data-testid="stFileUploaderFile"] {
            background: rgba(30, 41, 59, 0.85) !important;
            border-radius: 12px !important;
        }

        [data-testid="stFileUploaderFile"] * {
            color: #ffffff !important;
        }

        /* -------------------- SELECT BOX -------------------- */
        [data-baseweb="select"] > div {
            background-color: rgba(15, 23, 42, 0.92) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
        }

        [data-baseweb="select"] *,
        [data-baseweb="select"] span {
            color: #ffffff !important;
        }

        .stSelectbox label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* -------------------- BUTTONS -------------------- */
        .stButton > button {
            width: 100%;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
            color: white !important;
            font-weight: 600 !important;
        }

        /* Delete button */
        .stButton button[kind="secondary"] {
            background: rgba(30, 41, 59, 0.95) !important;
            border: 1px solid rgba(248, 113, 113, 0.30) !important;
            color: #fca5a5 !important;
        }

        /* -------------------- CHAT INPUT -------------------- */

        /* White box */
        div[data-testid="stChatInput"] {
            background: #ffffff !important;
            border-radius: 16px !important;
            border: 1px solid #e5e7eb !important;
        }

        /* BLACK typed text */
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            caret-color: #111827 !important;
            background: #ffffff !important;
        }

        /* Placeholder */
        div[data-testid="stChatInput"] textarea::placeholder,
        div[data-testid="stChatInput"] input::placeholder {
            color: #6b7280 !important;
            opacity: 1 !important;
        }

        /* -------------------- BADGES -------------------- */
        .badge {
            background: rgba(30, 41, 59, 0.82);
            color: #dbeafe !important;
            padding: 4px 8px;
            border-radius: 999px;
        }

        /* -------------------- FINAL FIX: ANSWER TEXT -------------------- */
        [data-testid="stChatMessage"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }

        [data-testid="stChatMessage"] p {
            color: #ffffff !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(settings) -> None:
    st.markdown(
        f"""
        <div class="hero-title">🧠 {settings.app_title}</div>
        <div class="hero-subtitle">
            RAG-powered deep learning interview preparation with LangGraph, ChromaDB, and a searchable study corpus.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_ribbon() -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">How to use</div>
            <div class="section-subtitle">
                Upload PDF/Markdown study material, inspect chunks in the viewer, and ask interview-style questions grounded in your corpus.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def refresh_documents(store: VectorStoreManager) -> None:
    st.session_state["ingested_documents"] = store.list_documents()


def _save_uploaded_files(uploaded_files: list) -> list[Path]:
    saved_paths: list[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))

    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    st.sidebar.markdown("## 📂 Corpus Ingestion")
    st.sidebar.caption("Upload clean study material in PDF or Markdown format.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "🚀 Ingest Documents",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if ingest_clicked and uploaded_files:
        try:
            with st.sidebar.spinner("Saving files..."):
                file_paths = _save_uploaded_files(uploaded_files)

            with st.sidebar.spinner("Chunking documents..."):
                chunks = chunker.chunk_files(file_paths)

            with st.sidebar.spinner("Ingesting into vector store..."):
                result = store.ingest(chunks)

            st.session_state["last_ingestion_result"] = result
            refresh_documents(store)

            if result.ingested > 0:
                st.sidebar.success(
                    f"✅ {result.ingested} chunks added, {result.skipped} duplicates skipped"
                )
            elif result.skipped > 0 and not result.errors:
                st.sidebar.warning(
                    f"⚠️ No new chunks added. {result.skipped} duplicates skipped."
                )
            else:
                st.sidebar.warning("No chunks were added.")

            if result.errors:
                with st.sidebar.expander("⚠️ Ingestion errors"):
                    for error in result.errors:
                        st.error(error)

        except Exception as exc:
            st.sidebar.error(f"Failed to ingest documents: {exc}")

    st.sidebar.divider()
    st.sidebar.markdown("### 📚 Ingested Documents")

    docs = st.session_state.get("ingested_documents", [])
    if not docs:
        st.sidebar.info("No documents ingested yet.")
        return

    current_selected = st.session_state.get("selected_document")

    for doc in docs:
        source = doc.get("source", "unknown")
        topic = doc.get("topic", "unknown")
        chunk_count = doc.get("chunk_count", 0)

        selected_class = "doc-selected" if current_selected == source else ""
        st.sidebar.markdown(
            f"""
            <div class="mini-card">
                <div class="doc-name {selected_class}">📄 {source}</div>
                <div class="muted">Topic: {topic}</div>
                <div class="muted">Chunks: {chunk_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.sidebar.columns([3, 1])
        with col_a:
            if st.button(
                "View",
                key=f"view_{source}",
                use_container_width=True,
            ):
                st.session_state["selected_document"] = source

        with col_b:
            if st.button(
                "🗑 Delete",
                key=f"delete_{source}",
                help=f"Remove {source}",
                use_container_width=True,
                type="secondary",
            ):
                deleted_count = store.delete_document(source)
                refresh_documents(store)

                if st.session_state.get("selected_document") == source:
                    st.session_state["selected_document"] = None

                if deleted_count > 0:
                    st.sidebar.success(f"Removed {source} ({deleted_count} chunks).")
                else:
                    st.sidebar.warning(f"No chunks found for {source}.")

        st.sidebar.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)


def render_corpus_stats(store: VectorStoreManager) -> None:
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Corpus Health")

    try:
        stats = store.get_collection_stats()
    except Exception as exc:
        st.sidebar.error(f"Unable to load corpus stats: {exc}")
        return

    st.sidebar.metric("Total Chunks", stats["total_chunks"])

    topics = stats.get("topics", [])
    if topics:
        st.sidebar.write("**Topics**")
        st.sidebar.caption(", ".join(topics))
    else:
        st.sidebar.caption("No topics available yet.")

    if stats.get("bonus_topics_present", False):
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.info("ℹ️ No bonus topics yet")


def render_document_viewer(store: VectorStoreManager) -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">📄 Document Viewer</div>
            <div class="section-subtitle">
                Inspect how your corpus was chunked and what metadata the retriever can use.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)

    docs = st.session_state.get("ingested_documents", [])
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [doc["source"] for doc in docs]

    current_selection = st.session_state.get("selected_document")
    if current_selection not in sources:
        current_selection = sources[0]
        st.session_state["selected_document"] = current_selection

    selected_source = st.selectbox(
        "Select document",
        options=sources,
        index=sources.index(current_selection),
    )
    st.session_state["selected_document"] = selected_source

    try:
        chunks = store.get_document_chunks(selected_source)
    except Exception as exc:
        st.error(f"Failed to load document chunks: {exc}")
        return

    if not chunks:
        st.warning("No chunks found for this document.")
        return

    doc_meta = next(
        (doc for doc in docs if doc["source"] == selected_source),
        None,
    )

    if doc_meta:
        topic = doc_meta.get("topic", "unknown")
        chunk_count = doc_meta.get("chunk_count", 0)
        st.markdown(
            f"""
            <div class="badge-row">
                <span class="badge badge-topic">Topic: {topic}</span>
                <span class="badge">Chunks: {chunk_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    viewer_container = st.container(height=560)
    with viewer_container:
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata

            badges = f"""
            <div class="badge-row">
                <span class="badge">Chunk {i}</span>
                <span class="badge badge-topic">{meta.topic}</span>
                <span class="badge badge-difficulty">{meta.difficulty}</span>
                <span class="badge badge-type">{meta.type}</span>
            """
            if meta.is_bonus:
                badges += '<span class="badge badge-bonus">bonus</span>'
            badges += "</div>"

            st.markdown(
                f"""
                <div class="mini-card">
                    {badges}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(chunk.chunk_text)

            if meta.related_topics:
                st.caption("Related topics: " + ", ".join(meta.related_topics))

            st.divider()

    st.caption(f"Showing {len(chunks)} chunks from `{selected_source}`.")


def render_message_block(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f'<div class="source-box">📄 <strong>{source}</strong></div>',
                        unsafe_allow_html=True,
                    )

        if message.get("rewritten_query"):
            with st.expander("🔎 Rewritten Query"):
                st.code(message["rewritten_query"])

        if message.get("confidence") is not None:
            try:
                confidence_value = float(message["confidence"])
            except (TypeError, ValueError):
                confidence_value = 0.0
            confidence_value = max(0.0, min(confidence_value, 1.0))
            st.progress(confidence_value)
            st.caption(f"Confidence: {confidence_value:.2f}")

        if message.get("no_context_found"):
            st.warning("⚠️ No relevant content found in corpus.")


def render_chat_interface(graph, store: VectorStoreManager) -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">💬 Interview Prep Chat</div>
            <div class="section-subtitle">
                Ask concept questions, compare topics, or request simplified explanations grounded in the ingested corpus.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)

    stats = store.get_collection_stats()
    topic_options = ["All"] + stats.get("topics", [])
    difficulty_options = ["All", "beginner", "intermediate", "advanced"]

    col_topic, col_diff = st.columns(2)

    current_topic = st.session_state["topic_filter"] or "All"
    current_difficulty = st.session_state["difficulty_filter"] or "All"

    with col_topic:
        selected_topic = st.selectbox(
            "Topic",
            options=topic_options,
            index=topic_options.index(current_topic)
            if current_topic in topic_options
            else 0,
        )

    with col_diff:
        selected_difficulty = st.selectbox(
            "Difficulty",
            options=difficulty_options,
            index=difficulty_options.index(current_difficulty)
            if current_difficulty in difficulty_options
            else 0,
        )

    st.session_state["topic_filter"] = (
        None if selected_topic == "All" else selected_topic
    )
    st.session_state["difficulty_filter"] = (
        None if selected_difficulty == "All" else selected_difficulty
    )

    chat_container = st.container(height=470)
    with chat_container:
        if not st.session_state["chat_history"]:
            st.info(
                "Try asking: “Explain perceptron in simple terms”, “What is backpropagation?”, or “Difference between CNN and RNN”."
            )

        for message in st.session_state["chat_history"]:
            render_message_block(message)

    query = st.chat_input("Ask about a deep learning topic...")

    if not query:
        return

    if not query.strip():
        st.warning("Please enter a non-empty question.")
        return

    st.session_state["chat_history"].append(
        {
            "role": "user",
            "content": query,
        }
    )

    try:
        graph_input = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "topic_filter": st.session_state["topic_filter"],
            "difficulty_filter": st.session_state["difficulty_filter"],
        }
        config = {
            "configurable": {
                "thread_id": st.session_state["thread_id"],
            }
        }

        with st.spinner("🧠 Thinking... generating answer..."):
            result = graph.invoke(graph_input, config=config)

        response = result.get("final_response", {})

        if isinstance(response, dict):
            answer = response.get("answer", "No response generated.")
            sources = response.get("sources", [])
            no_context_found = response.get("no_context_found", False)
            rewritten_query = response.get("rewritten_query", "")
            confidence = response.get("confidence", 0.0)
        else:
            answer = getattr(response, "answer", "No response generated.")
            sources = getattr(response, "sources", [])
            no_context_found = getattr(response, "no_context_found", False)
            rewritten_query = getattr(response, "rewritten_query", "")
            confidence = getattr(response, "confidence", 0.0)

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "no_context_found": no_context_found,
                "rewritten_query": rewritten_query,
                "confidence": confidence,
            }
        )

        st.rerun()

    except Exception as exc:
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": f"An error occurred while generating the response: {exc}",
                "sources": [],
                "no_context_found": True,
                "rewritten_query": "",
                "confidence": 0.0,
            }
        )
        st.rerun()


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    render_hero(settings)
    render_info_ribbon()

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    if not st.session_state["ingested_documents"]:
        refresh_documents(store)

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    viewer_col, chat_col = st.columns([1.05, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph, store)

    st.markdown(
        """
        <div class="footer-note">
            Built for deep learning interview preparation • grounded answers • transparent sources
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()