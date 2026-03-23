# Build Checklist

## What You Are Building

A RAG-powered interview preparation agent that:
- Ingests deep learning study material your team authors
- Stores and retrieves content from a ChromaDB vector store
- Uses LangChain for retrieval and prompt orchestration
- Uses LangGraph to manage agent state and multi-turn conversation
- Provides a UI with document ingestion, document viewing, and chat

**Your deliverable is a working, demonstrable system.**
Not a prototype. Not a wireframe. A running application.

---

**The critical path:** Corpus Architect's schema must be agreed with
Pipeline Engineer before either starts building. Everything else
flows from that agreement.

---

## Phase 1 Checklist 

---

### Corpus Architect

Your output is the foundation. Without quality content, the
retrieval pipeline has nothing to return.

**Before writing anything:**
- [ ] Agree metadata schema with Pipeline Engineer
      *(topic, difficulty, type, source, related_topics, is_bonus)*
- [ ] Confirm file naming convention with the team
      *(recommended: `ann_intermediate.md`, `lstm_advanced.md`)*
- [ ] Place all files in `data/corpus/`

**Content drafting:**
- [ ] Draft topic 1 — ANN
      *Hint: cover forward propagation, backprop, activation functions,
      loss functions, and vanishing gradients as separate chunks*
- [ ] Draft topic 2 — CNN
      *Hint: cover convolution operation, pooling, feature maps,
      and the LeNet/AlexNet architectures as separate chunks*
- [ ] Draft topic 3 — RNN
      *Hint: cover hidden state, sequence processing, BPTT,
      and why vanishing gradients are worse here than in ANNs*
- [ ] Locate at least one landmark paper PDF per topic drafted
      *(see landmark papers table in README)*

**Chunk quality — check every chunk before standup:**
- [ ] One atomic idea per chunk — if it could answer five questions, split it
- [ ] Between 100 and 300 words per chunk
- [ ] Remove the topic name — can you still identify the topic from the content?
      If no, the chunk is too generic
- [ ] Metadata complete and accurate on every chunk
- [ ] No topic bleeding — LSTM content does not appear in an RNN chunk
- [ ] Bonus topics flagged with `"is_bonus": true`

**Phase 1 milestone:** 3 topics drafted, minimum 3 chunks each,
schema agreed, at least one landmark paper PDF located per topic.

---

### Pipeline Engineer

You are building the backbone. Everything the UX Lead and Prompt
Engineer produce must connect through your code.

**Environment — verify before anything else:**
- [ ] `uv sync` completes without errors
- [ ] `.env` configured with working LLM provider
- [ ] `uv run python -c "import chromadb; import langchain; import langgraph; print('OK')`
      passes cleanly
- [ ] `data/chroma_db/` directory exists

**Implement in this order — do not skip ahead:**
- [ ] `EmbeddingFactory._create_local()` in `config.py`
      *Hint: use `HuggingFaceEmbeddings(model_name=self._settings.embedding_model)`
      from `langchain_community.embeddings`*
- [ ] `LLMFactory._create_groq()` / `_create_ollama()` / `_create_lmstudio()`
      in `config.py` — implement whichever provider your team is using
      *Hint: `ChatGroq(api_key=..., model_name=...)` for Groq,
      `ChatOllama(base_url=..., model=...)` for Ollama*
- [ ] `VectorStoreManager._initialise()` in `store.py`
      *Hint: `chromadb.PersistentClient(path=...)` then
      `client.get_or_create_collection(name=..., metadata={"hnsw:space": "cosine"})`*
- [ ] `VectorStoreManager.check_duplicate()` in `store.py`
      *Hint: `self._collection.get(ids=[chunk_id])` — returns a dict,
      check if `result["ids"]` is non-empty*
- [ ] `VectorStoreManager.ingest()` in `store.py`
      *Hint: loop chunks, call check_duplicate, embed with
      `self._embeddings.embed_documents([chunk.chunk_text])`,
      then `self._collection.upsert(ids, embeddings, documents, metadatas)`*

**Hello world test — before standup:**
- [ ] Write a scratch script (not production code) that ingests
      `examples/sample_chunk.json` and queries "what is a neural network"
- [ ] Confirm a chunk is returned with a similarity score
- [ ] Confirm running the script twice skips the chunk on the second run

**Phase 1 milestone:** environment verified, embedding and LLM factories
implemented, ChromaDB initialising, hello world retrieval returning results.

---

### UX Lead

You own the interface and the demo narrative. A polished UI
that crashes is worse than a plain UI that works.

**Decide immediately — do not revisit:**
- [ ] Framework chosen: Streamlit or Gradio
      *(Streamlit → Streamlit Community Cloud,
      Gradio → HuggingFace Spaces)*
- [ ] Deployment platform chosen and account created

**Read before building:**
- [ ] Open `src/rag_agent/agent/state.py` and read all data models
      *These are what the backend returns — know them before you build*
- [ ] Open `src/rag_agent/vectorstore/store.py` and read all method signatures
      *These are what you will call — note the return types*
- [ ] Confirm with Pipeline Engineer that signatures have not changed

**Build the static layout — no backend calls yet:**
- [ ] Panel 1 — Ingestion
      *Multi-file uploader (.pdf and .md), upload button,
      status display area, ingested documents list*
- [ ] Panel 2 — Document viewer
      *Document selector dropdown, content display area,
      chunk count and metadata display*
- [ ] Panel 3 — Chat
      *Scrollable chat history, query input, submit button,
      source citation display area, no-context indicator*
- [ ] All `st.session_state` keys initialised in `initialise_session_state()`
      *Hint: chat_history, ingested_documents, selected_document,
      thread_id, topic_filter, difficulty_filter*
- [ ] App runs locally without errors:
      `uv run streamlit run src/rag_agent/ui/app.py`

**Phase 1 milestone:** framework chosen, static three-panel layout
running locally, session state initialised, no crashes on load.

---

### Prompt Engineer

Your prompts determine the quality of every response the system
produces. Test before you integrate — never the other way around.

**Test all prompts manually before touching code:**
Open Claude, ChatGPT, or your chosen LLM in a browser.
Use `examples/sample_chunk.json` as your test context.

- [ ] **System prompt** — paste `SYSTEM_PROMPT` from `prompts.py` as the
      system message, paste the sample chunk as context, ask a question
      about LSTMs
      *Does it cite the source? Does it stay within the chunk?
      Does it refuse to answer when context is removed?*

- [ ] **Question generation prompt** — fill in `{context}` with the
      sample chunk and `{difficulty}` with "intermediate"
      *Does it return valid JSON? Is the question open-ended?
      Does it require connecting at least two concepts?
      If JSON is malformed, add: "Respond with JSON only.
      No preamble, explanation, or markdown code fences."*

- [ ] **Answer evaluation prompt** — fill in with a question from
      the previous test and a deliberately incomplete answer
      *Does it correctly identify what is missing?
      Is the score calibrated — not too generous, not too harsh?*

- [ ] **Query rewrite prompt** — test with a vague natural language query
      *Input: "I'm confused about how LSTMs remember things"
      Expected output: "LSTM long-term memory cell state forget gate mechanism"*

**Document failure modes — one per prompt minimum:**
- [ ] System prompt: *(e.g. model draws on general knowledge — tighten constraints)*
- [ ] Question generation: *(e.g. produces yes/no questions — add open-ended instruction)*
- [ ] Answer evaluation: *(e.g. score too generous — adjust scoring rubric)*
- [ ] Query rewrite: *(e.g. over-abbreviates — test with longer queries)*

**Phase 1 milestone:** all four prompts manually tested and validated,
failure modes documented, JSON reliability confirmed.

---

### QA Lead

Your job is to find failure cases before the judges do in Hour 3.
Start preparing now so Phase 2 testing is not rushed.

**Write your integration test plan:**
- [ ] Test 1 — Normal query
      *Input: "Explain the vanishing gradient problem"
      Expected: relevant chunks retrieved, accurate answer, source cited*
- [ ] Test 2 — Off-topic query
      *Input: "What is the capital of France"
      Expected: hallucination guard fires, clear no-context message,
      no fabricated deep learning answer*
- [ ] Test 3 — Duplicate ingestion
      *Input: upload the same file twice
      Expected: second upload detected and skipped,
      IngestionResult.skipped equals chunk count of the file*
- [ ] Test 4 — Empty query
      *Input: submit blank input
      Expected: graceful error message, application does not crash*
- [ ] Test 5 — Cross-topic query
      *Input: "How do LSTMs improve on RNNs for Seq2Seq tasks"
      Expected: chunks from at least two topics retrieved and synthesised*

**Prepare Hour 3 interview questions — 3 required:**
- [ ] Question 1: single topic, intermediate difficulty
      *(e.g. "Walk me through the three gates in an LSTM and what each controls")*
- [ ] Question 2: connects two topics
      *(e.g. "How does the encoder in a Seq2Seq model relate to an autoencoder?")*
- [ ] Question 3: system design or tradeoff
      *(e.g. "Why did your team choose your chunk size and what would break
      if you doubled it?")*
- [ ] Model answer written for each question

**Risk assessment — review rubric before standup:**
- [ ] Identify your team's top two risk categories from `docs/rubric.md`
- [ ] Write one action per risk to reduce it before Hour 3
- [ ] Share risk assessment with the team at standup

**Phase 1 milestone:** five test cases written with expected behaviours,
three Hour 3 questions drafted with model answers, risk assessment complete.

---

## Sanity Check 

**Online Class:** Find an agreed time for Phase 1 to end and meet to check your work.

**Hard stop. Every member checks in.**

Three questions only — keep it tight:
1. What do I have right now?
2. What do I need from someone else?
3. What is blocking me?

Post standup notes in your team channel immediately after.

**Common blockers and resolutions:**

| Blocker | Resolution |
|---|---|
| Metadata schema not agreed | Resolve with Pipeline Engineer before Phase 2 starts |
| ChromaDB not initialising | Check `CHROMA_DB_PATH` exists, confirm `PersistentClient` is used |
| UX Lead unsure what backend returns | Re-read `state.py` data models now |
| Prompts not tested | First 10 min of Phase 2 only — test then integrate |
| Embeddings failing | Run `uv run python -c "from sentence_transformers import SentenceTransformer; print('OK')"` |

---

## Phase 2 Checklist — Integration and Hardening 

Roles converge in this order. Do not jump to your Phase 2 tasks
until the dependency above you is unblocked.

---

### Integration Order — Follow This Sequence

**Step 1 — Pipeline Engineer + Corpus Architect**
- [ ] Run first real ingestion with Phase 1 corpus content
- [ ] Verify chunks stored with correct metadata in ChromaDB
- [ ] Verify duplicate detection fires on a second ingest run
- [ ] Verify query returns ranked results with scores above threshold

**Step 2 — Pipeline Engineer + Prompt Engineer**
- [ ] Implement `query_rewrite_node` in `nodes.py`
      *Hint: extract latest `HumanMessage` from state, call LLM with
      `QUERY_REWRITE_PROMPT`, return `{"rewritten_query": result}`*
- [ ] Implement `retrieval_node` in `nodes.py`
      *Hint: call `VectorStoreManager.query(state.rewritten_query)`,
      if empty set `{"no_context_found": True, "retrieved_chunks": []}`*
- [ ] Implement `generation_node` in `nodes.py`
      *Hint: check `state.no_context_found` first — if True return
      `NO_CONTEXT_RESPONSE` immediately. Otherwise build context string
      from retrieved chunks with citations, call LLM, return response*
- [ ] Assemble graph in `graph.py`
      *Hint: `StateGraph(AgentState)` → add nodes → add edges →
      add conditional edge from retrieval using `should_retry_retrieval` →
      `graph.compile(checkpointer=MemorySaver())`*

**Step 3 — Pipeline Engineer + UX Lead**
- [ ] Wire ingestion panel to `VectorStoreManager.ingest()`
- [ ] Wire document viewer to `VectorStoreManager.list_documents()`
      and `get_document_chunks()`
- [ ] Wire chat to compiled LangGraph graph
      *Hint: `graph.invoke({"messages": [HumanMessage(content=query)]},
      config={"configurable": {"thread_id": st.session_state.thread_id}})`*
- [ ] Verify source citations appear in every chat response
- [ ] Verify no-context indicator appears when hallucination guard fires

**Step 4 — QA Lead**
- [ ] Run all five test cases from Phase 1 test plan
- [ ] Record pass/fail and actual behaviour for each
- [ ] Flag critical failures immediately to the relevant role owner

---

### Phase 2 Per-Role Checklist

#### Corpus Architect
- [ ] Complete remaining core topics: LSTM, Seq2Seq, Autoencoder
      *(minimum 3 chunks each)*
- [ ] Ingest at least two landmark paper PDFs with Pipeline Engineer
- [ ] Review PDF chunks for noise — remove reference section chunks,
      equation-only chunks, and header/footer artifacts
- [ ] Final corpus quality check against Phase 1 checklist
- [ ] Add bonus topics if core topics complete: SOM, Boltzmann, GAN

#### Pipeline Engineer
- [ ] Implement `VectorStoreManager.query()` in `store.py`
      *Hint: embed query with `self._embeddings.embed_query(query_text)`,
      call `self._collection.query(query_embeddings, n_results=k,
      include=["documents", "metadatas", "distances"])`,
      convert distances to scores: `score = 1 - distance` for cosine,
      filter below `self._settings.similarity_threshold`*
- [ ] Implement `VectorStoreManager.list_documents()` and
      `get_document_chunks()` for the document viewer
- [ ] Implement conversation memory trimming in `generation_node`
      *Hint: use `trim_messages(messages, max_tokens=settings.max_context_tokens,
      strategy="last")` from `langchain_core.messages`*
- [ ] Confirm graph advances through all three nodes end to end

#### UX Lead
- [ ] Progress indicator during ingestion
      *(Streamlit: `st.spinner()` or `st.progress()`)*
- [ ] Ingestion result display: chunks added, duplicates skipped, errors
- [ ] Source citations visible in every chat response
- [ ] Clear no-context indicator when hallucination guard fires
- [ ] Stretch goal: streaming responses
      *(Hint: replace `graph.invoke` with `graph.stream` and
      use `st.write_stream()` to display tokens as they arrive)*

#### Prompt Engineer
- [ ] Integrate all prompts into the live system via Pipeline Engineer
- [ ] Run 10 manual test queries through the integrated system
- [ ] Verify question difficulty levels are being applied correctly
- [ ] Verify JSON parsing is reliable — no malformed responses in 10 tests
- [ ] Document final prompt versions in `prompts.py` docstrings

#### QA Lead
- [ ] Run all five integration test cases — record results in `docs/architecture.md`
- [ ] Confirm critical failures fixed: hallucination guard, duplicate
      detection, source citations, no crashes
- [ ] Write 60-second demo script hitting these beats in order:
      1. Upload two documents
      2. Upload one again — show duplicate detection
      3. Submit a normal query — show source citation
      4. Submit an off-topic query — show hallucination guard
      5. Generate an interview question with model answer
- [ ] Practice demo script once before rehearsal

---

## Demo Rehearsal

- [ ] One full end-to-end run-through with the actual demo script
- [ ] Every team member watches and notes anything wrong or confusing
- [ ] Decide now: fix broken things or work around them gracefully
      *(a clean workaround explained openly scores better than a
      hidden bug that surfaces in front of the judges)*
- [ ] Record a short video walkthrough as a backup
      *(Loom, OBS, or your phone — insurance if something breaks live)*

---

## Presentations

**F2F:** Class presentation

**Online Class:** 5 minute video presentation of your working demo

---

## Common Pitfalls

**ChromaDB not persisting between runs**
Confirm `PersistentClient` is used, not `EphemeralClient`.
Confirm `data/chroma_db/` directory exists before initialisation.

**Embeddings slow on first run**
Expected — sentence-transformers loads the model into memory on first call.
Wrap with `@st.cache_resource` so it loads once per session, not per click.

**LangGraph not advancing past first node**
Every node function must return a dict of state updates, not `None`.
Add a `print(state)` at the start of each node to confirm state is passing.

**Duplicate detection not firing**
Chunk IDs must be generated from content hash, not filename or timestamp.
Two uploads of the same file must produce identical IDs. Test with:
`VectorStoreManager.generate_chunk_id("f.md", "text") ==
VectorStoreManager.generate_chunk_id("f.md", "text")`

**Hallucination guard triggering on everything**
Similarity threshold is too high. Start at 0.3, print actual scores
during testing to calibrate. Scores are `1 - cosine_distance`.

**Streamlit losing state on every click**
Every persistent object must be in `st.session_state` or wrapped in
`@st.cache_resource`. VectorStoreManager, graph, and chat history
all need this treatment.

**JSON parsing failing on prompt output**
Add to the end of any JSON prompt: "Respond with the JSON object only.
No preamble, explanation, or markdown code fences."
Wrap all `json.loads()` calls in try/except and log the raw string on failure.
