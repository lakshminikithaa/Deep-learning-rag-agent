# System Architecture
## Team: ___________________
## Date: 3/24/2026
## Members and Roles:
- Corpus Architect: Lakshmi Nikitha Ankam
- pipeline engineer:Manoj Kumar Atti
- UX Lead:Sai Sukruth Chatla
- Prompt Engineer: Monica Valli Kandulapati
- QA Lead: Vihal Thatipamula

---

## Architecture Diagram
```mermaid
flowchart TD
    subgraph Ingestion["Ingestion Pipeline"]
        A[Corpus Files .md/.pdf] -->|"DocumentChunker (512 char, 50 overlap)"| B[Raw Chunks]
        B --> C{"Check Duplicate (SHA256 Hash)"}
        C -->|"Hash found"| D[Skip Chunk]
        C -->|"New Hash"| E["EmbeddingFactory (all-MiniLM-L6-v2)"]
        E --> F[("ChromaDB vector store")]
    end

    subgraph Chat["Retrieval & Generation Pipeline"]
        G["User Query (Streamlit UI)"] -->|"session_state.thread_id maintains memory"| H("query_rewrite_node")
        H -->|"Dense technical query"| I("retrieval_node")
        F -.->|"Cosine Similarity (k=4)"| I
        I --> J{"Similarity > 0.3 threshold"}
        J -->|"No (fails Hallucination Guard)"| K["Return 'No Context Found' warning"]
        J -->|"Yes"| L("generation_node")
        L -->|"Groq LLM + System constraints"| M["Final Answer w/ Citations"]
    end
    
    style D fill:#f9cfcf,stroke:#ff9999
    style K fill:#f9cfcf,stroke:#ff9999
    style M fill:#c2e8c6,stroke:#82c988
    style F fill:#e5ccff,stroke:#d18ce0
Replace this section with your team's completed flow chart.
Export from FigJam, Miro, or draw.io and embed as an image,
or describe the architecture as an ASCII diagram.

The diagram must show:
- [ ] How a corpus file becomes a chunk
- [ ] How a chunk becomes an embedding
- [ ] How duplicate detection fires
- [ ] How a user query flows through LangGraph to a response
- [ ] Where the hallucination guard sits in the graph
- [ ] How conversation memory is maintained across turns

*(replace this line with your diagram image or ASCII art)*

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:**
  Both .md (Markdown) and .pdf (Academic PDFs).

- **Landmark papers ingested:**
-  HochreiterS1997.pdf (LSTM Original Paper)
  - NIPS-2012-imagenet-classification.pdf (AlexNet Original Paper)
  - Markdown structured study guides (ann_intermediate.md, cnn_intermediate.md, rnn_intermediate.md, lstm_intermediate.md, seq2seq_intermediate.md, autoencoder_intermediate.md)
- **Chunking strategy:**
Recursive character splitting and markdown header splitting with max chunk size of 512 characters and 50 characters overlap. This balances context richness with retrieval precision and prevents concepts that span chunk boundaries from being lost.

- **Metadata schema:**
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | The deep learning topic covered (e.g. LSTM, CNN). |
  | difficulty | string | Complexity level (beginner, intermediate, advanced). |
  | type | string | Content type (e.g. concept_explanation). |
  | source | string | Source document filename (e.g. lstm_intermediate.md). |
  | related_topics | list | Other related DL topics for contextual expansion. |
  | is_bonus | bool | Flags whether the topic is a bonus/advanced topic (e.g. SOM, GAN). |
- **Duplicate detection approach:**
  A deterministic 16-character hex string ID is generated derived from the SHA-256 hash of `source::chunk_text`. Content hashing is used rather than simple filename matching because it detects identical content even if files are renamed, and re-processes intelligently.

- **Corpus coverage:**
  - [x] ANN
  - [x] CNN
  - [x] RNN
  - [x] LSTM
  - [x] Seq2Seq
  - [x] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB — PersistentClient
- **Local persistence path:** `./data/chroma_db`

- **Embedding model:** - all-MiniLM-L6-v2 via HuggingFaceEmbeddings/sentence-transformers
  

- **Why this embedding model:** Runs entirely on CPU locally, mitigating API costs, reducing latency over internet connections, and ensuring proprietary data never leaves the machine.

- **Similarity metric:** `cosine` similarity space.
  

- **Retrieval k:** `4` chunks per query to build a rich context up to roughly 2000 tokens while fitting safely inside context windows.


- **Similarity threshold:** `0.3` minimum score to pass the hallucination guard. Empirically tuned to ensure only reasonably matching results are returned to the final generation node.

- **Metadata filtering:**  The `query` method supports explicit mappings for `topic_filter` and `difficulty_filter`, combined using `$and` for granular ChromaDB queries via the UI.
 *

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
 | Node | Responsibility |
  |---|---|
  | query_rewrite_node | Transforms a conversational query into a dense, keyword-rich search term. |
  | retrieval_node | Embeds the rewritten query and searches ChromaDB for the top-k relevant chunks. |
  | generation_node | Feeds context chunks to the LLM to generate a strictly grounded, cited answer. |

- **Conditional edges:** After `retrieval`, the `should_retry_retrieval` condition checks if any chunk passed the similarity threshold. If none did, the graph takes the `"end"` edge (triggering the hallucination guard message). If chunks exist, it takes the `"generate"` edge to `generation_node`.
 
- **Hallucination guard:**
   When similarity thresholds fail, the system intercepts and immediately returns the `NO_CONTEXT_RESPONSE` text explaining that no relevant info was found.

- **Query rewriting:**
  - Raw query: "Can you tell me how LSTMs solve the vanishing gradient issue?"
  - Rewritten query: "LSTM long short-term memory vanishing gradient problem solution gate"

- **Conversation memory:** Maintained via the `MemorySaver` checkpointer in LangGraph, keyed by `thread_id` from the Streamlit session state. This captures multi-turn history until the application restarts.
 

- **LLM provider:** `Groq` using `llama-3.1-8b-instant`.
 

- **Why this provider:**  Groq's LPU provides virtually near-zero-latency generation which is critical for real-time interview preparations and streaming responses.
  

---

### Prompt Layer

- **System prompt summary:**
  *(describe the agent persona and the key constraints in your system prompt)*

- **Question generation prompt:**
  *(what inputs does it take and what does it return?)*

- **Answer evaluation prompt:**
  *(how does it score a candidate answer? what is the scoring rubric?)*

- **JSON reliability:**
  *(what did you add to your prompts to ensure consistent JSON output?)*

- **Failure modes identified:**
  *(list at least one failure mode per prompt and how you addressed it)*
  -
  -
  -

---

### Interface Layer

- **Framework:** *(Streamlit / Gradio)*
- **Deployment platform:** *(Streamlit Community Cloud / HuggingFace Spaces)*
- **Public URL:** *(paste your deployed app URL here once live)*

- **Ingestion panel features:**
  *(describe what the user sees — file uploader, status display, document list)*

- **Document viewer features:**
  *(describe how users browse ingested documents and chunks)*

- **Chat panel features:**
  *(describe how citations appear, how the hallucination guard is surfaced,
  and any filters available)*

- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | chat_history | |
  | ingested_documents | |
  | selected_document | |
  | thread_id | |

- **Stretch features implemented:**
  *(streaming responses, async ingestion, hybrid search, re-ranking, other)*

---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:**
   *(e.g. chunk size of 512 with 50 character overlap)*
   **Rationale:**
   *(why this over alternatives? what would break if you changed it?)*
   **Interview answer:**
   *(write a two sentence answer you could give in a technical screen)*

2. **Decision:**
   **Rationale:**
   **Interview answer:**

3. **Decision:**
   **Rationale:**
   **Interview answer:**

4. **Decision:** *(optional — bonus points in Hour 3)*
   **Rationale:**
   **Interview answer:**

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | | |
| Off-topic query | No context found message | | |
| Duplicate ingestion | Second upload skipped | | |
| Empty query | Graceful error, no crash | | |
| Cross-topic query | Multi-topic retrieval | | |

**Critical failures fixed before Hour 3:**
-
-

**Known issues not fixed (and why):**
-
-

---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- *(e.g. PDF chunking produces noisy chunks from reference sections)*
- *(e.g. similarity threshold was calibrated manually, not empirically)*
- *(e.g. conversation memory is lost when the app restarts)*

---

## What We Would Do With More Time

- *(e.g. implement hybrid search combining vector and BM25 keyword search)*
- *(e.g. add a re-ranking step using a cross-encoder)*
- *(e.g. async ingestion so large PDFs don't block the UI)*

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:**

Model answer:

**Question 2:**

Model answer:

**Question 3:**

Model answer:

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
-

**What confused us:**
-

**One thing each team member would study before a real interview:**
- Corpus Architect:
- Pipeline Engineer:
- UX Lead:
- Prompt Engineer:
- QA Lead:
