# System Architecture
## Team: RAG Architects
## Date: 3/24/2026
## Members and Roles:
- Corpus Architect: Lakshmi Nikitha Ankam
- pipeline engineer:Manoj Kumar Atti
- UX Lead:Sai Sukruth Chatla
- Prompt Engineer: Monica Valli Kandulapati
- QA Lead: Vihal Thatipamula

---

## Architecture Diagram

Replace this section with your team's completed flow chart.
Export from FigJam, Miro, or draw.io and embed as an image,
or describe the architecture as an ASCII diagram.

The diagram must show:
- [x] How a corpus file becomes a chunk
- [x] How a chunk becomes an embedding
- [x] How duplicate detection fires
- [x] How a user query flows through LangGraph to a response
- [x] Where the hallucination guard sits in the graph
- [x] How conversation memory is maintained across turns

```text
==================================================================================
                     DEEP LEARNING RAG AGENT ARCHITECTURE                   
==================================================================================

                             [ CORPUS INGESTION ]                        
                                     |
    +--------------------------------|-----------------------------------+
    |                   +-------------------------+                      |
    |                   | Corpus File (.md, .pdf) |                      |
    |                   +-------------------------+                      |
    |                                |                                   |
    |                     [ Read & Parse Content ]                       |
    |                                |                                   |
    |              +-------------------------------------+               |
    |              | Hash Generator (source + chunk_text)|               |
    |              +-------------------------------------+               |
    |                                |                                   |
    |                     < Duplicate Detection >                        |
    |                     /                     \                        |
    |              [Hash Exists]             [New Hash]                  |
    |                   |                        |                       |
    |          (Skip and Ignore)                 v                       |
    |                            +---------------------------------+     |
    |                            |   Recursive Character Splitter  |     |
    |                            | (max: 512 chars, overlap: 50)   |     |
    |                            +---------------------------------+     |
    |                                            |                       |
    |                                            v                       |
    |                           +-----------------------------------+    |
    |                           |     all-MiniLM-L6-v2 Embedder     |    |
    |                           +-----------------------------------+    |
    |                                            |                       |
    +--------------------------------------------|-----------------------+
                                                 |               
                                                 v               
                                    [( ChromaDB Vector Store )]  
                                                 |               
    +--------------------------------------------|-----------------------+
    |                        [ LANGGRAPH QUERY FLOW ]                    |
    |                                                                    |
    |  (User Query) ---> [ Streamlit UI ] ---> [ Query Rewrite Node ]    |
    |                                                  |                 |
    |                                                  v                 |
    |  [( ChromaDB )] <-----( retrieve K=4 )----- [ Retrieval Node ]     |
    |                                                  |                 |
    |                                                  v                 |
    |                        < Hallucination Guard >                     |
    |         ( Similarity < 0.3 )              ( Similarity >= 0.3 )    |
    |                  |                               |                 |
    |                  v                               v                 |
    |          [ End Node ]                     [ Generation Node ]      |
    |     "No context response"     <======[ Injection of Chat History]  |
    |                                                  |                 |
    |                                                  v                 |
    |                                         [ Groq LLM API ]           |
    |                                         ( Llama 3.1-8b )           |
    |                                                  |                 |
    |                                                  v                 |
    |  [( MemorySaver Checkpointer )] <----- [ Final Grounded Answer ]   |
    |        (saved by thread_id)                                        |
    +--------------------------------------------------------------------+
```

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
  *The system prompt defines the agent as a senior machine learning engineer conducting a technical interview preparation session. It enforces strict grounding in provided context, prohibits use of external knowledge, and requires all answers to include source citations in a fixed format. It also adapts response depth based on difficulty metadata and ensures partial answers are fairly evaluated. Additional constraints were added to explicitly prevent hallucination by disallowing any unsupported information from being included in responses.*

- **Question generation prompt:**
  *The question generation prompt takes a retrieved context chunk and a difficulty level as input. It generates a single interview-style question that requires conceptual understanding and reasoning rather than simple recall. The output is a structured JSON object containing the question, difficulty level, topic, a model answer grounded strictly in the context, a follow-up question, and source citations. Additional constraints were added to prevent trivial or definition-based questions and enforce deeper conceptual reasoning.*

- **Answer evaluation prompt:**
  *The answer evaluation prompt takes a question, candidate answer, and source context as input. It evaluates the answer strictly against the provided material and assigns a score from 0 to 10 based on completeness, correctness, and depth. The rubric penalizes vague or incomplete answers and rewards precise, well-articulated responses. The output includes structured JSON fields such as score, correct aspects, missing concepts, an ideal answer, interview verdict, and a coaching tip. Additional strict scoring instructions were added to prevent inflated scores for weak answers. *

- **JSON reliability:**
  *To ensure consistent JSON output, all prompts explicitly instruct the model to respond with only the JSON object and no additional text, explanations, or markdown code fences. This prevents parsing errors and ensures compatibility with downstream programmatic processing in the pipeline. These constraints were added after identifying that models may otherwise include extra formatting such as ```json blocks or natural language explanations.*

- **Failure modes identified:**
  *- Question Generation Prompt: Sometimes produced trivial or definition-based questions. This was addressed by adding constraints requiring reasoning-based and non-trivial questions.
- Answer Evaluation Prompt: Tended to give overly generous scores for weak answers. This was fixed by adding strict scoring instructions to penalize missing key concepts.
- System Prompt: Risk of hallucination when context was incomplete. This was mitigated by explicitly prohibiting use of unsupported or external knowledge.*
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
   Content-hash based Duplication Checking for Ingestion
   **Rationale:**
   Rather than merely relying on filenames (which users may change), generating a SHA-256 chunk hash allows robust skipping of strictly identical text blocks. It saves database size and stops redundant retrieval loops.
   **Interview answer:**
   We implemented content-addressed deduplication by hashing the source name and chunk content; this prevents redundant vector indexing even if the source file is renamed prior to upload.

2. **Decision:** Recursive Markdown Header Text Splitting
   **Rationale:**
   Standard text chunking slices sentences arbitrarily. By first splitting on markdown headers before sizing to 512 chunks, semantic boundaries are preserved. Concepts in a single section stay together, yielding higher retrieval quality.
   **Interview answer:**
   To preserve semantic coherence, we configured a header-aware chunker so that specific concepts bounded by headings stay together during retrieval indexing."

4. **Decision:** Strict Similarity Threshold / Guard Nodes 
   **Rationale:**
    Because RAG systems easily hallucinate answers on tangentially related vectors, setting a threshold of 0.3 enforces a hard block on the graph advancing to generation if confidence is low.
   **Interview answer:**
   "We designed our LangGraph pipeline to include an explicit edge condition that traps retrieval failures below a 0.3 threshold, completely bypassing generation to eliminate hallucination."

6. **Decision:** *(optional — bonus points in Hour 3)*
   **Rationale:**
   **Interview answer:**

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | Retrieved relevant segments and gave strict citations | Pass |
| Off-topic query | No context found message | Correctly bypassed LLM to output "I do not have context" | Pass |
| Duplicate ingestion | Second upload skipped | Hash match detected; skipping ingestion process | Pass |
| Empty query | Graceful error, no crash | Streamlit input validation blocked empty submit | Pass |
| Cross-topic query | Multi-topic retrieval | Retrieved multiple sources with lower but passing thresholds | Pass |

**Critical failures fixed before Hour 3:**
- Fixed LangGraph state typing errors causing runtime crashes during query processing.
- Resolved Streamlit session state issues where conversation memory was wiped on re-render.

**Known issues not fixed (and why):**
- UI freezes during large PDF vector generation because backend chunking is not fully asynchronous.
- LLM occasionally wraps generated structured responses in JSON markdown blocks despite strict prompt constraints.

---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- PDF chunking occasionally captures reference or acknowledgments sections natively, producing noisy contexts.
- The hallucination guard's similarity threshold (0.3) was calibrated heuristically based on a small batch of manual queries rather than via a rigorous automated evaluation.

---

## What We Would Do With More Time

- Implement a hybrid search mechanism that combines vector search (for semantic meaning) with BM25 keyword search (for exact term matching), significantly improving retrieval for specific algorithms or acronyms.
- Add a cross-encoder re-ranking step to evaluate the relevancy of the retrieved `k` chunks much more precisely before feeding them to the generation node.
- Move document ingestion to an asynchronous Celery task queue or background thread so that uploading large Deep Learning PDFs does not freeze the Streamlit interface.
- Implement an automated LLM-as-a-judge quantitative evaluation pipeline (e.g., using RAGAS) to continually measure context precision and answer faithfulness.

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:**
How does your system address the risk of hallucinations when given queries that sound plausible but aren't supported by the retrieved context?

Model answer:
We constructed a deterministic hallucination guard in our LangGraph setup that triggers after the retrieval node. It evaluates the similarity score of retrieved chunks, and if no chunk meets our threshold (e.g., 0.3), it routes the execution directly to the end node to output a hardcoded "no context found" message, completely skipping the generation step.

**Question 2:**
What specific chunking strategy and sizes did you implement, and how did you mathematically or empirically validate that it was correct for dense technical material like Deep Learning papers?

Model answer:
We utilized a recursive character splitter combined with markdown header splitting. We settled on a 512-character max chunk size with a 50-character overlap. We validated this by observing that academic definitions and equations generally fit within this window, and the overlap ensures that context spanning sentence boundaries isn't lost during similarity search.

**Question 3:**
How does your system manage identical document uploads, particularly if the file has been renamed by the user before ingestion?

Model answer:
We deployed a deterministic hashing method generating a 16-character hex ID derived from the SHA-256 hash of the `source` name plus `chunk_text`. Rather than matching simple filenames, ChromaDB checks for these content-derived IDs, enabling intelligent duplicate detection and skipping re-processing automatically.

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
-Constructing clear conditional edges in LangGraph.
- Utilising abstract LLM factories directly loading from environments.

**What confused us:**
 -Calibrating the cosine similarity threshold effectively against embedding quirks

**One thing each team member would study before a real interview:**
- Corpus Architect:  I would study advanced parsing strategies for complex document formats (like nested PDFs with tables and figures), focusing on preserving structural hierarchy and semantic coherence to ensure the generation of high-fidelity chunks for the retrieval model.
- Pipeline Engineer:I would dive deeper into advanced retrieval architectures such as Hybrid Search (combining keyword-based BM25 with dense vector embeddings), cross-encoder re-ranking for refined context selection, and asynchronous task queues for handling large-scale data ingestion without blocking the application.
- UX Lead: I would focus on mastering advanced frontend state management techniques, particularly optimizing real-time text streaming and utilizing native Streamlit caching mechanisms to eliminate unnecessary component reruns and reduce user-facing latency.
- Prompt Engineer: I would study advanced prompt engineering techniques such as prompt chaining, structured output enforcement, and handling LLM failure modes (like hallucination and inconsistent JSON generation) to build more robust and reliable AI systems.
- QA Lead: Dive deeper into automated evaluation frameworks for RAG pipelines (like Ragas) to systematically measure answer faithfulness instead of relying on manual test queries.
