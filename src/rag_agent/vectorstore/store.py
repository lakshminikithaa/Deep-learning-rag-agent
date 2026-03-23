"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        import chromadb
        Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._settings.chroma_db_path)
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialised vector store '{self._settings.chroma_collection_name}' with {self._collection.count()} items.")

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        
        # Process in batches to avoid overwhelming the memory/embedder
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            valid_chunks = []
            
            for chunk in batch:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                else:
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                continue
                
            try:
                embeddings = self._embeddings.embed_documents([c.chunk_text for c in valid_chunks])
                self._collection.upsert(
                    ids=[c.chunk_id for c in valid_chunks],
                    embeddings=embeddings,
                    documents=[c.chunk_text for c in valid_chunks],
                    metadatas=[c.metadata.to_dict() for c in valid_chunks]
                )
                result.ingested += len(valid_chunks)
                for chunk in valid_chunks:
                    if chunk.metadata.source not in result.document_ids:
                        result.document_ids.append(chunk.metadata.source)
            except Exception as e:
                logger.error(f"Error during ingestion: {str(e)}")
                result.errors.append(str(e))
                
        logger.info(f"Ingestion complete: {result.ingested} ingested, {result.skipped} skipped.")
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k or self._settings.retrieval_k
        where_filter = {}
        if topic_filter:
            where_filter["topic"] = topic_filter
        if difficulty_filter:
            where_filter["difficulty"] = difficulty_filter

        query_embedding = self._embeddings.embed_query(query_text)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_chunks = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved_chunks
            
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0], 
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        ):
            score = 1.0 - dist  # Assuming cosine dist
            if score >= self._settings.similarity_threshold:
                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=doc,
                    metadata=ChunkMetadata.from_dict(meta),
                    score=score
                )
                retrieved_chunks.append(chunk)
                
        retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
        return retrieved_chunks

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        results = self._collection.get(include=["metadatas"])
        
        source_counts = {}
        source_topics = {}
        
        if results.get("metadatas"):
            for meta in results["metadatas"]:
                if meta:
                    source = meta.get("source", "unknown")
                    topic = meta.get("topic", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                    source_topics[source] = topic
                    
        doc_list = [
            {"source": source, "topic": topic, "chunk_count": count}
            for source, count in source_counts.items()
            for src_topic, topic in source_topics.items() if src_topic == source
        ]
        
        return sorted(doc_list, key=lambda x: x["source"])

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results["ids"]:
            for chunk_id, doc, meta in zip(
                results["ids"], 
                results["documents"], 
                results["metadatas"]
            ):
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        chunk_text=doc,
                        metadata=ChunkMetadata.from_dict(meta)
                    )
                )
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        results = self._collection.get(include=["metadatas"])
        
        total_chunks = len(results["ids"])
        topics = set()
        sources = set()
        bonus_topics_present = False
        
        if results.get("metadatas"):
            for meta in results["metadatas"]:
                if meta:
                    topics.add(meta.get("topic", "unknown"))
                    sources.add(meta.get("source", "unknown"))
                    if str(meta.get("is_bonus", "False")).lower() == "true":
                        bonus_topics_present = True
                        
        return {
            "total_chunks": total_chunks,
            "topics": sorted(list(topics)),
            "sources": sorted(list(sources)),
            "bonus_topics_present": bonus_topics_present
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        results_before = self._collection.get(where={"source": source})
        count_before = len(results_before["ids"])
        if count_before > 0:
            self._collection.delete(where={"source": source})
        return count_before
