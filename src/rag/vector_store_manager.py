"""
vector_store_manager.py
========================
Handles:
  1. Embedding with Gemini-embedding-001 (batched to stay within rate limits)
  2. Storing documents in ChromaDB with HNSW index (cosine distance)
  3. Persisting to disk for reuse across sessions

Indexing choices:
  - ChromaDB uses HNSW (Hierarchical Navigable Small World) by default
  - HNSW params: M=32, ef_construction=200 → high recall, moderate build time
  - Cosine distance → appropriate for normalized semantic embeddings
  - Each collection is isolated → no cross-collection contamination during retrieval
"""

import time
import os
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import EmbeddingConfig, VectorStoreConfig, GOOGLE_API_KEY


class GeminiEmbedder:
    """
    Wraps Gemini-embedding-001 with batching and retry logic.
    Uses task_type='retrieval_document' for ingestion,
    'retrieval_query' is used at query time (handled in retriever.py).
    """

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file.\n"
                "Get one at: https://aistudio.google.com/app/apikey"
            )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EmbeddingConfig.MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            task_type=EmbeddingConfig.TASK_TYPE_DOC,
        )

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return self.embeddings


class VectorStoreManager:
    """
    Creates and persists a ChromaDB collection for a given document set.

    ChromaDB HNSW settings are set via collection metadata at creation time.
    Once a collection exists on disk, it is loaded from disk (not rebuilt).
    """

    def __init__(self):
        self.embedder = GeminiEmbedder()
        self.embeddings = self.embedder.get_embeddings()
        os.makedirs(VectorStoreConfig.PERSIST_DIR, exist_ok=True)

    def build_or_load(
        self,
        collection_name: str,
        documents: List[Document],
        force_rebuild: bool = False,
    ) -> Chroma:
        """
        If the collection already exists on disk and force_rebuild=False,
        load from disk. Otherwise embed all documents and persist.

        Args:
            collection_name: One of the names defined in VectorStoreConfig
            documents:       LangChain Document objects from document_builder.py
            force_rebuild:   Set True to re-embed even if collection exists
        """
        persist_path = os.path.join(
            VectorStoreConfig.PERSIST_DIR, collection_name
        )

        collection_exists = (
            os.path.exists(persist_path)
            and os.path.isdir(persist_path)
            and any(os.scandir(persist_path))
        )

        if collection_exists and not force_rebuild:
            print(f"[VectorStore] Loading existing collection: {collection_name}")
            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_path,
                collection_metadata={
                    "hnsw:space":           VectorStoreConfig.HNSW_SPACE,
                    "hnsw:M":               VectorStoreConfig.HNSW_M,
                    "hnsw:ef_construction": VectorStoreConfig.HNSW_EF_CONSTRUCTION,
                },
            )

        print(f"[VectorStore] Building collection: {collection_name}")
        print(f"  Documents to embed: {len(documents)}")

        vectorstore = self._embed_in_batches(
            collection_name=collection_name,
            documents=documents,
            persist_path=persist_path,
        )
        print(f"[VectorStore] Collection persisted to: {persist_path}")
        return vectorstore

    def _embed_in_batches(
        self,
        collection_name: str,
        documents: List[Document],
        persist_path: str,
    ) -> Chroma:
        """
        Embeds documents in batches to respect Gemini rate limits.
        First batch creates the collection; subsequent batches add to it.
        """
        batch_size = EmbeddingConfig.BATCH_SIZE
        vectorstore = None

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} docs)...")

            success = False
            for attempt in range(EmbeddingConfig.RETRY_ATTEMPTS):
                try:
                    if vectorstore is None:
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            collection_name=collection_name,
                            persist_directory=persist_path,
                            collection_metadata={
                                "hnsw:space":           VectorStoreConfig.HNSW_SPACE,
                                "hnsw:M":               VectorStoreConfig.HNSW_M,
                                "hnsw:ef_construction": VectorStoreConfig.HNSW_EF_CONSTRUCTION,
                            },
                        )
                    else:
                        vectorstore.add_documents(batch)
                    success = True
                    break
                except Exception as e:
                    wait = EmbeddingConfig.RETRY_DELAY_SEC * (attempt + 1)
                    print(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)

            if not success:
                raise RuntimeError(
                    f"Failed to embed batch {batch_num} after "
                    f"{EmbeddingConfig.RETRY_ATTEMPTS} attempts"
                )

            if i + batch_size < len(documents):
                time.sleep(1.0)

        return vectorstore

    def load_collection(self, collection_name: str) -> Chroma:
        """Load an already-persisted collection without rebuilding."""
        persist_path = os.path.join(
            VectorStoreConfig.PERSIST_DIR, collection_name
        )
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_path,
            collection_metadata={
                "hnsw:space":           VectorStoreConfig.HNSW_SPACE,
                "hnsw:M":               VectorStoreConfig.HNSW_M,
                "hnsw:ef_construction": VectorStoreConfig.HNSW_EF_CONSTRUCTION,
            },
        )
