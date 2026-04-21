"""
src/rag/vector_store_manager.py
================================
Resumable embedding pipeline with checkpoint tracking and full logging.

Every stage logs to both terminal (colored) and logs/agentic_doctor.log.

What you see in the terminal during ingestion:

    2025-01-08 10:30:00 | INFO     | rag.vector_store_manager | ──────────────────────────────────────
    2025-01-08 10:30:00 | INFO     | rag.vector_store_manager | Collection  : doctor_data
    2025-01-08 10:30:00 | INFO     | rag.vector_store_manager | Total docs  : 6520
    2025-01-08 10:30:00 | INFO     | rag.vector_store_manager | Batch size  : 50
    2025-01-08 10:30:00 | INFO     | rag.vector_store_manager | Total batches: 131
    2025-01-08 10:30:00 | INFO     | rag.vector_store_manager | ──────────────────────────────────────
    2025-01-08 10:30:01 | INFO     | rag.vector_store_manager | [Batch  1/131] docs    0–49   | embedded:    50 | remaining: 6470 | elapsed: 0:00:01 | ETA: ~0:21:50
    2025-01-08 10:30:09 | INFO     | rag.vector_store_manager | [Batch  2/131] docs   50–99   | embedded:   100 | remaining: 6420 | elapsed: 0:00:09 | ETA: ~0:19:31
    2025-01-08 10:31:30 | WARNING  | rag.vector_store_manager | [Batch  3/131] QUOTA EXHAUSTED — waiting 60s (attempt 1/3)
    2025-01-08 10:32:30 | INFO     | rag.vector_store_manager | [Batch  3/131] Retry 2 of 3...
    2025-01-08 10:32:38 | INFO     | rag.vector_store_manager | [Batch  3/131] docs  100–149   | embedded:   150 | remaining: 6370 | elapsed: 0:02:38 | ETA: ~0:57:22
    ...
    2025-01-08 10:30:00 | INFO     | rag.vector_store_manager | ✓ Ingestion complete — 6520 vectors in doctor_data (elapsed: 0:22:14)
"""

import os
import json
import time
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import List, Set

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import sys
from pathlib import Path

from src.rag.config import EmbeddingConfig, VectorStoreConfig, GOOGLE_API_KEY
from src.logger import get_logger

logger = get_logger(__name__)

CHECKPOINT_FILENAME = ".ingest_checkpoint.json"


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────

class GeminiEmbedder:
    """Wraps Gemini-embedding-001 with validation."""

    def __init__(self):
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not found in environment.")
            raise ValueError(
                "GOOGLE_API_KEY not set.\n"
                "Add it to your .env file.\n"
                "Get one at: https://aistudio.google.com/app/apikey"
            )
        logger.debug("Initialising Gemini embedder: %s", EmbeddingConfig.MODEL_NAME)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EmbeddingConfig.MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            task_type=EmbeddingConfig.TASK_TYPE_DOC,
        )
        logger.info("Gemini embedder ready — model: %s", EmbeddingConfig.MODEL_NAME)

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return self.embeddings


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

class IngestCheckpoint:
    """
    Reads and writes the checkpoint JSON file for a collection.
    Checkpoint is saved after every successful batch so progress
    is never lost on quota exhaustion or crashes.
    """

    def __init__(self, persist_path: str, collection_name: str):
        self.path             = os.path.join(persist_path, CHECKPOINT_FILENAME)
        self.collection_name  = collection_name
        self._data: dict      = {}

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def load(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        logger.debug(
            "Checkpoint loaded — last index: %d | embedded: %d",
            self._data.get("last_completed_index", -1),
            self._data.get("embedded_count", 0),
        )
        return self._data

    def save(
        self,
        total_documents: int,
        last_completed_batch: int,
        last_completed_index: int,
        embedded_ids: List[str],
    ) -> None:
        now = datetime.utcnow().isoformat()
        self._data = {
            "collection_name":      self.collection_name,
            "total_documents":      total_documents,
            "last_completed_batch": last_completed_batch,
            "last_completed_index": last_completed_index,
            "embedded_count":       len(embedded_ids),
            "embedded_ids":         embedded_ids,
            "updated_at":           now,
        }
        if "started_at" not in self._data:
            self._data["started_at"] = now

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

        logger.debug(
            "Checkpoint saved — batch: %d | index: %d | total embedded: %d",
            last_completed_batch,
            last_completed_index,
            len(embedded_ids),
        )

    def delete(self) -> None:
        if self.exists():
            os.remove(self.path)
            logger.info("Checkpoint deleted: %s", self.path)

    @property
    def last_completed_index(self) -> int:
        return self._data.get("last_completed_index", -1)

    @property
    def embedded_ids(self) -> Set[str]:
        return set(self._data.get("embedded_ids", []))

    @property
    def embedded_count(self) -> int:
        return self._data.get("embedded_count", 0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_doc_id(doc: Document, index: int) -> str:
    meta_id = doc.metadata.get("id", "")
    if meta_id:
        return meta_id
    content_hash = hashlib.md5(
        f"{index}:{doc.page_content[:100]}".encode()
    ).hexdigest()[:12]
    return f"doc_{index}_{content_hash}"


def _fmt_duration(seconds: float) -> str:
    """Formats seconds into H:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))


def _eta(elapsed: float, done: int, total: int) -> str:
    """Calculates ETA based on current throughput."""
    if done == 0:
        return "calculating..."
    rate     = done / elapsed           # docs per second
    remaining = total - done
    eta_secs  = remaining / rate
    return f"~{_fmt_duration(eta_secs)}"


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    """Returns a simple ASCII progress bar string."""
    pct   = done / total if total else 0
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Builds or loads ChromaDB collections with resumable embedding.

    On first run   : embeds all documents, saves checkpoint after every batch.
    On resume      : reads checkpoint, skips already-embedded docs, continues.
    On completion  : checkpoint stays as a completion record.
    On force-rebuild: deletes checkpoint and ChromaDB, re-embeds from scratch.
    """

    def __init__(self):
        self.embedder   = GeminiEmbedder()
        self.embeddings = self.embedder.get_embeddings()
        os.makedirs(VectorStoreConfig.PERSIST_DIR, exist_ok=True)

    def build_or_load(
        self,
        collection_name: str,
        documents: List[Document],
        force_rebuild: bool = False,
    ) -> Chroma:
        persist_path = os.path.join(VectorStoreConfig.PERSIST_DIR, collection_name)
        os.makedirs(persist_path, exist_ok=True)

        checkpoint = IngestCheckpoint(persist_path, collection_name)

        # ── Force rebuild ─────────────────────────────────────────
        if force_rebuild:
            logger.warning(
                "Force rebuild requested — clearing collection: %s", collection_name
            )
            checkpoint.delete()
            self._clear_chroma_collection(persist_path, collection_name)

        # ── Already fully embedded ────────────────────────────────
        if checkpoint.exists():
            checkpoint.load()
            already_done = checkpoint.last_completed_index >= len(documents) - 1
            if already_done:
                logger.info(
                    "Collection already complete — %d vectors on disk. Loading.",
                    checkpoint.embedded_count,
                )
                return self._load(persist_path, collection_name)

            # ── Partial — resume ──────────────────────────────────
            resume_from      = checkpoint.last_completed_index + 1
            already_embedded = checkpoint.embedded_ids
            logger.info(
                "Resuming ingestion — starting from doc %d/%d | already embedded: %d",
                resume_from,
                len(documents),
                checkpoint.embedded_count,
            )
            return self._embed_in_batches(
                collection_name=collection_name,
                documents=documents,
                persist_path=persist_path,
                checkpoint=checkpoint,
                resume_from=resume_from,
                already_embedded_ids=already_embedded,
            )

        # ── Fresh start ───────────────────────────────────────────
        logger.info(
            "Fresh ingestion — %d documents → collection: %s",
            len(documents),
            collection_name,
        )
        return self._embed_in_batches(
            collection_name=collection_name,
            documents=documents,
            persist_path=persist_path,
            checkpoint=checkpoint,
            resume_from=0,
            already_embedded_ids=set(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CORE BATCH EMBEDDING
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_in_batches(
        self,
        collection_name: str,
        documents: List[Document],
        persist_path: str,
        checkpoint: IngestCheckpoint,
        resume_from: int,
        already_embedded_ids: Set[str],
    ) -> Chroma:

        batch_size     = EmbeddingConfig.BATCH_SIZE
        total          = len(documents)
        vectorstore    = None
        embedded_ids   = list(already_embedded_ids)

        # Load existing store if resuming
        if resume_from > 0:
            logger.debug("Loading existing vectorstore from disk for resume.")
            vectorstore = self._load(persist_path, collection_name)

        remaining_docs  = documents[resume_from:]
        total_remaining = len(remaining_docs)
        n_batches       = (total_remaining + batch_size - 1) // batch_size

        if total_remaining == 0:
            logger.info("Nothing left to embed. Collection is complete.")
            return vectorstore or self._load(persist_path, collection_name)

        # ── Session header ────────────────────────────────────────
        logger.info("─" * 54)
        logger.info("Collection   : %s", collection_name)
        logger.info("Total docs   : %d", total)
        logger.info("Already done : %d", resume_from)
        logger.info("Remaining    : %d", total_remaining)
        logger.info("Batch size   : %d", batch_size)
        logger.info("Batches left : %d", n_batches)
        logger.info("Checkpoint   : saved after every batch")
        logger.info("─" * 54)

        session_start = time.time()
        docs_done_this_session = 0

        for batch_offset in range(0, total_remaining, batch_size):
            batch        = remaining_docs[batch_offset: batch_offset + batch_size]
            batch_num    = (batch_offset // batch_size) + 1
            global_start = resume_from + batch_offset
            global_end   = global_start + len(batch) - 1

            elapsed       = time.time() - session_start
            total_embedded_so_far = resume_from + batch_offset
            eta_str       = _eta(elapsed, docs_done_this_session, total_remaining)
            progress      = _progress_bar(total_embedded_so_far, total)
            pct_overall   = (total_embedded_so_far / total) * 100

            logger.info(
                "[Batch %3d/%d] docs %4d–%4d | embedded so far: %4d/%d (%.1f%%) | ETA: %s",
                batch_num, n_batches,
                global_start, global_end,
                total_embedded_so_far, total,
                pct_overall,
                eta_str,
            )
            logger.debug("Progress bar: %s", progress)

            success = False
            for attempt in range(EmbeddingConfig.RETRY_ATTEMPTS):
                try:
                    t_batch_start = time.time()

                    if vectorstore is None:
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            collection_name=collection_name,
                            persist_directory=persist_path,
                            collection_metadata=self._hnsw_meta(),
                        )
                    else:
                        vectorstore.add_documents(batch)

                    batch_elapsed = time.time() - t_batch_start
                    success = True

                    logger.debug(
                        "[Batch %d] Embedded %d docs in %.2fs (%.1f docs/s)",
                        batch_num,
                        len(batch),
                        batch_elapsed,
                        len(batch) / batch_elapsed if batch_elapsed > 0 else 0,
                    )
                    break

                except Exception as e:
                    err_msg  = str(e).lower()
                    is_quota = any(k in err_msg for k in [
                        "quota", "rate", "429", "resource_exhausted",
                        "limit", "exceeded", "too many requests",
                    ])

                    if is_quota:
                        wait = 60 * (attempt + 1)
                        logger.warning(
                            "[Batch %d] QUOTA EXHAUSTED — waiting %ds "
                            "(attempt %d/%d) | Progress safe in checkpoint.",
                            batch_num, wait,
                            attempt + 1, EmbeddingConfig.RETRY_ATTEMPTS,
                        )
                    else:
                        wait = EmbeddingConfig.RETRY_DELAY_SEC * (attempt + 1)
                        logger.warning(
                            "[Batch %d] Error on attempt %d/%d: %s — retrying in %ds",
                            batch_num,
                            attempt + 1, EmbeddingConfig.RETRY_ATTEMPTS,
                            str(e)[:120],
                            wait,
                        )

                    if attempt + 1 < EmbeddingConfig.RETRY_ATTEMPTS:
                        logger.info(
                            "[Batch %d] Retry %d of %d in %ds...",
                            batch_num, attempt + 2, EmbeddingConfig.RETRY_ATTEMPTS, wait,
                        )
                    time.sleep(wait)

            # ── All retries exhausted ─────────────────────────────
            if not success:
                checkpoint.save(
                    total_documents=total,
                    last_completed_batch=(resume_from // batch_size) + batch_num - 1,
                    last_completed_index=global_start - 1,
                    embedded_ids=embedded_ids,
                )
                logger.error(
                    "[Batch %d] All %d retries exhausted. "
                    "Checkpoint saved at doc %d. "
                    "Run again to resume: python -m src.rag.ingest --collection %s",
                    batch_num,
                    EmbeddingConfig.RETRY_ATTEMPTS,
                    global_start - 1,
                    self._collection_key(collection_name),
                )
                raise RuntimeError(
                    f"Embedding stopped at batch {batch_num}. "
                    f"Checkpoint saved — re-run to resume."
                )

            # ── Batch succeeded — update checkpoint ───────────────
            batch_ids = [
                _get_doc_id(doc, global_start + j)
                for j, doc in enumerate(batch)
            ]
            embedded_ids.extend(batch_ids)
            docs_done_this_session += len(batch)

            checkpoint.save(
                total_documents=total,
                last_completed_batch=(resume_from // batch_size) + batch_num,
                last_completed_index=global_end,
                embedded_ids=embedded_ids,
            )

            vectors_in_store = vectorstore._collection.count()
            elapsed_total    = time.time() - session_start
            remaining_docs_count = total - (resume_from + batch_offset + len(batch))

            logger.info(
                "[Batch %3d/%d] ✓ Done | vectors in store: %d | "
                "remaining docs: %d | elapsed: %s",
                batch_num, n_batches,
                vectors_in_store,
                remaining_docs_count,
                _fmt_duration(elapsed_total),
            )

            # Rate limit buffer
            if batch_offset + batch_size < total_remaining:
                time.sleep(1.5)

        # ── Session complete ──────────────────────────────────────
        total_elapsed = time.time() - session_start
        logger.info("─" * 54)
        logger.info(
            "✓ Ingestion complete — %d vectors in %s | elapsed: %s",
            len(embedded_ids),
            collection_name,
            _fmt_duration(total_elapsed),
        )
        logger.info("  Log file: logs/agentic_doctor.log")
        logger.info("─" * 54)

        return vectorstore

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, persist_path: str, collection_name: str) -> Chroma:
        logger.debug("Loading vectorstore from: %s", persist_path)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_path,
            collection_metadata=self._hnsw_meta(),
        )

    def _clear_chroma_collection(self, persist_path: str, collection_name: str) -> None:
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
            os.makedirs(persist_path, exist_ok=True)
            logger.info("Cleared ChromaDB collection at: %s", persist_path)

    @staticmethod
    def _collection_key(collection_name: str) -> str:
        return {
            "medicine_data":  "medicine",
            "doctor_data":    "doctors",
            "first_aid_data": "first_aid",
            "lab_test_data":  "lab_tests",
        }.get(collection_name, collection_name)

    @staticmethod
    def _hnsw_meta() -> dict:
        # Only hnsw:space is reliably supported across ChromaDB versions.
        # Passing hnsw:M or hnsw:ef_construction causes:
        # "Failed to parse hnsw parameters from segment metadata"
        return {
            "hnsw:space": VectorStoreConfig.HNSW_SPACE,
        }