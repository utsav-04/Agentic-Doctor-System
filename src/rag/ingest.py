"""
src/rag/ingest.py
==================
Run this ONCE before anything else. It:
  1. Validates all JSON data with Pydantic (catches bad records before they hit Gemini)
  2. Converts records to LangChain Documents with content + metadata
  3. Embeds with Gemini-embedding-001 in batches (rate-limit safe)
  4. Persists to ChromaDB under vector_stores/<collection>/
  5. Traces the entire run in LangSmith under project "agentic-doctor-system"

─── HOW TO RUN ────────────────────────────────────────────────────────────────

  From your project ROOT (agentic-doctor-system/):

  # First time — embed everything
  python -m src.rag.ingest

  # Re-embed a single collection (e.g. after updating medicine data)
  python -m src.rag.ingest --collection medicine

  # Force re-embed even if collection already exists
  python -m src.rag.ingest --force-rebuild

  # Dry-run — validate data only, no embedding
  python -m src.rag.ingest --dry-run

─── WHAT HAPPENS AFTER ────────────────────────────────────────────────────────

  vector_stores/
  ├── medicine_data/     ← ready for MedicineTool
  ├── doctor_data/       ← ready for DoctorTool
  ├── first_aid_data/    ← ready for FirstAidTool
  └── patient_data/      ← ready for patient context

  The CrewAI tools (src/tools/) load these at import time.
  You NEVER need to re-embed unless your raw data changes.

─── LANGSMITH TRACES ──────────────────────────────────────────────────────────

  Every run creates a LangSmith run named "rag_ingest_<collection>".
  You can see: validation errors, batch sizes, embedding duration,
  total vectors stored, and any retry events.
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Any

from langsmith import traceable, Client as LangSmithClient
from langchain_core.documents import Document

from src.rag.config import DataPaths, VectorStoreConfig, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
from src.logger import get_logger

logger = get_logger(__name__)
from src.schema.schemas import (
    MedicineRecord, DoctorRecord, FirstAidRecord, LabTestRecord,
    validate_json_records,
)
from src.rag.document_builder import (
    MedicineDocumentBuilder,
    DoctorDocumentBuilder,
    FirstAidDocumentBuilder,
    LabTestDocumentBuilder,
)
from src.rag.vector_store_manager import VectorStoreManager


os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)


PIPELINE_CONFIG = {
    "medicine": {
        "builder_cls":    MedicineDocumentBuilder,
        "collection":     VectorStoreConfig.COLLECTION_MEDICINE,
        "schema_cls":     MedicineRecord,
        "data_path":      DataPaths.MEDICINE_JSON,
        "description":    "Medicine data — symptoms, dosage, side effects",
    },
    "doctors": {
        "builder_cls":    DoctorDocumentBuilder,
        "collection":     VectorStoreConfig.COLLECTION_DOCTORS,
        "schema_cls":     DoctorRecord,
        "data_path":      DataPaths.DOCTORS_JSON,
        "description":    "Doctor data — 6520 doctors across all Indian states",
    },
    "first_aid": {
        "builder_cls":    FirstAidDocumentBuilder,
        "collection":     VectorStoreConfig.COLLECTION_FIRST_AID,
        "schema_cls":     FirstAidRecord,
        "data_path":      DataPaths.FIRST_AID_JSON,
        "description":    "First aid + Indian home remedies",
    },
    "lab_tests": {
        "builder_cls":    LabTestDocumentBuilder,
        "collection":     VectorStoreConfig.COLLECTION_LAB_TESTS,
        "schema_cls":     LabTestRecord,
        "data_path":      DataPaths.LABTEST_JSON,
        "description":    "Lab test recommendations mapped to patient symptom scenarios",
    },
}


@traceable(name="validate_raw_data", run_type="tool")
def validate_data(
    raw_data: List[Dict],
    schema_cls: Any,
    collection_key: str,
) -> Dict[str, Any]:
    """
    Validates raw JSON records against Pydantic schema.
    Traced in LangSmith: shows exactly how many records passed/failed.
    """
    if schema_cls is None:
        logger.info("[%s] Validation skipped (no schema defined)", collection_key)
        return {
            "collection": collection_key,
            "total": len(raw_data),
            "valid": len(raw_data),
            "invalid": 0,
            "error_log": [],
            "skipped_validation": True,
        }

    logger.info("[%s] Validating %d records with Pydantic...", collection_key, len(raw_data))
    valid_records, error_log = validate_json_records(raw_data, schema_cls, collection_key)

    if error_log:
        logger.warning(
            "[%s] %d/%d records failed validation",
            collection_key, len(error_log), len(raw_data)
        )
        for err in error_log[:5]:
            logger.warning("  Record %s: %s", err["id"], err["error"][:100])
        if len(error_log) > 5:
            logger.warning("  ... and %d more errors", len(error_log) - 5)
    else:
        logger.info("[%s] All %d records passed validation", collection_key, len(valid_records))

    return {
        "collection": collection_key,
        "total": len(raw_data),
        "valid": len(valid_records),
        "invalid": len(error_log),
        "error_log": error_log[:10],
        "skipped_validation": False,
    }


@traceable(name="build_documents", run_type="tool")
def build_documents(
    builder_cls: Any,
    collection_key: str,
) -> Dict[str, Any]:
    """
    Builds LangChain Documents from raw JSON.
    Traced: shows doc count, sample content length, metadata keys.
    """
    logger.info("[%s] Building LangChain documents...", collection_key)
    builder = builder_cls()
    documents = builder.build()

    sample_meta_keys = list(documents[0].metadata.keys()) if documents else []
    sample_content_length = len(documents[0].page_content) if documents else 0

    logger.info(
        "[%s] Built %d documents | metadata keys: %s | ~%d chars/doc",
        collection_key, len(documents), sample_meta_keys, sample_content_length
    )

    return {
        "collection": collection_key,
        "document_count": len(documents),
        "sample_metadata_keys": sample_meta_keys,
        "sample_content_chars": sample_content_length,
        "documents": documents,
    }


@traceable(name="embed_and_store", run_type="tool")
def embed_and_store(
    documents: List[Document],
    collection_name: str,
    manager: VectorStoreManager,
    force_rebuild: bool,
) -> Dict[str, Any]:
    """
    Embeds documents and persists to ChromaDB.
    Traced: shows batch count, embedding duration, final vector count.
    """
    logger.info(
        "Embedding %d documents into collection: %s (force_rebuild=%s)",
        len(documents), collection_name, force_rebuild
    )
    start = time.time()

    vectorstore = manager.build_or_load(
        collection_name=collection_name,
        documents=documents,
        force_rebuild=force_rebuild,
    )

    elapsed = time.time() - start
    vector_count = vectorstore._collection.count()

    logger.info(
        "Collection '%s' ready — %d vectors | took %.1fs",
        collection_name, vector_count, elapsed
    )

    return {
        "collection_name": collection_name,
        "vectors_stored": vector_count,
        "embedding_duration_seconds": round(elapsed, 2),
        "rebuilt": force_rebuild,
        "persist_path": f"vector_stores/{collection_name}",
    }


@traceable(name="run_ingestion_pipeline", run_type="chain")
def run_pipeline(
    collection_key: str,
    manager: VectorStoreManager,
    force_rebuild: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cfg = PIPELINE_CONFIG[collection_key]

    logger.info("=" * 54)
    logger.info("Pipeline: %s  →  %s", collection_key.upper(), cfg["collection"])
    logger.info("Description: %s", cfg["description"])
    logger.info("=" * 54)

    try:
        with open(cfg["data_path"], "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        logger.error("Data file not found: %s", cfg["data_path"])
        logger.error("Place your JSON file there and re-run.")
        return {"status": "skipped", "reason": "file_not_found"}

    logger.info("Raw records loaded: %d from %s", len(raw_data), cfg["data_path"])

    validation_result = validate_data(raw_data, cfg["schema_cls"], collection_key)
    logger.info(
        "Validation: %d/%d valid",
        validation_result["valid"], validation_result["total"]
    )

    if dry_run:
        logger.info("[DRY RUN] Stopping here — no embedding performed.")
        return {"status": "dry_run", "validation": validation_result}

    doc_result = build_documents(cfg["builder_cls"], collection_key)
    documents = doc_result["documents"]

    embed_result = embed_and_store(
        documents=documents,
        collection_name=cfg["collection"],
        manager=manager,
        force_rebuild=force_rebuild,
    )

    logger.info(
        "Pipeline '%s' complete — %d vectors in %.1fs",
        collection_key,
        embed_result["vectors_stored"],
        embed_result["embedding_duration_seconds"],
    )

    return {
        "status": "success",
        "collection": collection_key,
        "validation": validation_result,
        "documents": doc_result["document_count"],
        "embedding": embed_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest medical data into ChromaDB vector stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.rag.ingest                        # embed all collections
  python -m src.rag.ingest --collection medicine  # only medicine
  python -m src.rag.ingest --force-rebuild        # re-embed everything
  python -m src.rag.ingest --dry-run              # validate only, no embedding
        """,
    )
    parser.add_argument(
        "--collection",
        choices=list(PIPELINE_CONFIG.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 54)
    logger.info("Agentic Doctor System — RAG Ingestion Pipeline")
    logger.info("=" * 54)
    logger.info("Embedding model : models/embedding-001 (Gemini)")
    logger.info("Vector DB       : ChromaDB + HNSW (cosine)")
    logger.info("Retrieval stack : BM25 + Dense -> RRF -> MMR")
    logger.info("Validation      : Pydantic v2")
    logger.info("LangSmith       : %s", LANGCHAIN_PROJECT)
    logger.info("Force rebuild   : %s", args.force_rebuild)
    logger.info("Dry run         : %s", args.dry_run)
    logger.info("Log file        : logs/agentic_doctor.log")
    logger.info("=" * 54)

    manager = VectorStoreManager()
    collections = (
        list(PIPELINE_CONFIG.keys())
        if args.collection == "all"
        else [args.collection]
    )

    results = {}
    total_start = time.time()

    for key in collections:
        try:
            results[key] = run_pipeline(
                collection_key=key,
                manager=manager,
                force_rebuild=args.force_rebuild,
                dry_run=args.dry_run,
            )
        except FileNotFoundError as e:
            logger.error("[%s] Data file not found: %s", key, e)
            results[key] = {"status": "skipped"}
        except Exception as e:
            logger.error("[%s] Pipeline failed: %s", key, e, exc_info=True)
            results[key] = {"status": "error"}

    total_elapsed = time.time() - total_start

    logger.info("=" * 54)
    logger.info("ALL PIPELINES COMPLETE in %.1fs", total_elapsed)
    logger.info("=" * 54)
    for key, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            vecs = result["embedding"]["vectors_stored"]
            logger.info("  %-12s : %d vectors  [OK]", key, vecs)
        elif status == "skipped":
            logger.warning("  %-12s : SKIPPED — file not found", key)
        elif status == "dry_run":
            valid = result["validation"]["valid"]
            total_v = result["validation"]["total"]
            logger.info("  %-12s : %d/%d valid  [DRY RUN]", key, valid, total_v)
        elif status == "error":
            logger.error("  %-12s : FAILED — see log for details", key)

    logger.info("Vector stores : vector_stores/")
    logger.info("LangSmith     : https://smith.langchain.com")
    logger.info("Log file      : logs/agentic_doctor.log")
    logger.info("Next step     : python -m src.tools.run_tools_test")


if __name__ == "__main__":
    main()