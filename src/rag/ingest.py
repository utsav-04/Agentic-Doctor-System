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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DataPaths, VectorStoreConfig, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
from src.schemas import (
    MedicineRecord, DoctorRecord, FirstAidRecord,
    validate_json_records,
)
from src.rag.document_builder import (
    MedicineDocumentBuilder,
    DoctorDocumentBuilder,
    FirstAidDocumentBuilder,
    PatientDocumentBuilder,
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
    "patients": {
        "builder_cls":    PatientDocumentBuilder,
        "collection":     VectorStoreConfig.COLLECTION_PATIENTS,
        "schema_cls":     None,
        "data_path":      DataPaths.PATIENTS_JSON,
        "description":    "Patient records for context",
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
        return {
            "collection": collection_key,
            "total": len(raw_data),
            "valid": len(raw_data),
            "invalid": 0,
            "error_log": [],
            "skipped_validation": True,
        }

    valid_records, error_log = validate_json_records(raw_data, schema_cls, collection_key)

    result = {
        "collection": collection_key,
        "total": len(raw_data),
        "valid": len(valid_records),
        "invalid": len(error_log),
        "error_log": error_log[:10],
        "skipped_validation": False,
    }

    if error_log:
        print(f"\n  [WARN] {len(error_log)} validation errors in {collection_key}:")
        for err in error_log[:5]:
            print(f"    Record {err['id']}: {err['error'][:100]}")
        if len(error_log) > 5:
            print(f"    ... and {len(error_log) - 5} more")

    return result


@traceable(name="build_documents", run_type="tool")
def build_documents(
    builder_cls: Any,
    collection_key: str,
) -> Dict[str, Any]:
    """
    Builds LangChain Documents from raw JSON.
    Traced: shows doc count, sample content length, metadata keys.
    """
    builder = builder_cls()
    documents = builder.build()

    sample_meta_keys = list(documents[0].metadata.keys()) if documents else []
    sample_content_length = len(documents[0].page_content) if documents else 0

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
    start = time.time()

    vectorstore = manager.build_or_load(
        collection_name=collection_name,
        documents=documents,
        force_rebuild=force_rebuild,
    )

    elapsed = time.time() - start
    vector_count = vectorstore._collection.count()

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
    """
    Full pipeline for one collection.
    This is the parent trace — validation, building, embedding are child traces.
    In LangSmith you see the full tree: pipeline → validate → build → embed
    """
    cfg = PIPELINE_CONFIG[collection_key]
    print(f"\n{'─'*56}")
    print(f"  Collection : {collection_key}")
    print(f"  Description: {cfg['description']}")
    print(f"{'─'*56}")

    # Load raw data
    try:
        with open(cfg["data_path"], "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"  [SKIP] Data file not found: {cfg['data_path']}")
        print(f"         Place your JSON file there and re-run.")
        return {"status": "skipped", "reason": "file_not_found"}

    print(f"  Raw records loaded : {len(raw_data)}")

    # Step 1: Validate
    validation_result = validate_data(raw_data, cfg["schema_cls"], collection_key)
    print(f"  Valid records      : {validation_result['valid']}/{validation_result['total']}")

    if dry_run:
        print(f"  [DRY RUN] Stopping here — no embedding performed.")
        return {"status": "dry_run", "validation": validation_result}

    # Step 2: Build Documents
    doc_result = build_documents(cfg["builder_cls"], collection_key)
    documents = doc_result["documents"]
    print(f"  Documents built    : {doc_result['document_count']}")
    print(f"  Metadata keys      : {doc_result['sample_metadata_keys']}")
    print(f"  Content length     : ~{doc_result['sample_content_chars']} chars/doc")

    # Step 3: Embed + Store
    print(f"  Embedding...       (this takes a few minutes for large collections)")
    embed_result = embed_and_store(
        documents=documents,
        collection_name=cfg["collection"],
        manager=manager,
        force_rebuild=force_rebuild,
    )
    print(f"  Vectors stored     : {embed_result['vectors_stored']}")
    print(f"  Embedding time     : {embed_result['embedding_duration_seconds']}s")
    print(f"  Persisted to       : {embed_result['persist_path']}/")

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

    print("\n" + "="*56)
    print("  Agentic Doctor System — RAG Ingestion Pipeline")
    print("="*56)
    print(f"  Embedding model   : models/embedding-001 (Gemini)")
    print(f"  Vector DB         : ChromaDB + HNSW (cosine)")
    print(f"  Retrieval stack   : BM25 + Dense → RRF → MMR")
    print(f"  Validation        : Pydantic v2")
    print(f"  Tracing           : LangSmith → {LANGCHAIN_PROJECT}")
    print(f"  Force rebuild     : {args.force_rebuild}")
    print(f"  Dry run           : {args.dry_run}")
    print("="*56)

    manager = VectorStoreManager()
    collections = (
        list(PIPELINE_CONFIG.keys())
        if args.collection == "all"
        else [args.collection]
    )

    results = {}
    total_start = time.time()

    for key in collections:
        results[key] = run_pipeline(
            collection_key=key,
            manager=manager,
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run,
        )

    total_elapsed = time.time() - total_start

    print(f"\n{'='*56}")
    print(f"  INGESTION COMPLETE — {total_elapsed:.1f}s total")
    print(f"{'='*56}")
    for key, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            vecs = result["embedding"]["vectors_stored"]
            print(f"  {key:<12} : {vecs} vectors  [OK]")
        elif status == "skipped":
            print(f"  {key:<12} : SKIPPED — file not found")
        elif status == "dry_run":
            valid = result["validation"]["valid"]
            total = result["validation"]["total"]
            print(f"  {key:<12} : {valid}/{total} records valid  [DRY RUN]")

    print(f"\n  Vector stores saved to: vector_stores/")
    print(f"  LangSmith traces at  : https://smith.langchain.com")
    print(f"\n  Next step: python -m src.tools.run_tools_test")
    print()


if __name__ == "__main__":
    main()
