"""
config.py — Central configuration for the Agentic Doctor System RAG pipeline.
All tuneable parameters live here so nothing is hardcoded deeper in the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

class EmbeddingConfig:
    MODEL_NAME       = "models/embedding-001"
    TASK_TYPE_DOC    = "retrieval_document"
    TASK_TYPE_QUERY  = "retrieval_query"
    BATCH_SIZE       = 50
    RETRY_ATTEMPTS   = 3
    RETRY_DELAY_SEC  = 2

class VectorStoreConfig:
    PERSIST_DIR      = str(BASE_DIR / "vector_stores")

    COLLECTION_MEDICINE    = "medicine_data"
    COLLECTION_DOCTORS     = "doctor_data"
    COLLECTION_FIRST_AID   = "first_aid_data"
    COLLECTION_LAB_TESTS   = "lab_test_data"

    DISTANCE_METRIC        = "cosine"
    HNSW_SPACE             = "cosine"
    HNSW_EF_CONSTRUCTION   = 200
    HNSW_M                 = 32

class RetrievalConfig:
    TOP_K_DENSE            = 10
    TOP_K_BM25             = 10
    TOP_K_FINAL            = 5
    MMR_LAMBDA             = 0.6
    MMR_FETCH_K            = 20
    RRF_K                  = 60

class ChunkConfig:
    MEDICINE_CHUNK_SIZE    = 600
    MEDICINE_OVERLAP       = 80
    DOCTOR_CHUNK_SIZE      = 450
    DOCTOR_OVERLAP         = 50
    FIRST_AID_CHUNK_SIZE   = 700
    FIRST_AID_OVERLAP      = 100
    LABTEST_CHUNK_SIZE     = 800
    LABTEST_OVERLAP        = 100

class DataPaths:
    MEDICINE_JSON    = str(BASE_DIR / "data" / "medicine_data.json")
    DOCTORS_JSON     = str(BASE_DIR / "data" / "doctors_full.json")
    FIRST_AID_JSON   = str(BASE_DIR / "data" / "first_aid_data.json")
    LABTEST_JSON     = str(BASE_DIR / "data" / "lab_test_data.json")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
