"""
src/rag/retriever.py
=====================
Hybrid retrieval: metadata filter → BM25 + dense → RRF → MMR

Every method is decorated with @traceable so LangSmith captures:
  - The exact query sent to the retriever
  - The metadata filters applied
  - How many docs dense search returned
  - How many docs BM25 returned
  - The RRF-merged ranking
  - The final MMR-reranked results with their content
  - Total latency per stage

This means in LangSmith you can click any retrieval call and see
exactly what documents the agent used to generate its answer —
the single most important thing for a medical RAG system.
"""

import os
import time
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langsmith import traceable
from rank_bm25 import BM25Okapi

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    GOOGLE_API_KEY,
)


def _build_embeddings(task_type: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EmbeddingConfig.MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        task_type=task_type,
    )


def _build_chroma_filter(filters: Optional[Dict[str, Any]]) -> Optional[Dict]:
    """Convert simple key-value dict into Chroma $and/$eq filter syntax."""
    if not filters:
        return None
    conditions = [
        {k: {"$eq": str(v) if isinstance(v, bool) else v}}
        for k, v in filters.items()
    ]
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class HybridRetriever:
    """
    Full hybrid retriever: metadata filter → BM25 + dense → RRF → MMR.
    Every stage is individually traced in LangSmith.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._embeddings = _build_embeddings(EmbeddingConfig.TASK_TYPE_QUERY)
        persist_path = os.path.join(VectorStoreConfig.PERSIST_DIR, collection_name)
        self._vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_path,
            collection_metadata={
                "hnsw:space":           VectorStoreConfig.HNSW_SPACE,
                "hnsw:M":               VectorStoreConfig.HNSW_M,
                "hnsw:ef_construction": VectorStoreConfig.HNSW_EF_CONSTRUCTION,
            },
        )

    @traceable(name="hybrid_retrieve", run_type="retriever")
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = RetrievalConfig.TOP_K_FINAL,
    ) -> Dict[str, Any]:
        """
        Main retrieval entry point — fully traced in LangSmith.

        LangSmith will show:
          - query text
          - filters applied
          - dense results count
          - BM25 results count
          - final documents with their full page_content and metadata

        Returns dict (not just docs) so LangSmith captures rich metadata.
        """
        t_start = time.time()
        chroma_filter = _build_chroma_filter(filters)

        # Stage 1: Dense retrieval
        dense_result = self._dense_retrieve(query, chroma_filter)
        dense_docs = dense_result["documents"]

        # Stage 2: Candidate pool for BM25
        pool_result = self._get_candidate_pool(query, chroma_filter)
        candidate_docs = pool_result["documents"]

        # Stage 3: BM25 sparse retrieval
        bm25_result = self._bm25_retrieve(query, candidate_docs)
        bm25_docs = bm25_result["documents"]

        # Stage 4: RRF merge
        fused_result = self._reciprocal_rank_fusion(dense_docs, bm25_docs)
        fused_docs = fused_result["documents"]

        # Stage 5: MMR rerank
        mmr_result = self._mmr_rerank(query, fused_docs, top_k)
        final_docs = mmr_result["documents"]

        elapsed = round(time.time() - t_start, 3)

        return {
            "query": query,
            "filters_applied": filters or {},
            "collection": self.collection_name,
            "stage_counts": {
                "dense": dense_result["count"],
                "bm25":  bm25_result["count"],
                "fused": fused_result["count"],
                "final": len(final_docs),
            },
            "retrieval_latency_seconds": elapsed,
            "documents": final_docs,
            "document_contents": [
                {
                    "id":       doc.metadata.get("id", ""),
                    "name":     doc.metadata.get("name", ""),
                    "content":  doc.page_content[:300],
                    "metadata": doc.metadata,
                }
                for doc in final_docs
            ],
        }

    @traceable(name="dense_retrieve_stage", run_type="retriever")
    def _dense_retrieve(
        self,
        query: str,
        chroma_filter: Optional[Dict],
    ) -> Dict[str, Any]:
        """Gemini embedding → HNSW cosine similarity search."""
        kwargs: Dict[str, Any] = {"query": query, "k": RetrievalConfig.TOP_K_DENSE}
        if chroma_filter:
            kwargs["filter"] = chroma_filter
        docs = self._vectorstore.similarity_search(**kwargs)
        return {
            "stage": "dense",
            "count": len(docs),
            "top_ids": [d.metadata.get("id", "") for d in docs[:3]],
            "documents": docs,
        }

    @traceable(name="candidate_pool_stage", run_type="retriever")
    def _get_candidate_pool(
        self,
        query: str,
        chroma_filter: Optional[Dict],
    ) -> Dict[str, Any]:
        """Larger pool for BM25 to score against."""
        kwargs: Dict[str, Any] = {"query": query, "k": RetrievalConfig.MMR_FETCH_K}
        if chroma_filter:
            kwargs["filter"] = chroma_filter
        docs = self._vectorstore.similarity_search(**kwargs)
        return {"stage": "candidate_pool", "count": len(docs), "documents": docs}

    @traceable(name="bm25_retrieve_stage", run_type="retriever")
    def _bm25_retrieve(
        self,
        query: str,
        candidate_docs: List[Document],
    ) -> Dict[str, Any]:
        """BM25 Okapi over candidate pool — catches exact medical term matches."""
        if not candidate_docs:
            return {"stage": "bm25", "count": 0, "documents": [], "top_terms": []}

        tokenized_corpus = [doc.page_content.lower().split() for doc in candidate_docs]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        scored = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
        top_n = min(RetrievalConfig.TOP_K_BM25, len(scored))
        top_docs = [doc for _, doc in scored[:top_n]]
        top_scores = [round(float(s), 4) for s, _ in scored[:top_n]]

        return {
            "stage": "bm25",
            "count": len(top_docs),
            "top_scores": top_scores[:3],
            "top_ids": [d.metadata.get("id", "") for d in top_docs[:3]],
            "documents": top_docs,
        }

    @traceable(name="rrf_fusion_stage", run_type="retriever")
    def _reciprocal_rank_fusion(
        self,
        dense_docs: List[Document],
        bm25_docs: List[Document],
    ) -> Dict[str, Any]:
        """RRF merges dense + sparse without requiring score normalization."""
        k = RetrievalConfig.RRF_K
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            key = doc.page_content[:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs, start=1):
            key = doc.page_content[:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        fused = [doc_map[k] for k in sorted_keys]

        return {
            "stage": "rrf",
            "count": len(fused),
            "top_rrf_scores": [round(rrf_scores[k], 6) for k in sorted_keys[:3]],
            "top_ids": [doc_map[k].metadata.get("id", "") for k in sorted_keys[:3]],
            "documents": fused,
        }

    @traceable(name="mmr_rerank_stage", run_type="retriever")
    def _mmr_rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int,
    ) -> Dict[str, Any]:
        """
        MMR reranking — balances relevance (60%) with diversity (40%).
        Prevents returning near-identical medicine or doctor records.
        """
        if not docs:
            return {"stage": "mmr", "count": 0, "documents": [], "fallback": False}

        try:
            query_vec = np.array(self._embeddings.embed_query(query))
            doc_vecs = np.array(self._embeddings.embed_documents(
                [doc.page_content for doc in docs]
            ))
        except Exception as e:
            return {
                "stage": "mmr",
                "count": min(top_k, len(docs)),
                "documents": docs[:top_k],
                "fallback": True,
                "fallback_reason": str(e),
            }

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

        lmb = RetrievalConfig.MMR_LAMBDA
        selected: List[int] = []
        remaining = list(range(len(docs)))

        while len(selected) < top_k and remaining:
            if not selected:
                scores = [cosine(query_vec, doc_vecs[i]) for i in remaining]
                best = remaining[int(np.argmax(scores))]
            else:
                best, best_score = remaining[0], -float("inf")
                for i in remaining:
                    rel = cosine(query_vec, doc_vecs[i])
                    redundancy = max(cosine(doc_vecs[i], doc_vecs[j]) for j in selected)
                    mmr = lmb * rel - (1 - lmb) * redundancy
                    if mmr > best_score:
                        best_score, best = mmr, i
            selected.append(best)
            remaining.remove(best)

        final = [docs[i] for i in selected]
        return {
            "stage": "mmr",
            "lambda": lmb,
            "count": len(final),
            "selected_indices": selected,
            "documents": final,
            "fallback": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALISED RETRIEVERS  (used directly by CrewAI tools)
# ─────────────────────────────────────────────────────────────────────────────

class MedicineRetriever(HybridRetriever):
    def __init__(self):
        super().__init__(VectorStoreConfig.COLLECTION_MEDICINE)

    @traceable(name="medicine_retriever", run_type="retriever")
    def get_medicines(
        self,
        symptoms: str,
        disease: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {"data_type": "medicine"}
        if disease:
            filters["disease"] = disease
        return self.retrieve(query=symptoms, filters=filters, top_k=top_k)


class DoctorRetriever(HybridRetriever):
    def __init__(self):
        super().__init__(VectorStoreConfig.COLLECTION_DOCTORS)

    @traceable(name="doctor_retriever", run_type="retriever")
    def get_doctors(
        self,
        symptoms: str,
        state: str,
        city: Optional[str] = None,
        department: Optional[str] = None,
        emergency_only: bool = False,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {"data_type": "doctor", "state": state}
        if city:
            filters["city"] = city
        if department:
            filters["department"] = department
        if emergency_only:
            filters["accepts_emergency"] = "True"
        return self.retrieve(query=symptoms, filters=filters, top_k=top_k)


class FirstAidRetriever(HybridRetriever):
    def __init__(self):
        super().__init__(VectorStoreConfig.COLLECTION_FIRST_AID)

    @traceable(name="first_aid_retriever", run_type="retriever")
    def get_first_aid(
        self,
        condition: str,
        severity: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {"data_type": "first_aid"}
        if severity:
            filters["severity_level"] = severity
        return self.retrieve(query=condition, filters=filters, top_k=top_k)



class LabTestRetriever(HybridRetriever):
    def __init__(self):
        super().__init__(VectorStoreConfig.COLLECTION_LAB_TESTS)

    @traceable(name="lab_test_retriever", run_type="retriever")
    def get_lab_tests(
        self,
        symptoms: str,
        severity: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Retrieves lab test recommendations matching the user's symptom description.

        Args:
            symptoms: User's symptoms in plain language
            severity: Optional filter — 'low', 'medium', 'high', 'critical'
            top_k:    Number of results to return

        LangSmith traces:
            - query sent to retriever
            - severity filter applied
            - dense + BM25 + RRF + MMR stages
            - final documents with test names and reasons
        """
        filters: Dict[str, Any] = {"data_type": "lab_test"}
        if severity:
            filters["severity"] = severity
        return self.retrieve(query=symptoms, filters=filters, top_k=top_k)
