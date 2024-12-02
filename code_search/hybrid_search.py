from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
    SearchRequest,
    NamedVector,
    NamedSparseVector
)
from fastembed import TextEmbedding, SparseTextEmbedding

@dataclass
class SearchResult:
    id: str
    score: float
    payload: Dict[str, Any]

class HybridSearch:
    def __init__(
        self,
        collection_name: str,
        dense_model_name: str = "BAAI/bge-large-en-v1.5",
        sparse_model_name: str = "prithvida/Splade_PP_en_v1",
        in_memory: bool = True
    ):
        # Initialize Qdrant client
        self.client = QdrantClient(":memory:" if in_memory else None)
        self.collection_name = collection_name
        
        # Initialize embedding models
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
        # Create collection with hybrid schema
        self._create_collection()

    def _create_collection(self):
        """Create Qdrant collection with hybrid schema"""
        self.client.create_collection(
            self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=1024,  # Size for BGE embeddings
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )

    def index_documents(self, documents: List[Dict[str, str]]):
        """Index documents with both dense and sparse embeddings"""
        points = []
        
        # Process documents in batches
        for doc in documents:
            text = doc["content"]
            
            # Generate embeddings
            dense_vector = next(self.dense_model.embed([text]))
            sparse_embedding = next(self.sparse_model.embed([text]))
            
            # Create sparse vector
            sparse_vector = SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
            
            # Create point
            point = PointStruct(
                id=doc["id"],
                payload=doc,
                vector={
                    "dense": dense_vector.tolist(),
                    "sparse": sparse_vector,
                },
            )
            points.append(point)
        
        # Upload points to Qdrant
        self.client.upsert(self.collection_name, points)

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Perform hybrid search using both dense and sparse embeddings"""
        # Generate query embeddings
        dense_vector = next(self.dense_model.embed([query]))
        sparse_embedding = next(self.sparse_model.embed([query]))
        
        # Create search requests
        search_results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(
                    vector=NamedVector(
                        name="dense",
                        vector=dense_vector.tolist(),
                    ),
                    limit=limit,
                    with_payload=True,
                ),
                SearchRequest(
                    vector=NamedSparseVector(
                        name="sparse",
                        vector=SparseVector(
                            indices=sparse_embedding.indices.tolist(),
                            values=sparse_embedding.values.tolist(),
                        ),
                    ),
                    limit=limit,
                    with_payload=True,
                ),
            ],
        )
        
        # Combine results using RRF
        dense_results = [(hit.id, hit.score, hit.payload) for hit in search_results[0]]
        sparse_results = [(hit.id, hit.score, hit.payload) for hit in search_results[1]]
        
        combined_results = self._reciprocal_rank_fusion(
            [dense_results, sparse_results]
        )
        
        return [
            SearchResult(id=id_, score=score, payload=payload)
            for id_, score, payload in combined_results
        ]

    def _reciprocal_rank_fusion(
        self,
        rank_lists: List[List[Tuple[str, float, Dict[str, Any]]]],
        k: float = 60.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Combine multiple ranking lists using Reciprocal Rank Fusion"""
        # Create a map of document ID to all its scores
        doc_scores = {}
        
        for rank_list in rank_lists:
            for rank, (doc_id, score, payload) in enumerate(rank_list, 1):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"score": 0.0, "payload": payload}
                doc_scores[doc_id]["score"] += 1.0 / (k + rank)
        
        # Sort by combined score
        sorted_results = sorted(
            [
                (doc_id, data["score"], data["payload"])
                for doc_id, data in doc_scores.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_results
