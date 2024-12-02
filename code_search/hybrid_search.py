from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import os
import atexit
from pathlib import Path
import hashlib
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
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

def hash_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer):
        self.indexer = indexer

    def on_modified(self, event):
        if not event.is_directory:
            self.indexer.process_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.indexer.process_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.indexer.remove_file(event.src_path)

class HybridSearch:
    def __init__(
        self,
        collection_name: str,
        dense_model_name: str = "BAAI/bge-large-en-v1.5",
        sparse_model_name: str = "prithvida/Splade_PP_en_v1",
        in_memory: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        watch: bool = False,
        supported_extensions: set = {'.py', '.js', '.java', '.cpp', '.h', '.cs', '.go'}
    ):
        # Initialize Qdrant client
        self.client = QdrantClient(":memory:" if in_memory else None)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions
        self.hash_cache = {}
        
        # Initialize embedding models
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
        # Create collection with hybrid schema
        self._create_collection()

        if watch:
            self._start_watching()
            atexit.register(self._stop_watching)

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

    def _start_watching(self):
        """Start watching for file changes"""
        self.observer = Observer()
        event_handler = CodeChangeHandler(self)
        self.observer.schedule(event_handler, ".", recursive=True)
        self.observer.start()

    def _stop_watching(self):
        """Stop watching for file changes"""
        if hasattr(self, "observer"):
            self.observer.stop()
            self.observer.join()

    def _split_code(self, text: str) -> List[str]:
        """Split code into chunks with overlap"""
        if not text.strip():
            return []
            
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                # Keep last few lines for overlap
                overlap_lines = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_lines
                current_size = sum(len(line) for line in overlap_lines)
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def process_file(self, file_path: str):
        """Process a single file"""
        # Check if file extension is supported
        if Path(file_path).suffix not in self.supported_extensions:
            return

        # Check if file has changed
        current_hash = hash_md5(file_path)
        if file_path in self.hash_cache and self.hash_cache[file_path] == current_hash:
            return
        
        self.hash_cache[file_path] = current_hash
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove existing documents for this file
            self.remove_file(file_path)
            
            # Split into chunks and index
            chunks = self._split_code(content)
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "id": f"{file_path}_{i}",
                    "content": chunk,
                    "file_path": file_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                documents.append(doc)
            
            if documents:
                self.index_documents(documents)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    def remove_file(self, file_path: str):
        """Remove all chunks for a given file"""
        # Delete points where file_path matches
        self.client.delete(
            collection_name=self.collection_name,
            filter={
                "must": [
                    {
                        "key": "file_path",
                        "match": {"value": file_path}
                    }
                ]
            }
        )
        if file_path in self.hash_cache:
            del self.hash_cache[file_path]

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

    def index_directory(self, directory: str):
        """Index all supported files in a directory"""
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                self.process_file(file_path)

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
