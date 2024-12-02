"""
Code Search Engine
~~~~~~~~~~~~~~~~~

A powerful code search and indexing system that combines multiple advanced techniques
for efficient code discovery and understanding.

:copyright: (c) 2024
:license: MIT
"""

from .code_indexer import CodeIndexer
from .hybrid_search import HybridSearch
from .merkle_tree import MerkleTree

__version__ = "0.1.0"
__all__ = ["CodeIndexer", "HybridSearch", "MerkleTree"]
