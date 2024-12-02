import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

@dataclass
class MerkleNode:
    hash: str
    path: str
    children: Dict[str, 'MerkleNode']
    is_file: bool
    content_hash: Optional[str] = None

class MerkleTree:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.root = self._build_tree(root_path)
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file's contents"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _build_tree(self, path: str) -> MerkleNode:
        """Recursively build Merkle tree from directory structure"""
        if os.path.isfile(path):
            content_hash = self._calculate_file_hash(path)
            return MerkleNode(
                hash=content_hash,
                path=path,
                children={},
                is_file=True,
                content_hash=content_hash
            )
        
        children = {}
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            children[item] = self._build_tree(item_path)
        
        # Calculate directory hash from children's hashes
        dir_hash = hashlib.sha256()
        for name, child in sorted(children.items()):
            dir_hash.update(f"{name}:{child.hash}".encode())
        
        return MerkleNode(
            hash=dir_hash.hexdigest(),
            path=path,
            children=children,
            is_file=False
        )

    def get_changed_files(self, other_tree: 'MerkleTree') -> List[str]:
        """Compare with another Merkle tree and return list of changed files"""
        changed_files = []
        self._compare_nodes(self.root, other_tree.root, changed_files)
        return changed_files

    def _compare_nodes(self, node1: MerkleNode, node2: MerkleNode, changed_files: List[str]):
        """Recursively compare two nodes and collect changed files"""
        if node1.is_file and node2.is_file:
            if node1.hash != node2.hash:
                changed_files.append(node1.path)
            return

        # Compare directory contents
        all_children = set(node1.children.keys()) | set(node2.children.keys())
        for child_name in all_children:
            child1 = node1.children.get(child_name)
            child2 = node2.children.get(child_name)
            
            if child1 is None:  # File/directory was deleted
                if child2.is_file:
                    changed_files.append(child2.path)
            elif child2 is None:  # File/directory was added
                if child1.is_file:
                    changed_files.append(child1.path)
            else:  # Both exist, compare them
                self._compare_nodes(child1, child2, changed_files)

    def update(self):
        """Update the Merkle tree with current filesystem state"""
        self.root = self._build_tree(self.root_path)
