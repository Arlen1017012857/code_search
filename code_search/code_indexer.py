import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import git
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .merkle_tree import MerkleTree
from .hybrid_search import HybridSearch

@dataclass
class CodeDocument:
    id: str
    content: str
    file_path: str
    language: str
    metadata: Dict[str, any]

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer: 'CodeIndexer'):
        self.indexer = indexer

    def on_modified(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            self.indexer.update_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            self.indexer.add_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            self.indexer.remove_file(event.src_path)

    def _is_supported_file(self, file_path: str) -> bool:
        return any(file_path.endswith(ext) for ext in self.indexer.supported_extensions)

class CodeIndexer:
    def __init__(
        self,
        repo_path: str,
        collection_name: str = "code_search",
        watch: bool = True,
        supported_extensions: Optional[List[str]] = None
    ):
        self.repo_path = repo_path
        self.collection_name = collection_name
        self.supported_extensions = supported_extensions or [
            '.py', '.js', '.java', '.cpp', '.h', '.cs', '.go'
        ]
        
        # Initialize components
        self.merkle_tree = MerkleTree(repo_path)
        self.search_engine = HybridSearch(collection_name)
        
        # Set up Git repository tracking
        self.repo = git.Repo(repo_path)
        
        # Initialize file watcher if requested
        if watch:
            self._setup_file_watcher()
        
        # Perform initial indexing
        self.index_repository()

    def _setup_file_watcher(self):
        """Set up file system watcher for real-time updates"""
        self.observer = Observer()
        event_handler = CodeChangeHandler(self)
        self.observer.schedule(event_handler, self.repo_path, recursive=True)
        self.observer.start()

    def index_repository(self):
        """Index entire repository"""
        documents = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.supported_extensions):
                    file_path = os.path.join(root, file)
                    doc = self._create_document(file_path)
                    if doc:
                        documents.append(doc)
        
        self.search_engine.index_documents(documents)

    def update_file(self, file_path: str):
        """Update index for modified file"""
        doc = self._create_document(file_path)
        if doc:
            self.search_engine.index_documents([doc])

    def add_file(self, file_path: str):
        """Add new file to index"""
        self.update_file(file_path)

    def remove_file(self, file_path: str):
        """Remove file from index"""
        # Note: Implementation depends on search engine's delete capability
        pass

    def _create_document(self, file_path: str) -> Optional[CodeDocument]:
        """Create document from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            relative_path = os.path.relpath(file_path, self.repo_path)
            language = os.path.splitext(file_path)[1][1:]  # Remove leading dot
            
            # Get Git metadata
            git_metadata = self._get_git_metadata(relative_path)
            
            return CodeDocument(
                id=relative_path,
                content=content,
                file_path=relative_path,
                language=language,
                metadata={
                    "size": os.path.getsize(file_path),
                    "last_modified": os.path.getmtime(file_path),
                    **git_metadata
                }
            ).__dict__
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None

    def _get_git_metadata(self, relative_path: str) -> Dict[str, any]:
        """Get Git-related metadata for file"""
        try:
            # Get last commit for file
            commits = list(self.repo.iter_commits(paths=relative_path, max_count=1))
            if commits:
                last_commit = commits[0]
                return {
                    "last_commit_hash": last_commit.hexsha,
                    "last_commit_author": last_commit.author.name,
                    "last_commit_date": last_commit.committed_datetime.isoformat(),
                    "last_commit_message": last_commit.message.strip()
                }
        except Exception as e:
            print(f"Error getting git metadata for {relative_path}: {str(e)}")
        
        return {}

    def search(self, query: str, limit: int = 10) -> List[CodeDocument]:
        """Search code repository"""
        results = self.search_engine.search(query, limit=limit)
        return [CodeDocument(**result.payload) for result in results]

    def get_changed_files(self) -> List[str]:
        """Get list of changed files since last indexing"""
        # Create new Merkle tree from current state
        current_tree = MerkleTree(self.repo_path)
        return self.merkle_tree.get_changed_files(current_tree)

    def stop(self):
        """Stop file watcher and cleanup resources"""
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
