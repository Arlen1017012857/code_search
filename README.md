# Code Search Engine

A powerful code search and indexing system that combines multiple advanced techniques for efficient code discovery and understanding.

## Features

- Hybrid search combining dense and sparse embeddings
- Merkle tree-based code change tracking
- Real-time file watching and index updates
- Git integration for metadata enrichment
- Support for multiple programming languages
- Efficient ranking using Reciprocal Rank Fusion (RRF)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from code_search.code_indexer import CodeIndexer

# Initialize indexer with repository path
indexer = CodeIndexer(
    repo_path="/path/to/your/repo",
    collection_name="my_code_search",
    watch=True  # Enable real-time file watching
)

# Search code
results = indexer.search("implement binary search tree")
for result in results:
    print(f"File: {result.file_path}")
    print(f"Language: {result.language}")
    print(f"Content: {result.content[:200]}...")  # Show first 200 chars
    print("---")

# Get changed files
changed_files = indexer.get_changed_files()
print(f"Changed files: {changed_files}")

# Stop indexer (important for cleanup)
indexer.stop()
```

## Architecture

The system consists of three main components:

1. **Code Indexer**: Main component that orchestrates the indexing and search process
2. **Merkle Tree**: Tracks code changes efficiently
3. **Hybrid Search**: Combines dense and sparse embeddings for better search results

## Supported Languages

- Python (.py)
- JavaScript (.js)
- Java (.java)
- C++ (.cpp, .h)
- C# (.cs)
- Go (.go)

More languages can be added by extending the `supported_extensions` list.

## License

MIT License
