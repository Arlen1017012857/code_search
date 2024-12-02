from setuptools import setup, find_packages

setup(
    name="code_search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "chromadb>=0.4.0",
        "qdrant-client>=1.6.0",
        "fastembed>=0.1.0",
        "langchain>=0.0.300",
        "llama-index>=0.8.0",
        "transformers>=4.30.0",
        "tree-sitter>=0.20.0",
        "watchdog>=3.0.0",
        "python-git>=2023.1.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful code search and indexing system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/code_search",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
