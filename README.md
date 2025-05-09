# RAG Assistant

A simple Retrieval-Augmented Generation (RAG) assistant that answers questions based on your knowledge base.

## Overview

This tool combines document retrieval with AI to provide accurate answers from your documents. It features both a web interface and command line interface.

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Ollama and download the llama3 model:
   ```
   ollama pull llama3
   ```
4. Create directories:
   ```
   mkdir -p docs faiss_index
   ```

## Usage

### Web Interface
```
streamlit run app.py
```

### Command Line
```
python cli.py
```

## Adding Knowledge
1. Add text files to the `docs/` folder
2. Run ingestion:
   ```python
   from rag_engine import ingest_data
   ingest_data()
   ```

## Files
- `rag_engine.py`: Core retrieval and generation logic
- `app.py`: Streamlit web interface
- `cli.py`: Command line interface

## Requirements
- Python 3.8+
- Ollama with llama3 model
- See requirements.txt for Python packages

For  AI Engineering internship, Inflera Technologies Pty Limited  
