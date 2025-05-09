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
5. Run data ingestion command after adding all the txt files in `/docs` in a different terminal window
   ```
   python -c "import rag_engine; rag_engine.ingest_data()"
   ```
Now you can run CLI version or WebAPP version
```
py cli.py
```
```
streamlit run app.py
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

## Files
- `rag_engine.py`: Core retrieval and generation logic
- `app.py`: Streamlit web interface
- `cli.py`: Command line interface

## Requirements
- Python 3.8+
- Ollama with llama3 model
- See requirements.txt for Python packages

For  AI Engineering internship, Inflera Technologies Pty Limited  
