import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "docs/"
DB_DIR = "faiss_index/"

def ingest_data():
    """
        This function is designed to process text files and create 
        a searchable vector database
    """
    loaders = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            loaders.append(TextLoader(file_path)) #a utility handler for reading and processing text files

    docs = []
    for loader in loaders:
        docs.extend(loader.load())
        logger.info(f"Loaded {len(docs)} documents from {loader.file_path}")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(docs)
    embedding = OllamaEmbeddings(model="llama3")
    db = FAISS.from_documents(texts, embedding)
    db.save_local(DB_DIR)
    logger.info(f"Saved FAISS index to {DB_DIR}")

def load_vectorstore():
    """
        This function loads the vectorstore from the specified directory 
    """
    embedding= OllamaEmbeddings(model="llama3")
    return FAISS.load_local(DB_DIR, embedding)
