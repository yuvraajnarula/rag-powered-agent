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

def get_rag_answer(query, retriever, llm):
    """
        This function retrieves relevant documents from the vectorstore
        and generates an answer using the LLM
        Args:
            query (str): The question to be answered.
            retriever: The vectorstore retriever.
            llm: The language model to generate the answer.
    """
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    logger.info(f"Retrieved {len(docs)} documents for query: {query}")
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
    answer = llm.invoke(prompt)
    return docs, answer

def agent_pipeline(query):
    """
        This function initializes the agent pipeline with the LLM and tools
        and decides which tool to use based on the query.
        Args:
            query (str): The question to be answered.
    """
    llm = Ollama(model="llama3")
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
    # Initialize the agent with the LLM and tools
    def calc_tool(q):
        try:
            result = eval(q.split(" ")[-1])
            return f"The result of the calculation is: {result}"
        except Exception as e:
            logger.error(f"Error in calc_tool: {e}")
            return "Error in calc_tool"
    
    tools =[
        Tool(
            name = "Calculator",
            func=calc_tool,
            description="A simple calculator that can perform basic arithmetic operations. Example: 2 + 2"
        )
    ]
    if "calculate" in query.lower():
        logger.info("Agent decision: Calculator Tool")
        return "Calculator Tool", [], calc_tool(query)
    else:
        logger.info("Agent decision: RAG Pipeline")
        docs, answer = get_rag_answer(query, retriever, llm)
        return "RAG Pipeline", docs, answer