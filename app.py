import streamlit as st
from rag_engine import agent_pipeline, ingest_data
import time
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Knowledge Explorer | RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1v0mbdj.e115fcil1 {
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        font-size: 16px;
        padding: 12px 15px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
    }
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
    }
    .doc-container {
        border-left: 4px solid #4361ee;
        padding-left: 15px;
        margin: 10px 0;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    .metadata {
        font-size: 12px;
        color: #666;
        margin-bottom: 5px;
    }
    .tool-badge {
        background-color: #4cc9f0;
        padding: 5px 10px;
        border-radius: 15px;
        color: white;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    .answer-container {
        background-color: #e6f7e6;
        border-radius: 8px;
        padding: 20px;
        color : #111;
        border-left: 5px solid #28a745;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-text {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Sidebar content
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.markdown("---")
    
    st.subheader("About")
    st.write("""
    This Retrieval-Augmented Generation (RAG) assistant helps you get accurate answers 
    from your knowledge base by combining document retrieval with generative AI.
    """)
    
    st.markdown("---")
    
    st.subheader("How it works")
    st.markdown("""
    1. 🔍 **Query Analysis**: Your question is analyzed to determine the best approach
    2. 📚 **Document Retrieval**: Relevant documents are fetched from the knowledge base
    3. 🧠 **Response Generation**: AI generates an answer based on retrieved context
    """)
    
    st.markdown("---")
    
    # Sample questions to help users get started
    st.subheader("Sample Questions")
    sample_questions = [
        "calculate 2+5",
        "What are whales?",
        "How long can whales live?",
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}"):
            st.session_state.query = q
            st.rerun()

# Main content
st.markdown("""
<div class="header-container">
    <div class="header-text">
        <h1>Knowledge Explorer</h1>
        <p>Get smart answers from your organization's knowledge base</p>
    </div>
</div>
""", unsafe_allow_html=True)

if 'query' not in st.session_state:
    st.session_state.query = ""

query = st.text_input("What would you like to know?", value=st.session_state.query, key="query_input")

# Search button
col1, col2 = st.columns([6, 1])
with col2:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

# Process query
if query and (search_clicked or st.session_state.query == query):
    st.session_state.query = query 
    
    with st.spinner():
        lottie_search = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qwgbfrz1.json")
        if lottie_search:
            search_animation = st_lottie(lottie_search, height=150, key="search_animation")
        
        # Get response from RAG pipeline
        start_time = time.time()
        tool, docs, answer = agent_pipeline(query)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = round(end_time - start_time, 2)
    
    # Display metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Processing Time", f"{processing_time} sec")
    with metrics_col2:
        st.metric("Documents Retrieved", len(docs) if docs else 0)
    with metrics_col3:
        st.metric("Tool Used", tool)
    
    st.markdown("### 💡 Answer")
    st.markdown(f"""
    <div class="answer-container">
        {answer}
    </div>
    """, unsafe_allow_html=True)
    if docs:
        with st.expander("📚 View Retrieved Context", expanded=False):
            tabs = st.tabs([f"Document {i+1}" for i in range(len(docs))])
            
            for i, (tab, doc) in enumerate(zip(tabs, docs)):
                with tab:
                    st.markdown(f"""
                    <div class="doc-container">
                        <div class="metadata">
                            <strong>Source:</strong> {getattr(doc, 'metadata', {}).get('source', 'Unknown')}
                            &nbsp;|&nbsp;
                            <strong>Relevance Score:</strong> {getattr(doc, 'metadata', {}).get('score', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display document content with syntax highlighting
                    st.code(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""), language="markdown")
    
    # Add a horizontal rule to separate multiple queries
    st.markdown("---")
    
    # Keep track of search history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Add current query to history
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'query': query,
        'tool': tool,
        'docs': len(docs) if docs else 0
    })
    
    # Show search history
    if len(st.session_state.history) > 1:
        with st.expander("📜 Search History", expanded=False):
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
            
            # Create a simple visualization of tools used
            if len(st.session_state.history) >= 3:
                tool_counts = history_df['tool'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=tool_counts.index, 
                    values=tool_counts.values,
                    hole=.3,
                    marker_colors=['#4361ee', '#3a0ca3', '#4cc9f0', '#f72585']
                )])
                
                fig.update_layout(
                    title_text="Tool Usage Distribution",
                    showlegend=True,
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)

else:
    # Show welcome animation when no query is entered
    welcome_lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ktwnwv5m.json")
    if welcome_lottie:
        st_lottie(welcome_lottie, height=300, key="welcome_animation")
    
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <h3>Ask a question to get started</h3>
        <p>Type your query above or select a sample question from the sidebar</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #666; font-size: 12px;">
    Powered by RAG Engine • Last updated: May 2025
</div>
""", unsafe_allow_html=True)