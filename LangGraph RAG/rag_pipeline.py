import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated
import time
import requests
from urllib.parse import urlparse

# Core LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate

# Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ChromaDB integration
from langchain_chroma import Chroma

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader,
    WebBaseLoader,
    DirectoryLoader,
    TextLoader
)

# PowerPoint loader
from pptx import Presentation
import docx
import PyPDF2
import pdfplumber
import bs4

class RAGState(TypedDict):
    """State for RAG pipeline"""
    question: str
    context: List[Document]
    answer: str
    sources: List[str]

class DocumentProcessor:
    """Handles different document types"""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load PDF documents"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            st.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_docx(file_path: str) -> List[Document]:
        """Load Word documents"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            st.error(f"Error loading DOCX {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_pptx(file_path: str) -> List[Document]:
        """Load PowerPoint presentations"""
        try:
            presentation = Presentation(file_path)
            text_content = []
            
            for slide_num, slide in enumerate(presentation.slides):
                slide_text = f"Slide {slide_num + 1}:\n"
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text += shape.text + "\n"
                text_content.append(slide_text)
            
            full_text = "\n\n".join(text_content)
            doc = Document(
                page_content=full_text,
                metadata={"source": file_path, "type": "pptx"}
            )
            return [doc]
        except Exception as e:
            st.error(f"Error loading PPTX {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_txt(file_path: str) -> List[Document]:
        """Load text files"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            st.error(f"Error loading TXT {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_web_url(url: str) -> List[Document]:
        """Load content from web URL"""
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section"]
                    )
                )
            )
            documents = loader.load()
            return documents
        except Exception as e:
            st.error(f"Error loading URL {url}: {str(e)}")
            return []
    
    @staticmethod
    def load_directory(dir_path: str) -> List[Document]:
        """Load all supported files from directory"""
        documents = []
        supported_extensions = ['.pdf', '.docx', '.pptx', '.txt']
        
        for file_path in Path(dir_path).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                file_str = str(file_path)
                if file_path.suffix.lower() == '.pdf':
                    documents.extend(DocumentProcessor.load_pdf(file_str))
                elif file_path.suffix.lower() == '.docx':
                    documents.extend(DocumentProcessor.load_docx(file_str))
                elif file_path.suffix.lower() == '.pptx':
                    documents.extend(DocumentProcessor.load_pptx(file_str))
                elif file_path.suffix.lower() == '.txt':
                    documents.extend(DocumentProcessor.load_txt(file_str))
        
        return documents

class RAGPipeline:
    """Complete RAG Pipeline with LangGraph"""
    
    def __init__(self, google_api_key: str, persist_directory: str = "./chroma_db"):
        """Initialize RAG pipeline"""
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )
        
        # Initialize Gemini embeddings using a supported model for the current API
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key
        )
        
        # Initialize ChromaDB vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        # Create RAG prompt
        self.rag_prompt = PromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question}
        Context: {context}
        Answer:
        """)
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph for RAG pipeline"""
        
        def retrieve(state: RAGState):
            """Retrieve relevant documents"""
            retrieved_docs = self.vector_store.similarity_search(
                state["question"], 
                k=4
            )
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
            return {
                "context": retrieved_docs,
                "sources": sources
            }
        
        def generate(state: RAGState):
            """Generate answer using retrieved context"""
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.rag_prompt.invoke({
                "question": state["question"], 
                "context": docs_content
            })
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        
        # Create graph
        graph_builder = StateGraph(RAGState).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to vector store"""
        if not documents:
            return 0
            
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(splits)
        
        return len(splits)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        result = self.graph.invoke({"question": question})
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "context": result.get("context", [])
        }
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {
                "document_count": count,
                "collection_name": collection.name if hasattr(collection, 'name') else "default"
            }
        except:
            return {"document_count": 0, "collection_name": "default"}


# ======================================================================
# New UI — modern dark glassmorphism design system
# ======================================================================
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-0: #0a0a14;
    --bg-1: #101024;
    --card: rgba(255, 255, 255, 0.05);
    --card-border: rgba(255, 255, 255, 0.10);
    --accent: #7c5cff;
    --accent-2: #22d3ee;
    --text: #eef0ff;
    --muted: #9aa0b8;
}

/* ---------- Global ---------- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(124, 92, 255, 0.25), transparent 60%),
        radial-gradient(1000px 600px at 110% 10%, rgba(34, 211, 238, 0.15), transparent 55%),
        linear-gradient(160deg, var(--bg-0), var(--bg-1));
    color: var(--text);
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Hero banner ---------- */
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 1.4rem;
    background: linear-gradient(120deg, rgba(124,92,255,.28), rgba(34,211,238,.16));
    border: 1px solid var(--card-border);
    backdrop-filter: blur(12px);
    margin-bottom: 1.6rem;
}
.hero h1 {
    margin: 0;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #ffffff, #b9a8ff 55%, #7ee7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { margin: .5rem 0 0; color: var(--muted); font-size: 1.02rem; }

/* ---------- Glass cards ---------- */
.glass-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 1.2rem;
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,.35);
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: .55rem;
    margin-bottom: .35rem;
}
.section-sub { color: var(--muted); font-size: .88rem; margin-bottom: .5rem; }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 1rem;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricValue"] { color: var(--accent-2); font-weight: 700; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--muted) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: rgba(12, 12, 26, 0.85);
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] * { color: var(--text); }

/* ---------- Inputs ---------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: .7rem !important;
    color: var(--text) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,92,255,.25) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color: var(--muted); }
label, .stMarkdown p, li { color: var(--text); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(120deg, var(--accent), #5b8cff);
    color: #fff;
    border: none;
    border-radius: .8rem;
    padding: .65rem 1.4rem;
    font-weight: 600;
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
    box-shadow: 0 6px 18px rgba(124,92,255,.35);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.12);
    box-shadow: 0 10px 24px rgba(124,92,255,.45);
    border: none;
    color: #fff;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.04);
    border: 1.5px dashed rgba(124,92,255,.55) !important;
    border-radius: 1rem;
}
[data-testid="stFileUploaderDropzone"] span { color: var(--muted); }

/* ---------- Expanders ---------- */
.streamlit-expanderHeader, [data-testid="stExpander"] details {
    background: rgba(255,255,255,.04);
    border: 1px solid var(--card-border) !important;
    border-radius: .8rem !important;
}
[data-testid="stExpander"] summary p { color: var(--text); }

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    background: rgba(255,255,255,.05);
    border: 1px solid var(--card-border);
    border-radius: .8rem;
    color: var(--text);
}

/* ---------- Dividers & tabs ---------- */
hr { border-color: var(--card-border); }
.stTabs [data-baseweb="tab-list"] { gap: .5rem; background: transparent; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.05);
    border: 1px solid var(--card-border);
    border-radius: .7rem;
    color: var(--muted);
    padding: .45rem 1.1rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, var(--accent), #5b8cff) !important;
    color: #fff !important;
    border-color: transparent;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: .85rem;
    padding: 1.4rem 0 .4rem;
}
</style>
"""


def _card(title: str, subtitle: str) -> None:
    """Render a glassmorphism section header card."""
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">{title}</div>
            <div class="section-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Streamlit UI — modern dark glassmorphism theme"""
    st.set_page_config(
        page_title="RAG Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject the new design system
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Hero banner
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
            <h1>🤖 RAG Pipeline Studio</h1>
            <p>LangGraph · Gemini · ChromaDB — ingest your documents, then chat with your knowledge base.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Sidebar — configuration & stats
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Google API Key input
        google_api_key = st.text_input(
            "🔑 Google API Key",
            type="password",
            help="Enter your Google API key for Gemini"
        )

        if not google_api_key:
            st.warning("Please enter your Google API key to continue.")
            st.stop()

        st.markdown("---")

        # Initialize RAG pipeline (re-initialize if key changes)
        if ("rag_pipeline" not in st.session_state
                or st.session_state.get("api_key") != google_api_key):
            with st.spinner("Initializing RAG pipeline..."):
                st.session_state.rag_pipeline = RAGPipeline(google_api_key)
                st.session_state.api_key = google_api_key

        # Vector store statistics
        st.markdown("### 📊 Vector Store")
        stats = st.session_state.rag_pipeline.get_vector_store_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats["document_count"])
        with col2:
            st.metric("Collection", stats["collection_name"])

    # ------------------------------------------------------------------
    # Main interface — tabbed Ingest / Query layout
    # ------------------------------------------------------------------
    tab_ingest, tab_query = st.tabs(["📥 Ingest", "💬 Query"])

    # ---------------- Ingest tab ----------------
    with tab_ingest:
        col1, col2 = st.columns([1, 1])

        with col1:
            _card("📄 Document Upload", "PDF · DOCX · PPTX · TXT — multiple files supported")
            uploaded_files = st.file_uploader(
                "Drop files here",
                type=["pdf", "docx", "pptx", "txt"],
                accept_multiple_files=True,
                help="Upload PDF, Word, PowerPoint, or text files"
            )

        with col2:
            _card("🌐 Web Source", "Scrape and index content from any public URL")
            web_url = st.text_input(
                "Enter web URL to scrape:",
                placeholder="https://example.com/article"
            )

        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files or web_url:
                with st.spinner("Processing documents..."):
                    all_documents = []

                    # Process uploaded files
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name

                            # Process based on file type
                            file_extension = uploaded_file.name.split('.')[-1].lower()

                            if file_extension == 'pdf':
                                docs = DocumentProcessor.load_pdf(tmp_path)
                            elif file_extension == 'docx':
                                docs = DocumentProcessor.load_docx(tmp_path)
                            elif file_extension == 'pptx':
                                docs = DocumentProcessor.load_pptx(tmp_path)
                            elif file_extension == 'txt':
                                docs = DocumentProcessor.load_txt(tmp_path)
                            else:
                                docs = []

                            # Update metadata
                            for doc in docs:
                                doc.metadata["source"] = uploaded_file.name
                                doc.metadata["type"] = file_extension

                            all_documents.extend(docs)

                            # Clean up temp file
                            os.unlink(tmp_path)

                    # Process web URL
                    if web_url:
                        web_docs = DocumentProcessor.load_web_url(web_url)
                        all_documents.extend(web_docs)

                    # Add documents to vector store
                    if all_documents:
                        num_chunks = st.session_state.rag_pipeline.add_documents(all_documents)
                        st.success(f"✅ Processed {len(all_documents)} documents into {num_chunks} chunks")

                        # Show document details
                        _card("📋 Processed Documents", "Preview of the indexed content")
                        for i, doc in enumerate(all_documents[:5]):  # Show first 5
                            with st.expander(f"Document {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                                st.write(f"**Type:** {doc.metadata.get('type', 'Unknown')}")
                                st.write("**Content Preview:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

                        if len(all_documents) > 5:
                            st.info(f"... and {len(all_documents) - 5} more documents")

                        # Update stats
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("❌ No documents were processed successfully")
            else:
                st.warning("Please upload files or enter a web URL")

    # ---------------- Query tab ----------------
    with tab_query:
        _card("💬 Ask Your Knowledge Base", "Answers are generated by Gemini using retrieved context")

        # Query input
        user_question = st.text_area(
            "Ask a question about your documents:",
            height=120,
            placeholder="What is the main topic discussed in the documents?"
        )

        # Query button
        if st.button("🔍 Search", type="primary"):
            if user_question and st.session_state.rag_pipeline.get_vector_store_stats()["document_count"] > 0:
                with st.spinner("Searching for answer..."):
                    result = st.session_state.rag_pipeline.query(user_question)

                    # Persist to history
                    if "query_history" not in st.session_state:
                        st.session_state.query_history = []
                    st.session_state.query_history.append((user_question, result))

                    # Display results
                    _card("🎯 Answer", f"Q: {user_question}")
                    st.markdown(result["answer"])

                    # Display sources
                    if result["sources"]:
                        _card("📚 Sources", "Documents used to build this answer")
                        for source in result["sources"]:
                            st.markdown(f"&nbsp;&nbsp;• `{source}`")

                    # Display retrieved context
                    if result["context"]:
                        _card("📄 Retrieved Context", "Chunk-level matches from the vector store")
                        for i, doc in enumerate(result["context"]):
                            with st.expander(f"Context {i+1} from {doc.metadata.get('source', 'Unknown')}"):
                                st.write(doc.page_content)

            elif not user_question:
                st.warning("Please enter a question")
            else:
                st.warning("Please upload and process documents first")

        # Query history
        if "query_history" not in st.session_state:
            st.session_state.query_history = []

        if st.session_state.query_history:
            _card("📜 Recent Queries", "Your last three questions and answers")
            for i, (question, result) in enumerate(st.session_state.query_history[-3:]):
                with st.expander(f"Q: {question[:50]}..."):
                    st.markdown(f"**A:** {result['answer']}")
                    if result.get("sources"):
                        st.caption("Sources: " + ", ".join(result["sources"]))

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>🤖 RAG Pipeline Studio — powered by LangGraph, Gemini &amp; ChromaDB</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
