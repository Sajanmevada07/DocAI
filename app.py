import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import time
import os
import logging
from transformers.utils import logging as transformers_logging
import transformers

# Suppress unnecessary warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers_logging.set_verbosity_error()
logging.getLogger("langchain").setLevel(logging.ERROR)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/bert-base-cased-squad2"
CHUNK_SIZE = 800  # Reduced for CPU efficiency
CHUNK_OVERLAP = 100  # Reduced overlap
MAX_CONTEXT_LENGTH = 384  # Reduced for CPU
MAX_ANSWER_WORDS = 150


def limit_answer_length(answer, max_words=MAX_ANSWER_WORDS):
    """Truncate answer to a maximum number of words while preserving meaning"""
    if not isinstance(answer, str) or not answer.strip():
        return answer
    
    words = answer.split()
    if len(words) <= max_words:
        return answer
    
    # Find a natural truncation point near the word limit
    truncated = words[:max_words]
    # Add ellipsis only if we're truncating mid-sentence
    return ' '.join(truncated) + ('...' if not answer.endswith('.') else '')



@st.cache_data(show_spinner=False)
def extract_text(pdf_files):
    """Extract text from PDF files with error handling"""
    all_text = ""
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {str(e)}")
    return all_text


@st.cache_data(show_spinner=False)
def split_text(text):
    """Split text into chunks with validation"""
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)



@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load embedding model with CPU enforcement"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource(show_spinner=False)
def create_vector_store(_embeddings, text_chunks):
    """Create vector store with cache"""
    if not text_chunks:
        return None
    return FAISS.from_texts(text_chunks, _embeddings)

def load_qa_model():
    """Load Hugging Face QA model optimized for CPU"""
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    
    qa_pipeline = pipeline( 
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # Force CPU usage
        max_length=MAX_CONTEXT_LENGTH
        )
    return HuggingFacePipeline(pipeline=qa_pipeline)

def get_answer(vector_store, qa_pipeline, question):
    """Retrieve context and generate answer"""
    # Retrieve relevant chunks
    docs = vector_store.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])
    
    # Generate answer
    response = qa_pipeline.invoke(
        input={"question": question, "context": context}
    )
    return response["answer"]

# Streamlit UI
st.set_page_config(page_title="Multi-PDF Analyzer", layout="wide")
st.title("📄 Docai")
st.subheader("Upload 1-5 PDF files for AI-powered Q&A")

# File upload
uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Maximum 5 PDFs allowed. Using first 5 files.")
        uploaded_files = uploaded_files[:5]
    
    # Process PDFs
    with st.spinner("Analyzing documents..."):
        # Extract and split text
        raw_text = extract_text(uploaded_files)
        text_chunks = split_text(raw_text)
        
        # Create vector store
        vector_store = create_vector_store(text_chunks)
        
        # Load QA model
        qa_pipeline = load_qa_model()
    
    st.success(f"Processed {len(uploaded_files)} PDF(s) with {len(text_chunks)} text chunks")
    
    # Q&A interface
    question = st.text_input("Ask about your documents:", placeholder="Enter your question here...")
    
    if question:
        with st.spinner("Searching for answers..."):
            try:
                answer = get_answer(vector_store, qa_pipeline, question)
                st.subheader("Answer:")
                st.info(answer)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
else:
    st.info("Please upload PDF documents to get started")