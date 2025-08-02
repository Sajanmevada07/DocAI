import streamlit as st
import pdfplumber
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/bert-base-cased-squad2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTEXT_LENGTH = 512


def extract_text(pdf_files):
    """Extract text from multiple PDFs"""
    all_text = ""
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() + "\n"
    return all_text

def split_text(text):
    """Split text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)


def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_texts(text_chunks, embeddings)

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