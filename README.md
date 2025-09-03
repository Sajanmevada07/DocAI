DocAI : Advanced Document Analyzer with RAG

Features:-

Multi-PDF Processing: Analyze 1-5 PDF documents simultaneously
RAG Architecture: Combines retrieval and generation for accurate answers
Hardware Flexibility: Runs on both CPU and GPU environments
Streamlit UI: User-friendly web interface
Open-Source Models: Uses Hugging Face transformers (no OpenAI required)
Document Source Tracking: Shows which parts of documents were used for answers

Models:-

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (80MB, CPU-optimized)
QA Model: deepset/bert-base-cased-squad2 (438MB, fine-tuned on SQuAD 2.0)
Alternative Options: Configurable to use other Hugging Face models

Installation:
1. Clone Repository :
  git clone <repository-url>
  cd docai_project

2. Create a virtual environment (recommended):
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3.Install dependencies:
  pip install -r requirements.txt

4.Start the application:
  streamlit run app.py

  1.Open your browser and navigate to the local URL shown in the terminal (typically http://localhost:8501)
  2.Upload 1-5 PDF documents using the file uploader
  3.Ask questions about the content of your documents in the text input
  4.View answers with source context in the expandable section

How It Works

Document Processing:
PDF text extraction using pdfplumber and PyPDF2
Text chunking with configurable size and overlap
Embedding generation using sentence transformers

Vector Storage:
FAISS index for efficient similarity search
CPU-optimized implementation

Question Answering:
Semantic search to find relevant document sections
Context-aware answer generation using transformer models
Confidence scoring for answer quality

Note: This Project is Poc and Can Still be imporved . I will consistenly Try To improve It. 

Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

-------------------------------------------------------------------------------- Thanks for visiting the Repo -----------------------------------------------------------------------------------------------------------
   
