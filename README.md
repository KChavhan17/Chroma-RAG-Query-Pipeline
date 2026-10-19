# ⚡ Chroma-RAG-Query-Pipeline

A production-ready Retrieval-Augmented Generation (RAG) assistant designed for document-grounded question answering with zero hallucination. Built using Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Google Gemini.

---

## 🚀 Live Demo
🔗 **Try the Application:** [https://chroma-rag-query-pipeline.streamlit.app](https://chroma-rag-query-pipeline.streamlit.app)

---

## 📌 Architecture Overview


[ User PDF Upload ]
│
▼
[ PyPDFLoader & RecursiveCharacterTextSplitter ] (Chunk size: 1000, Overlap: 150)
│
▼
[ HuggingFace Embeddings ] (sentence-transformers/all-MiniLM-L6-v2)
│
▼
[ ChromaDB Vector Store ] (Local persistent indexing)
│
▼
[ Relevant Context Retrieval ] (Top-K similarity matching)
│
▼
[ LangChain LCEL Pipeline ] + [ Google Gemini ]
│
▼
[ Synthesized Answer / Streamlit UI ]


---

## 🛠️ Tech Stack


* **Frontend / UI:** Streamlit
  
* **RAG Orchestration:** LangChain (LCEL)
  
* **Vector Store:** ChromaDB

  
* **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`

  
* **Large Language Model:** Google Gemini (`gemini-1.5-flash` / Google GenAI)

  
* **Document Parsing:** PyPDF

---

## ⚙️ Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KChavhan17/Chroma-RAG-Query-Pipeline.git](https://github.com/KChavhan17/Chroma-RAG-Query-Pipeline.git)
   cd Chroma-RAG-Query-Pipeline


1.Create and activate a virtual environment:


python -m venv venv


# Windows:


venv\Scripts\activate


# Linux/macOS:


source venv/bin/activate


2.Install dependencies:


pip install -r requirements.txt


3.Set up environment variables:


Create a .env file in the root directory:

GOOGLE_API_KEY=your_google_api_key_here


4.Run the Streamlit application:


streamlit run app.py


🔒 Security
API keys are handled via python-dotenv for local development 


and encrypted st.secrets for production cloud deployment. 


Secret keys, virtual environments, and 


local ChromaDB persistence stores are strictly excluded 


from version control via .gitignore.

