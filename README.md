# ⚡ Chroma-RAG-Query-Pipeline

A production-ready Retrieval-Augmented Generation (RAG) assistant designed for document-grounded question answering with zero hallucination. Built using Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Google Gemini.

---

## 🚀 Live Demo
🔗 **Try the Application:** [https://chroma-rag-query-pipeline.streamlit.app](https://chroma-rag-query-pipeline.streamlit.app)

---


## 📌 Architecture Overview


```text
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
```


## 🛠️ Tech Stack


* **Frontend / UI:** Streamlit
  
* **RAG Orchestration:** LangChain (LCEL)
  
* **Vector Store:** ChromaDB

  
* **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`

  
* **Large Language Model:** Google Gemini (`gemini-1.5-flash` / Google GenAI)

  
* **Document Parsing:** PyPDF

---


## ⚙️ Local Setup Instructions

**1. Clone the repository:**
```bash
git clone [https://github.com/KChavhan17/Chroma-RAG-Query-Pipeline.git](https://github.com/KChavhan17/Chroma-RAG-Query-Pipeline.git)
cd Chroma-RAG-Query-Pipeline
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
```
* **Windows:**
  ```powershell
  venv\Scripts\activate
  ```
* **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**  
Create a `.env` file in the root folder:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

**5. Run the application:**
```bash
streamlit run app.py
```


## 🔒 Security
API keys are handled via `python-dotenv` for local runs and encrypted `st.secrets` for cloud deployment. No credentials or temporary vector data are tracked in version control.
