import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# NEW (WORKS SEAMLESSLY BOTH LOCALLY & ON CLOUD)
load_dotenv()

try:
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key or ""

st.set_page_config(page_title="Free RAG Assistant", layout="wide")
st.title("📄 Free Local RAG Assistant (Gemini)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    @st.cache_resource
    def process_pdf(path):
        loader = PyPDFLoader(path)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")

    retriever = process_pdf(temp_path).as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        "Answer based only on the provided context:\n{context}\n\nQuestion: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about the PDF:"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            ans = rag_chain.invoke(query)
            st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})