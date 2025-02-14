import streamlit as st
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

# Streamlit UI
st.set_page_config(page_title="PDF AI Assistant", layout="wide")
st.title("📄 AI-Powered PDF Assistant using Ollama")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Extract text from PDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])

        # Save and load PDF using Langchain
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Create embeddings and FAISS vector store
        embeddings = OllamaEmbeddings(model="llama3.2:latest")  # Update model name
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Load Ollama model and set up RetrievalQA
        llm = Ollama(model="llama3.2:latest")  # Update model name
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

        st.success("PDF processed successfully! Now you can ask questions.")

    # User input for Q&A
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Getting answer..."):
            response = qa_chain.run(query)
            st.write("**Answer:**", response)
