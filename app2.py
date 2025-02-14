import streamlit as st
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Streamlit UI
st.set_page_config(page_title="📄 AI PDF Assistant", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📄 AI PDF Assistant</h1>", 
    unsafe_allow_html=True
)

st.sidebar.title("📂 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or multiple PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Persistent chat box
query = st.text_input("💬 Ask a question from the PDFs:")

# Chat button
chat_button = st.button("💡 Chat")

if chat_button:
    # Check if PDFs are uploaded
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF before chatting.")
    # Check if user entered a query
    elif not query.strip():
        st.warning("⚠️ Please enter a question before clicking Chat.")
    else:
        with st.spinner("Processing PDFs..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text = "\n".join([page.get_text("text") for page in doc])

                # Save PDF for loading in Langchain
                pdf_path = f"temp_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                loader = PyMuPDFLoader(pdf_path)
                documents = loader.load()
                all_docs.extend(documents)

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(all_docs)

            # Create embeddings and FAISS vector store
            embeddings = OllamaEmbeddings(model="llama3.2:latest")    
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Load Ollama model and set up RetrievalQA
            llm = OllamaLLM(model="llama3.2:latest")
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

            st.success(f"✅ {len(uploaded_files)} PDF(s) processed successfully!")

        # Display reference files
        st.sidebar.subheader("📑 PDFs in Use:")
        for uploaded_file in uploaded_files:
            st.sidebar.write(f"📘 {uploaded_file.name}")

        # Process user query
        with st.spinner("Fetching answer..."):
            response = qa_chain.run(query)
            st.markdown("### 📜 Answer:")
            st.write(response)
