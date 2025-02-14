import streamlit as st
import fitz  # PyMuPDF
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Store session state for uploaded PDFs and vectorstore
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI Configuration
st.set_page_config(page_title="📄 AI PDF Chat Assistant", layout="wide")

# Custom CSS for chat UI and disabling page scroll
st.markdown(
    """
    <style>
        /* Disable main page scrolling */
        html, body, [data-testid="stAppViewContainer"] {
            overflow: hidden;
        }

        /* Chat container styling */
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ddd;
            background-color: black;
            color: white;
        }

        .user-message, .bot-message {
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            margin: 5px 0;
        }

        .user-message { background-color: #4a235a; align-self: flex-end; }
        .bot-message { background-color: #6a1b9a; align-self: flex-start; }

        .message-wrapper {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📄 AI PDF Chat Assistant</h1>", 
    unsafe_allow_html=True
)

# Sidebar - Model Selection and File Upload
st.sidebar.title("🔧 Settings")

# Get list of available Ollama models
def get_available_models():
    model_list = os.popen("ollama list").read().split("\n")
    models = [line.split(" ")[0] for line in model_list if line]
    return models

available_models = get_available_models()
selected_model = st.sidebar.selectbox("🧠 Choose AI Model", available_models, index=0)

uploaded_files = st.sidebar.file_uploader(
    "📂 Upload PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)

clear_data_button = st.sidebar.button("🗑️ Clear PDF Data & Embeddings")

if clear_data_button:
    st.session_state.pdf_files = []
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.sidebar.success("Cleared all uploaded PDFs and embeddings.")

# Process PDFs ONLY if they haven't been processed before
if uploaded_files:
    new_pdfs = [file for file in uploaded_files if file.name not in st.session_state.pdf_files]

    if new_pdfs:  # Process only new PDFs
        with st.spinner("🔄 Processing PDFs..."):
            all_docs = []

            for uploaded_file in new_pdfs:
                st.session_state.pdf_files.append(uploaded_file.name)
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text = "\n".join([page.get_text("text") for page in doc])

                # Save PDF temporarily
                pdf_path = f"temp_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                loader = PyMuPDFLoader(pdf_path)
                documents = loader.load()
                all_docs.extend(documents)

            # Ensure valid text exists
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                docs = text_splitter.split_documents(all_docs)

                if docs:
                    embeddings = OllamaEmbeddings(model=selected_model)
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                    else:
                        st.session_state.vectorstore.add_documents(docs)
                    st.sidebar.success(f"✅ {len(new_pdfs)} new PDF(s) processed successfully!")
                else:
                    st.error("⚠️ No valid text extracted from the PDFs. Please try a different file.")
            else:
                st.error("⚠️ No readable content found in the uploaded PDFs.")

# Persistent chat input box
query = st.text_input("💬 Ask a question from the PDFs:")

# "Chat" button
chat_button = st.button("💡 Chat")

if chat_button:
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload at least one PDF before chatting.")
    elif not query.strip():
        st.warning("⚠️ Please enter a question before clicking Chat.")
    else:
        with st.spinner("🤖 Generating answer..."):
            # Use selected AI model
            llm = OllamaLLM(
                model=selected_model,
                temperature=0.7,
                max_tokens=4096,
                stream=True
            )

            # Improved retrieval: fetch top 5 relevant chunks
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

            response = qa_chain.run(query)

            # Store chat history
            st.session_state.chat_history.append(("👤", query))
            st.session_state.chat_history.append(("🤖", response))

# Display chat history
st.markdown("### 🗨️ Chat History")
chat_placeholder = st.container()

with chat_placeholder:
    chat_html = '<div class="chat-container">'
    for sender, message in st.session_state.chat_history:
        css_class = "user-message" if sender == "👤" else "bot-message"
        chat_html += f'<div class="message-wrapper"><div class="{css_class}">{sender} {message}</div></div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

# Display reference files
st.sidebar.subheader("📑 PDFs in Use:")
for pdf_name in st.session_state.pdf_files:
    st.sidebar.write(f"📘 {pdf_name}")
