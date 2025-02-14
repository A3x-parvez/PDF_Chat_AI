import streamlit as st
import fitz  # PyMuPDF
import os
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Store session state for PDFs, vectorstore, and chat history
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI Configuration
st.set_page_config(page_title="OllamaPDF", layout="wide")

st.markdown("""
    <style>
    /* Change file uploader button color */
    div.stFileUploader > label {
        background-color: #9b59b6 !important;  /* Custom background color */
        color: white !important;  /* Text color */
        border-radius: 5px !important;
        padding: 10px !important;
        font-weight: bold !important;
        text-align: center !important;
        font-size: 18px !important;
    }

    /* Change hover effect */
    div.stFileUploader > label:hover {
        background-color: #512e5f !important;  /* Darker purple on hover */
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* Style the selectbox */
    div[data-baseweb="select"] > div {
        background-color: #9b59b6 !important;  /* Background color */
        border-radius: 5px !important;
        border: 2px solid white !important;
        color: black !important;
        font-size: 18px !important;
    }
    
    /* Change dropdown text color */
    div[data-baseweb="select"] span {
        color: white !important;
        font-weight: bold !important;
    }

    /* Change hover effect */
    div[data-baseweb="select"]:hover > div {
        background-color: #512e5f !important;  /* Hover color */
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    .stButton>button {
        background-color:#9b59b6  ;  /* Replace with your desired color */
        color: black;
        border: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        font-size: 24px;
        width : 100px;
    }
    .stButton>button:hover {
        background-color:#512e5f  ;  /* Hover color */
    }
    </style>
""", unsafe_allow_html=True)


# Custom CSS for chat UI
st.markdown(
    """
    <style>
        .chat-container {
            max-height: 500px;
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
        .timestamp { font-size: 12px; color: #aaa; margin-left: 10px; }
        .message-wrapper {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown("""
#     <style>
#         /* Change the slider track (line) color */
#         div[data-baseweb="slider"] div:first-child {
#             background: #9b59b6 !important; /* Purple */
#         }

#         /* Change the slider progress (filled part) color */
#         div[data-baseweb="slider"] div[aria-hidden="true"] {
#             background: #9b59b6 !important;  /* Purple */
#         }

#         /* Change the slider thumb (dot) color */
#         div[data-baseweb="slider"] div[role="slider"] {
#             background: #9b59b6 !important; /* Purple */
#             border: 2px solid white !important; /* White border for visibility */
#         }
#     </style>
# """, unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: #9b59b6 ;'>Ollama PDF Chat Assistant</h2>", unsafe_allow_html=True)

# Sidebar - Model Selection and File Upload
st.sidebar.title("⚙️Settings")

# Get available Ollama models
def get_available_models():
    model_list = os.popen("ollama list").read().split("\n")
    models = [line.split(" ")[0] for line in model_list if line]
    return models[1:]

available_models = get_available_models()
selected_model = st.sidebar.selectbox("🧠Choose AI Model", available_models, index=0)

# 🔥 Add Temperature Slider (0.0 - 1.0)
temp = st.sidebar.slider("🎛️ Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

uploaded_files = st.sidebar.file_uploader("📂Upload PDFs", type=["pdf"], accept_multiple_files=True)

clear_data_button = st.sidebar.button("Clear All")

if clear_data_button:
    st.session_state.pdf_files = []
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.sidebar.success("Cleared all uploaded PDFs and embeddings.")

# Process PDFs only if they haven't been processed before
new_files = [file for file in uploaded_files if file.name not in st.session_state.pdf_files]

if new_files:
    with st.spinner("🔄 Processing PDFs..."):
        all_docs = []
        
        for uploaded_file in new_files:
            st.session_state.pdf_files.append(uploaded_file.name)
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc])

            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            all_docs.extend(documents)

        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            docs = text_splitter.split_documents(all_docs)

            if docs:
                embeddings = OllamaEmbeddings(model=selected_model)
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                st.sidebar.success(f"✅ {len(new_files)} new PDF(s) processed successfully!")
            else:
                st.error("⚠️ No valid text extracted from the PDFs.")
        else:
            st.error("⚠️ No readable content found in the uploaded PDFs.")

chat_placeholder = st.container()

with chat_placeholder:
    chat_html = '<div class="chat-container">'
    for sender, message, timestamp in st.session_state.chat_history:
        css_class = "user-message" if sender == "😃" else "bot-message"
        chat_html += f'<div class="message-wrapper"><div class="{css_class}">{sender} {message}<span class="timestamp"> ⏰ {timestamp}</span></div></div>'
    chat_html += "</br></div></br>"
    st.markdown(chat_html, unsafe_allow_html=True)

# Chat input box
query = st.text_area("Enter text :", placeholder="Type your message here...", label_visibility="collapsed")


# "Chat" button
chat_button = st.button("Chat",type="secondary")

if chat_button:
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload at least one PDF before chatting.")
    elif not query.strip():
        st.warning("⚠️ Please enter a question before clicking Chat.")
    else:
        with st.spinner("🤖 Generating answer..."):
            llm = OllamaLLM(
                model=selected_model,
                temperature=temp,
                max_tokens=4096,
                stream=True
            )

            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

            response = qa_chain.run(query)

            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append(("😃", query, timestamp))
            st.session_state.chat_history.append(("🤖", response, timestamp))

            # ✅ Force Streamlit to rerun so messages appear immediately
            st.rerun()

# Display reference files
st.sidebar.subheader("📑 PDFs in Use:")
for pdf_name in st.session_state.pdf_files:
    st.sidebar.write(f"📘 {pdf_name}")
