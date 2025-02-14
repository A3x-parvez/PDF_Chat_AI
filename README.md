# Ollama PDF Chat Assistant

Ollama PDF Chat Assistant is a Streamlit-based application that allows users to upload PDF files and interact with them using AI-powered chat. It utilizes Ollama LLM for generating responses and FAISS for efficient document retrieval.

## Features

- 📂 **Upload and Process PDFs**: Extract text from PDFs and create embeddings.
- 🤖 **AI-Powered Chat**: Ask questions about the uploaded PDFs using a retrieval-based chat system.
- 🧠 **Customizable AI Model**: Select from available Ollama models.
- 🎛️ **Temperature Control**: Adjust response creativity with a temperature slider.
- 📑 **Chat History**: Keeps track of past interactions.
- 🎨 **Custom UI**: Modern and visually appealing chat interface.
- 🔄 **Automatic PDF Processing**: Automatically processes newly uploaded PDFs.
- ⚡ **Efficient Search & Retrieval**: Uses FAISS for quick and accurate document retrieval.

## Tech Stack

- **Frontend**: Streamlit (for UI and interactivity)
- **Backend**: LangChain, Ollama LLM
- **PDF Processing**: PyMuPDF (Fitz)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embedding Model**: Ollama Embeddings
- **Styling**: Custom CSS for improved UI

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Ollama (Installed and running locally)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/ollama-pdf-chat.git
   cd ollama-pdf-chat
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage

1. Open the application in a browser.
2. Upload one or more PDF files.
3. Select the AI model from the sidebar.
4. Adjust the temperature setting if needed.
5. Enter a question related to the PDFs and click "Chat."
6. View AI-generated responses in the chat window.

## Technologies Used

- **Streamlit**: For the web interface.
- **PyMuPDF (Fitz)**: To extract text from PDFs.
- **FAISS**: For vector search and retrieval.
- **LangChain**: To integrate AI models and create the chat pipeline.
- **Ollama**: For AI-powered responses.

## Customizations

- Modify `st.session_state` to enhance session handling.
- Adjust chunk sizes in `RecursiveCharacterTextSplitter` to optimize document segmentation.
- Customize the UI further in the `st.markdown()` sections using CSS.

## Contribution

Feel free to fork and submit pull requests! If you encounter issues, open a GitHub issue.

## Contact

For inquiries or support, reach out via:
- 📧 Email: support@ollamapdfchat.com
- 🐦 Twitter: [@OllamaChat](https://twitter.com/OllamaChat)
- 📘 GitHub Issues: [Submit an Issue](https://github.com/your-username/ollama-pdf-chat/issues)

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to the open-source community for building amazing AI tools! 🚀

