# AskMyPDF

### `AskMyPDF` is a RAG application to chat with your PDF.

#### Description

Given a PDF, build a RAG chain and use a local LLM to ask questions relevant to the document

#### Purpose 

The application has been developed for experimentation and learning purposes.


#### PreReqs

The app is built using
- `langchain_core` and `langchain_community` for the runnables
- `llama_cpp_python` to use compiled version of GGUF models (Llama-2, Mistral, etc)
- `chromadb` to store the vector embeddings
- `streamlit` to build the UI for the Human-AI QA interaction

