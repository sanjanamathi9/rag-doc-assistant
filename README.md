# AI Document Q&A Assistant using RAG

An AI-powered document question-answering web application
built using Retrieval-Augmented Generation (RAG).

## Technologies

- Python
- Streamlit
- LangChain
- Google Gemini
- FAISS
- PyPDF
- Embeddings

## Features

- Upload PDF documents
- Extract text from PDFs
- Split documents into chunks
- Generate embeddings
- Store embeddings using FAISS
- Retrieve relevant document content
- Generate answers using Gemini

## Architecture

PDF
→ Text Extraction
→ Chunking
→ Embeddings
→ FAISS
→ Similarity Search
→ Context
→ Gemini
→ Answer