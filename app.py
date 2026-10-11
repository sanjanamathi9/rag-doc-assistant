import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )
    vector_store.save_local("faiss_index")

def ask_question(question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    documents = vector_store.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are an AI Document Assistant.

Your job is to answer the user's question using ONLY
the information provided in the uploaded document.

IMPORTANT RULES:

1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not present in the document, say:
   "The answer is not available in the uploaded document."
4. Give a clear and easy-to-understand answer.
5. Use short paragraphs when explanation is needed.
6. Use bullet points when listing multiple items.
7. Use numbered points when explaining steps.
8. Use Markdown formatting for headings and important terms.
9. Do NOT return JSON.
10. Do NOT return Python objects.
11. Do NOT return metadata, signatures, or technical information.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2
    )

    response = model.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts)
    return str(content)


def main():

    st.set_page_config(
        page_title="AI Document Assistant",
        layout="wide"
    )

    st.title("AI Document Assistant")

    st.write(
        "Upload your PDF documents and ask questions "
        "using Retrieval-Augmented Generation (RAG)."
    )

    st.divider()
    with st.sidebar:

        st.header("Upload Documents")

        pdf_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        st.write("")
        if st.button(
            "Process Documents",
            use_container_width=True
        ):

            if pdf_files:

                with st.spinner(
                    "Reading and processing your documents..."
                ):
                    raw_text = get_pdf_text(pdf_files)
                    if not raw_text.strip():

                        st.error(
                            " Could not extract text from the PDF."
                        )

                    else:
                        text_chunks = get_text_chunks(
                            raw_text
                        )
                        create_vector_store(
                            text_chunks
                        )
                        st.success(
                            " Documents processed successfully!"
                        )
                        st.info(
                            f"Created {len(text_chunks)} text chunks."
                        )

            else:
                st.warning(
                    " Please upload at least one PDF."
                )

        st.divider()

        st.caption(
            "RAG Pipeline: PDF → Chunks → Embeddings → "
            "FAISS → Retrieval → Gemini"
        )

    st.header("Ask a Question")

    question = st.text_input(
        "Enter your question about the uploaded document:",
        placeholder="Example: What are the main components of IoT?"
    )

    if question:
        if os.path.exists("faiss_index"):

            with st.spinner(
                "Searching the document and generating answer..."
            ):

                try:
                    answer = ask_question(question)
                    st.subheader(" Answer")
                    st.markdown(answer)

                except Exception as e:

                    st.error(
                        "Something went wrong while generating "
                        "the answer."
                    )

                    st.exception(e)
        else:
            st.warning(
                "Please upload and process a document first."
            )

if __name__ == "__main__":
    main()