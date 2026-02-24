import streamlit as st
import os
from rag_pipeline import (
    load_documents, split_documents,
    create_vectorstore, load_vectorstore, build_chain
)

st.set_page_config(page_title="Documents RAG Chatbot", page_icon="🤖", layout="wide")
st.title("Documents RAG Chatbot - Ask about your document📑")

with st.sidebar:
    st.header("Upload your Document")
    uploaded_file = st.file_uploader("Choose PDF File", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        if st.button("File Processing"):
            with st.spinner("File is being analyzed..."):
                docs = load_documents("temp.pdf")
                chunks = split_documents(docs)
                create_vectorstore(chunks)
                st.success(f"{len(chunks)} text snippet has been processed!")

if "chain" not in st.session_state:
    if os.path.exists("./chroma_db"):
        vectorstore = load_vectorstore()
        st.session_state.chain = build_chain(vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = []

# disply chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])       

if question := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.write(question)
    
    if "chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                result = st.session_state.chain({"question": question})
                answer = result["answer"]
                sources = result["source_documents"]
                
                st.write(answer)
                
                # disply sources
                with st.expander("Sources 📚"):
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i+1}:** Page {doc.metadata.get('page', '?')}")
                        st.write(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({
                    "role": "assistant", "content": answer
                })
    else:
        st.warning("Please upload and process the PDF file first ⚠️")