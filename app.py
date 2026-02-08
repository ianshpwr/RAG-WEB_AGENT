import streamlit as st
import os
import shutil
from agent import agent

st.set_page_config(page_title="RAG + Web Search Agent")

st.title("🔍 RAG + Web Search Agent")

st.markdown(
    "Upload your own knowledge file (TXT) and ask questions grounded in it."
)

# --- File upload ---
uploaded_file = st.file_uploader(
    "Upload your knowledge file (.txt)",
    type=["txt"]
)

if uploaded_file:
    os.makedirs("data", exist_ok=True)

    with open("data/knowledge.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Remove old FAISS index so it rebuilds
    if os.path.exists("db"):
        shutil.rmtree("db")

    st.success("✅ Knowledge base uploaded and indexed successfully!")

# --- Question input ---
query = st.text_input(
    "Ask a question based on your knowledge + live web"
)

if st.button("Search") and query:
    with st.spinner("Thinking..."):
        answer = agent(query)
        st.write(answer)
