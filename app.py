import streamlit as st
from agent import agent

st.title("🔍 RAG + Web Search Agent")

query = st.text_input("Ask your question")

if st.button("Search"):
    with st.spinner("Thinking..."):
        answer = agent(query)
        st.write(answer)
