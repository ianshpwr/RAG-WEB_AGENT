from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def rag_search(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("db", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])
