import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = "db"
DATA_PATH = "data/knowledge.txt"


def create_db():
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    return db


def rag_search(query: str) -> str:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if not os.path.exists(DATA_PATH):
        return ""

    if not os.path.exists(DB_PATH):
        db = create_db()
    else:
        db = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])
