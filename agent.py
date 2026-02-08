from rag import rag_search
from web_search import web_search
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def agent(query):
    rag_context = rag_search(query)
    web_context = web_search(query)

    context = f"""
RAG Context:
{rag_context}

Web Context:
{web_context}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Answer using the provided context."},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
        ]
    )

    return response.choices[0].message.content
