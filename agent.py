from groq import Groq
import os
from dotenv import load_dotenv

# 👇 IMPORT YOUR FUNCTIONS
from rag import rag_search
from web_search import web_search

# Load environment variables from .env
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def agent(query: str) -> str:
    """
    Main agent function that combines:
    - RAG context (local documents)
    - Web search context (Serper)
    """

    # Retrieve contexts
    rag_context = rag_search(query)
    web_context = web_search(query)

    # Combine contexts
    context = f"""
RAG Context:
{rag_context}

Web Context:
{web_context}
"""

    # Call Groq LLM
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "Answer the question using ONLY the provided context. If the answer is not present, say you don't know."
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# Optional local test
if __name__ == "__main__":
    print(agent("What is RAG in AI?"))
