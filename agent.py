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
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
You are an intelligent Retrieval-Augmented AI Agent.

Your goal is to answer user queries accurately, clearly, and concisely by
combining information from:
1. Retrieved context from a local knowledge base (RAG)
2. Live web search results provided via Serper

Operating rules:
1. Always prioritize retrieved local documents when they are relevant.
2. Use web search results only when:
   - The retrieved context is insufficient, outdated, or missing
   - The question requires recent, real-time, or factual updates
3. Never hallucinate information.
4. If the answer cannot be found in either retrieved context or web results,
   clearly say you do not know.

Context usage rules:
- Use retrieved document chunks as authoritative sources.
- Use web search snippets to supplement or validate information.
- If both sources are available, merge them logically into a single coherent response.
- Do not mention internal implementation details (embeddings, vector stores, APIs).

Response guidelines:
- Answer in clear, structured natural language.
- Be concise but informative.
- If multiple facts are presented, use bullet points where appropriate.
- When web search is used, base answers strictly on the provided search results.

Failure handling:
- If no relevant information is found, respond with:
  "I could not find reliable information to answer this question."

You are a professional AI research assistant, not a chatbot.
"""
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
