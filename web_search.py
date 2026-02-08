import requests
import os

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def web_search(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query}

    res = requests.post(url, headers=headers, json=payload)
    data = res.json()

    snippets = []
    for item in data.get("organic", [])[:3]:
        snippets.append(item["snippet"])

    return "\n".join(snippets)
