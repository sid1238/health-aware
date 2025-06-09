from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup

import re

app = Flask(__name__)
CORS(app)

# Load LLM
# llm = Llama(
#     model_path="./models/TinyLlama-GGUF/TinyLlama-1.1b-chat-v1.0.Q4_K_M.gguf",
#     n_ctx=2048,
#     n_threads=4,
#     temperature=0.7,
#     top_p=0.9
# )
llm = Llama.from_pretrained(
	repo_id="google/gemma-3-1b-it-qat-q4_0-gguf",
	filename="gemma-3-1b-it-q4_0.gguf",
)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get content from Wikipedia
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# Chunk text into manageable pieces
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ----------- Load & Prepare Data -----------

# Example Wikipedia URLs
wiki_urls = [
    "https://en.wikipedia.org/wiki/Cancer",
    "https://en.wikipedia.org/wiki/Diabetes"
]

# Example custom strings
custom_sentences = [
    "Regular exercise can help reduce the risk of cardiovascular diseases.",
    "Drinking enough water is essential for maintaining good health.",
    "Meditation and mindfulness help improve mental health.",
    "A diet rich in fruits and vegetables supports a healthy immune system."
]

# Process all sources
all_chunks = []

# Wiki docs
for url in wiki_urls:
    text = get_text_from_url(url)
    chunks = split_into_chunks(text)
    all_chunks.extend(chunks)

# Add custom sentence chunks
all_chunks.extend(split_into_chunks(" ".join(custom_sentences)))

# Compute embeddings and build FAISS index
embeddings = embedder.encode(all_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# ----------- API Endpoint -----------

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Step 1: Embed question and search
        q_embed = embedder.encode([user_question])
        _, I = index.search(np.array(q_embed), k=1)
        context = "\n".join([all_chunks[i] for i in I[0]])[:200]

        # Step 2: Format prompt
        prompt = f"context:{context}\tAnswer the following question: {user_question}"
        response = llm.create_chat_completion(
            messages =[
                {"role": "user", "content": prompt}
                ]
            )
        
        
        return jsonify({"answer": response['choices'][0]['message']['content']})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# ----------- Run App -----------

if __name__ == "__main__":
    app.run(debug=True)