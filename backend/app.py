from flask import Flask, request, jsonify 
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Text chunking function
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Fetch text from Wikipedia URL
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
    except:
        return ""

# === Step 1: Fetch and embed multiple articles ===
wiki_urls = [
    "https://en.wikipedia.org/wiki/Cancer",
    "https://en.wikipedia.org/wiki/Diabetes",
    "https://en.wikipedia.org/wiki/Hypertension"
]

all_chunks = []
for url in wiki_urls:
    text = get_text_from_url(url)
    chunks = split_into_chunks(text)
    all_chunks.extend(chunks)

# Embed and index
embeddings = embedder.encode(all_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# === Step 2: Ask endpoint ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    try:
        question_embedding = embedder.encode([question])
        D, I = index.search(np.array(question_embedding), k=3)
        relevant_context = " ".join([all_chunks[i] for i in I[0]])
        answer = qa_pipeline(question=question, context=relevant_context)
        return jsonify({"answer": answer["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
