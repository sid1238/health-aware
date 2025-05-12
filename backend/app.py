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

# Load embedding model and QA pipeline
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Step 1: Get content from Wikipedia
def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])

# Step 2: Chunk the document
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 3: Prepare documents
wiki_url = "https://en.wikipedia.org/wiki/Cancer"
raw_text = get_text_from_url(wiki_url)
chunks = split_into_chunks(raw_text)

# Step 4: Embed and index with FAISS
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    # Step 5: Embed the question
    question_embedding = embedder.encode([question])
    D, I = index.search(np.array(question_embedding), k=3)

    # Step 6: Get top 3 relevant chunks
    relevant_context = " ".join([chunks[i] for i in I[0]])

    # Step 7: Use QA model
    try:
        answer = qa_pipeline(question=question, context=relevant_context)
        return jsonify({"answer": answer["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)