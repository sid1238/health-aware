from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
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
def split_into_chunks(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_json(text):
    import json
    import re

    # Try to extract JSON block from text
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from model"}

    return {"error": "No JSON found"}

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
        # -------- Step 1: Retrieve Context --------
        q_embed = embedder.encode([user_question])
        _, I = index.search(np.array(q_embed), k=3)
        context = "\n".join([all_chunks[i] for i in I[0]])

        # -------- Step 2: Generate Answer --------
        prompt = (
            f"You are a helpful health assistant. You will answer without adding any references or links and will not add markdowns.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{user_question}\n\n"
            f"Answer:"
        )

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response['choices'][0]['message']['content']

        # -------- Step 3: LLM Evaluation --------
        eval_prompt = f"""
You are an evaluation system.

Evaluate the answer.

Question: {user_question}
Answer: {answer}

Return ONLY valid JSON:

{{
  "relevance": 1-5,
  "completeness": 1-5,
  "fluency": 1-5,
  "overall": average,
  "feedback": "short"
}}
"""

        eval_response = llm.create_chat_completion(
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens = 120
        )

        raw_eval = eval_response['choices'][0]['message']['content']
        evaluation = extract_json(raw_eval)

        if "error" in evaluation: 
            evaluation = { "relevance": None, 
                          "completeness": None, 
                          "fluency": None, 
                          "overall": None, 
                          "feedback": "Evaluation failed" } 
        return jsonify({ "question": user_question, 
                            "answer": answer, 
                            "evaluation": evaluation })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# ----------- Run App -----------

if __name__ == "__main__":
    app.run(debug=True)