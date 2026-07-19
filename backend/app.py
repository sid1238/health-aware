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
import os
import hashlib

app = Flask(__name__)
CORS(app)

llm = Llama.from_pretrained(
    repo_id="google/gemma-3-1b-it-qat-q4_0-gguf",
    filename="gemma-3-1b-it-q4_0.gguf",
    n_ctx=2048,       # default is only 512 — too small once context + prompt + answer are combined
    n_batch=512,      # can leave as-is; only relevant to prompt-processing speed
)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Wikimedia will often return 403 for requests without a proper User-Agent.
# This was silently swallowed before (the except block just returned ""),
# which can shrink your corpus down to almost nothing without any visible error.
HEADERS = {
    "User-Agent": "HealthChatbotProject/1.0 (contact: youremail@example.com)"
}

# ----------- Persistence config -----------
# Where the built FAISS index and chunk metadata get cached on disk so we don't
# re-scrape all 21 URLs and re-embed everything on every single restart.

CACHE_DIR = "rag_cache"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
SOURCES_HASH_PATH = os.path.join(CACHE_DIR, "sources.hash")

os.makedirs(CACHE_DIR, exist_ok=True)

# ----------- Text acquisition & cleaning -----------

def get_text_from_url(url):
    """Fetch and clean the main body text of a Wikipedia (or similar) article."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"[WARN] {url} returned status {response.status_code}")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup.find_all(["table", "sup", "style", "script"]):
            tag.decompose()

        content = soup.find("div", {"id": "mw-content-text"}) or soup
        paragraphs = content.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip())

        cleaned = clean_text(text)
        if not cleaned:
            print(f"[WARN] {url} returned 200 but no paragraph text was extracted")
        return cleaned
    except Exception as e:
        print(f"[ERROR] Exception fetching {url}: {e}")
        return ""


def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[edit\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(text, target_words=150, overlap_sentences=2):
    sentences = split_into_sentences(text)
    chunks = []
    current, word_count = [], 0

    for sent in sentences:
        current.append(sent)
        word_count += len(sent.split())
        if word_count >= target_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            word_count = sum(len(s.split()) for s in current)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def truncate_to_word_budget(text, max_words=220):
    """
    Rough safety cap on context size. One word is roughly ~1.3 tokens for
    English text, so 220 words keeps the context comfortably under ~300
    tokens even in a worst case, leaving headroom in a 2048-token window for
    the system prompt, question, and the model's own answer.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from model"}
    return {"error": "No JSON found"}


# ----------- Load & Prepare Data -----------
# Expanded corpus covering the categories users are most likely to ask about:
# chronic diseases, common symptoms/acute conditions, medications, nutrition,
# and mental health. Add/remove URLs here as you learn what your users ask about.

wiki_urls = [
    # Chronic diseases
    "https://en.wikipedia.org/wiki/Cancer",
    "https://en.wikipedia.org/wiki/Diabetes",
    "https://en.wikipedia.org/wiki/Hypertension",
    "https://en.wikipedia.org/wiki/Asthma",
    "https://en.wikipedia.org/wiki/Obesity",
    "https://en.wikipedia.org/wiki/Chronic_kidney_disease",

    # Common symptoms / acute conditions
    "https://en.wikipedia.org/wiki/Common_cold",
    "https://en.wikipedia.org/wiki/Influenza",
    "https://en.wikipedia.org/wiki/Migraine",
    "https://en.wikipedia.org/wiki/Fever",
    "https://en.wikipedia.org/wiki/Gastroenteritis",

    # Medications
    "https://en.wikipedia.org/wiki/Paracetamol",
    "https://en.wikipedia.org/wiki/Ibuprofen",
    "https://en.wikipedia.org/wiki/Antibiotic",
    "https://en.wikipedia.org/wiki/Insulin_(medication)",

    # Nutrition
    "https://en.wikipedia.org/wiki/Nutrition",
    "https://en.wikipedia.org/wiki/Vitamin_D",
    "https://en.wikipedia.org/wiki/Dietary_fiber",

    # Mental health
    "https://en.wikipedia.org/wiki/Major_depressive_disorder",
    "https://en.wikipedia.org/wiki/Anxiety_disorder",
    "https://en.wikipedia.org/wiki/Sleep",
    "https://en.wikipedia.org/wiki/Stress_(biology)",
]

custom_sentences = [
    "Regular exercise can help reduce the risk of cardiovascular diseases.",
    "Drinking enough water is essential for maintaining good health.",
    "Meditation and mindfulness help improve mental health.",
    "A diet rich in fruits and vegetables supports a healthy immune system.",
]

def get_sources_fingerprint():
    """Hash of the current source list. If wiki_urls or custom_sentences change,
    this hash changes too, which tells load_or_build_index() the cache is stale
    and needs to be rebuilt instead of silently serving outdated content."""
    raw = "|".join(wiki_urls) + "|" + "|".join(custom_sentences)
    return hashlib.sha256(raw.encode()).hexdigest()


def build_corpus():
    """Scrape, clean, and chunk every source. Returns a list of
    {"text": ..., "source": ...} dicts. This is only called when there's no
    valid cache on disk."""
    all_chunks = []

    print("\n--- Building corpus (scraping + chunking) ---")
    for url in wiki_urls:
        text = get_text_from_url(url)
        if not text:
            print(f"[SKIP] {url} produced no usable text.")
            continue
        chunks = chunk_by_sentences(text)
        print(f"[OK] {url} -> {len(chunks)} chunks, {len(text.split())} words")
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": url})

    custom_text = " ".join(custom_sentences)
    custom_chunks = chunk_by_sentences(custom_text, target_words=50, overlap_sentences=1)
    print(f"[OK] custom_health_tips -> {len(custom_chunks)} chunks")
    for chunk in custom_chunks:
        all_chunks.append({"text": chunk, "source": "custom_health_tips"})

    print(f"--- Total chunks built: {len(all_chunks)} ---\n")

    if len(all_chunks) <= len(custom_chunks):
        print("*** WARNING: none of the Wikipedia URLs produced content. "
              "Your corpus is effectively just the 4 custom sentences. "
              "Check the [WARN]/[ERROR] messages above. ***\n")

    return all_chunks


def load_or_build_index():
    """Load the FAISS index + chunk metadata from disk if a valid cache exists
    for the current source list; otherwise scrape/embed fresh and cache the
    result so the next restart can skip straight to loading."""
    current_hash = get_sources_fingerprint()

    cache_valid = (
        os.path.exists(INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
        and os.path.exists(SOURCES_HASH_PATH)
    )

    if cache_valid:
        with open(SOURCES_HASH_PATH, "r") as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            print("--- Loading cached FAISS index and chunks from disk ---")
            cached_index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "r") as f:
                cached_chunks = json.load(f)
            print(f"--- Loaded {len(cached_chunks)} chunks from cache ---\n")
            return cached_index, cached_chunks
        else:
            print("--- Source list changed since last run — rebuilding index ---")

    chunks = build_corpus()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    new_index = faiss.IndexFlatIP(embeddings.shape[1])
    new_index.add(embeddings)

    # Persist to disk so the next startup can skip scraping + embedding entirely
    faiss.write_index(new_index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)
    with open(SOURCES_HASH_PATH, "w") as f:
        f.write(current_hash)

    print(f"--- Cached index + chunks to '{CACHE_DIR}/' for future restarts ---\n")
    return new_index, chunks


index, all_chunks = load_or_build_index()

# Lowered from 0.35 — all-MiniLM-L6-v2 is a general-purpose symmetric similarity
# model, not tuned for question-vs-passage retrieval, so genuine matches often
# score in the 0.15-0.30 range. Use the /debug endpoint below to see real scores
# for your own questions and adjust this number accordingly.
RELEVANCE_THRESHOLD = 0.15

# ----------- Debug endpoint -----------
# Hit this with the same question that's failing to see the ACTUAL similarity
# scores being returned, before any threshold is applied. This tells you whether
# you have a scraping problem (few/no chunks, or all from custom_health_tips)
# or a threshold problem (good chunks are there, but scoring below the cutoff).

@app.route("/debug", methods=["POST"])
def debug():
    data = request.get_json()
    user_question = data.get("question", "")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    q_embed = embedder.encode([user_question], normalize_embeddings=True)
    q_embed = np.array(q_embed).astype("float32")

    k = 5
    scores, idxs = index.search(q_embed, k)
    scores, idxs = scores[0], idxs[0]

    results = [
        {
            "score": float(score),
            "source": all_chunks[i]["source"],
            "text_preview": all_chunks[i]["text"][:200],
            "passes_current_threshold": float(score) >= RELEVANCE_THRESHOLD,
        }
        for i, score in zip(idxs, scores)
    ]

    return jsonify({
        "question": user_question,
        "total_chunks_in_index": len(all_chunks),
        "current_threshold": RELEVANCE_THRESHOLD,
        "top_matches": results,
    })


# ----------- API Endpoint -----------

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        q_embed = embedder.encode([user_question], normalize_embeddings=True)
        q_embed = np.array(q_embed).astype("float32")

        k = 3
        scores, idxs = index.search(q_embed, k)
        scores, idxs = scores[0], idxs[0]

        retrieved = [
            (all_chunks[i], float(score))
            for i, score in zip(idxs, scores)
            if score >= RELEVANCE_THRESHOLD
        ]

        if not retrieved:
            answer = (
                "I don't have enough reliable information in my current sources "
                "to answer that confidently. Please consult a healthcare "
                "professional or rephrase your question."
            )
            return jsonify({
                "question": user_question,
                "answer": answer,
                "evaluation": {
                    "relevance": None, "completeness": None, "fluency": None,
                    "overall": None, "feedback": "No relevant context retrieved"
                }
            })

        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in retrieved)
        context = truncate_to_word_budget(context, max_words=220)

        system_prompt = (
            "You are a helpful health assistant. Answer using ONLY the provided "
            "context. If the context does not fully answer the question, say so "
            "honestly instead of guessing. Do not include references, links, or "
            "markdown formatting in your answer."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_question}"

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.3,
        )

        answer = response["choices"][0]["message"]["content"]

        eval_prompt = f"""You are an evaluation system.

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
            max_tokens=120,
        )

        raw_eval = eval_response["choices"][0]["message"]["content"]
        evaluation = extract_json(raw_eval)

        if "error" in evaluation:
            evaluation = {
                "relevance": None, "completeness": None, "fluency": None,
                "overall": None, "feedback": "Evaluation failed"
            }

        return jsonify({
            "question": user_question,
            "answer": answer,
            "evaluation": evaluation,
            "sources": [c["source"] for c, _ in retrieved],
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


# ----------- Force a cache rebuild -----------
# Run `python health_chatbot.py --rebuild` to wipe the cache and re-scrape
# everything from scratch — useful if a source's content has changed on the
# web even though your wiki_urls list hasn't, since the hash check alone
# wouldn't catch that.

if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        for path in [INDEX_PATH, CHUNKS_PATH, SOURCES_HASH_PATH]:
            if os.path.exists(path):
                os.remove(path)
        print("--- Cache cleared, rebuilding now ---")
        index, all_chunks = load_or_build_index()

    app.run(debug=True)