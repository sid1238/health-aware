from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import inspect
import json
import re
import os
import hashlib
import shutil
import sys
import threading
import time
from rank_bm25 import BM25Okapi

app = Flask(__name__)
CORS(app)

# The LLM is loaded on the first /ask rather than at import. Only /ask needs it,
# so /debug, run_eval.py and index rebuilds stay fast, and Flask's debug reloader
# no longer holds a second copy of the model in its parent process.
_llm = None
_llm_lock = threading.Lock()  # llama.cpp contexts aren't safe for concurrent calls


def get_llm():
    """Must be called with _llm_lock held."""
    global _llm
    if _llm is None:
        print("--- Loading LLM (first question only) ---")
        _llm = Llama.from_pretrained(
            repo_id="google/gemma-3-1b-it-qat-q4_0-gguf",
            filename="gemma-3-1b-it-q4_0.gguf",
            n_ctx=2048,       # default is only 512 — too small once context + prompt + answer are combined
            n_batch=512,      # can leave as-is; only relevant to prompt-processing speed
        )
    return _llm


# Load embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Wikimedia will often return 403 for requests without a proper User-Agent.
# This was silently swallowed before (the except block just returned ""),
# which can shrink your corpus down to almost nothing without any visible error.
HEADERS = {
    "User-Agent": "HealthChatbotProject/1.0 (contact: youremail@example.com)"
}

# ----------- Persistence config -----------
# Where the built FAISS index and chunk metadata get cached on disk so we don't
# re-scrape every URL and re-embed everything on every single restart.
# Anchored to this file so the cache is found whatever directory you start from.

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_cache")
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
# Raw pages are kept separately so tuning the cleaning/chunking code re-embeds
# without re-downloading anything.
HTML_DIR = os.path.join(CACHE_DIR, "html")

os.makedirs(HTML_DIR, exist_ok=True)

# ----------- Text acquisition & cleaning -----------

def fetch_html(url, retries=3):
    """Return a page's HTML from the on-disk cache, or download and cache it.
    Returns None if it can't be fetched. Transient failures (rate limiting,
    server errors, timeouts) are retried with a growing delay."""
    path = os.path.join(HTML_DIR, hashlib.sha256(url.encode()).hexdigest()[:16] + ".html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                time.sleep(0.5)  # be polite to Wikipedia during a full scrape
                return response.text
            print(f"[WARN] {url} returned status {response.status_code}")
            if response.status_code != 429 and response.status_code < 500:
                return None
        except requests.RequestException as e:
            print(f"[WARN] Exception fetching {url}: {e}")
        time.sleep(2 ** attempt)
    return None


# Page furniture that isn't article prose.
NOISE_SELECTORS = [
    "table", "sup", "style", "script", "cite", "figure", "img",
    ".navbox", ".vertical-navbox", ".sidebar", ".infobox", ".hatnote",
    ".thumb", ".metadata", ".ambox", ".reflist", ".refbegin", ".gallery",
    ".mw-editsection", ".mw-empty-elt", ".shortdescription", ".noprint",
    ".portalbox", ".side-box", ".toc",
]

# Trailing sections that are lists of links and citations, not health content.
SKIPPED_SECTIONS = {
    "references", "see also", "external links", "further reading", "notes",
    "bibliography", "sources", "footnotes", "citations", "explanatory notes",
    "general bibliography", "gallery",
}


def extract_article(html):
    """Split an article into {"title": ..., "sections": [(name, text), ...]}.

    Lists are kept alongside paragraphs: symptom lists, diagnostic criteria and
    risk factors are often written as bullets, and scraping only <p> tags left
    chunks ending in "the following criteria must be met:" with nothing after.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1", {"id": "firstHeading"})
    title = heading.get_text(strip=True) if heading else ""

    root = soup.select_one("#mw-content-text .mw-parser-output") \
        or soup.find("div", {"id": "mw-content-text"}) or soup

    # Keep formulas readable as plain text instead of duplicated LaTeX markup.
    for math in root.select(".mwe-math-element"):
        node = math.find("math")
        alt = node.get("alttext", "") if node else ""
        math.replace_with(re.sub(r"\\displaystyle\s*", "", alt).strip("{} "))

    for selector in NOISE_SELECTORS:
        for tag in root.select(selector):
            tag.decompose()

    sections = []
    top_heading, name, parts = "", "Overview", []
    skipping = False

    for el in root.find_all(["h2", "h3", "p", "ul", "ol"]):
        if el.name in ("h2", "h3"):
            text = clean_text(" ".join(parts))
            if text:
                sections.append((name, text))
            parts = []
            heading_text = clean_text(el.get_text())
            if el.name == "h2":
                top_heading = heading_text
                name = heading_text
                skipping = heading_text.lower() in SKIPPED_SECTIONS
            else:
                name = f"{top_heading}: {heading_text}" if top_heading else heading_text
        elif skipping:
            continue
        elif el.name == "p":
            text = el.get_text(" ", strip=True)
            if len(text.split()) >= 4:
                parts.append(text)
        else:
            # Deepest items first, removing each one, so a parent item doesn't
            # repeat the text of the sub-list nested inside it.
            items = []
            for li in reversed(el.find_all("li")):
                item = clean_text(li.get_text(" ", strip=True))
                li.decompose()
                if item:
                    items.append(item if item[-1] in ".!?:" else item + ".")
            parts.extend(reversed(items))

    text = clean_text(" ".join(parts))
    if text and not skipping:
        sections.append((name, text))

    return {"title": title, "sections": sections}


def clean_text(text):
    text = re.sub(r"\[\s*(?:\d+|[a-z]|edit|citation needed|note \d+|nb \d+)\s*\]", "", text, flags=re.I)
    # Bibliography identifiers that survive inside body paragraphs.
    text = re.sub(r"\bdoi:\s*\S+", "", text)
    text = re.sub(r"\b(?:PMC|PMID|ISBN)[ :]*[\dX-]+\.?", "", text)
    text = re.sub(r"\s+", " ", text)
    # get_text(" ") and removed citations leave gaps before punctuation and inside brackets.
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


# Periods here don't end a sentence. Case-sensitive, so "No." and "So." still do.
SENTENCE_ABBREVIATIONS = r"\b(?:e\.g|i\.e|et al|etc|vs|cf|ca|approx|Dr|Mr|Mrs|Ms|Prof|St|Fig)\."
SENTENCE_BOUNDARY = r"(?<=[.!?])\s+(?=[A-Z0-9\"“'(\[])|(?<=[.!?][\"”')\]])\s+(?=[A-Z0-9\"“'(\[])"
MAX_SENTENCE_WORDS = 120
_DOT = "\u2024"  # stand-in for a protected period while splitting


def split_into_sentences(text):
    protected = re.sub(SENTENCE_ABBREVIATIONS, lambda m: m.group().replace(".", _DOT), text)
    sentences = []
    for sent in re.split(SENTENCE_BOUNDARY, protected):
        words = sent.replace(_DOT, ".").split()
        # A run-on with no punctuation would otherwise become one giant chunk.
        for i in range(0, len(words), MAX_SENTENCE_WORDS):
            sentences.append(" ".join(words[i:i + MAX_SENTENCE_WORDS]))
    return sentences


def chunk_by_sentences(text, target_words=150, overlap_sentences=2,
                       max_overlap_words=50, min_tail_words=30):
    """Group whole sentences into chunks of about target_words, repeating the
    last sentences of each chunk at the start of the next."""
    chunks = []
    current, new_sentences = [], []

    def word_count(sentences):
        return sum(len(s.split()) for s in sentences)

    for sent in split_into_sentences(text):
        current.append(sent)
        new_sentences.append(sent)
        if word_count(current) >= target_words:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            # A couple of long sentences shouldn't turn into half a chunk of repeats.
            while overlap and word_count(overlap) > max_overlap_words:
                overlap = overlap[1:]
            current, new_sentences = overlap, []

    # Only emit a tail if it holds something new: a tail of nothing but overlap
    # would be a duplicate of the end of the previous chunk.
    if new_sentences:
        if chunks and word_count(new_sentences) < min_tail_words:
            chunks[-1] += " " + " ".join(new_sentences)
        else:
            chunks.append(" ".join(current))

    return chunks


def truncate_to_word_budget(text, max_words=300):
    """
    Cap on context size, kept well below what n_ctx would allow. Measured on the
    Gemma 1B model: answers were good at up to ~475 prompt tokens (300 words), but
    at 630-720 tokens (500 words) most turned to gibberish like "The context.
    The context." Raising n_batch/n_ubatch made no difference, so it isn't a
    batching problem. Re-test with a few real questions before raising this.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def tokenize(text):
    """Simple lowercase word tokenizer for BM25. Deliberately basic (no
    stemming/lemmatization) — BM25 matches on exact tokens, which is exactly
    what we want it to complement embeddings with: exact drug names, dosage
    numbers, and specific terms that semantic search can blur past."""
    return re.findall(r"[a-z0-9]+", text.lower())


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
# COVID-19, bone fractures and anaphylaxis are deliberately absent for now:
# eval_questions.json uses them as out-of-scope tests (ids 26, 28, 29). If you add
# them, update those entries too.

wiki_urls = [
    # Chronic diseases
    "https://en.wikipedia.org/wiki/Cancer",
    "https://en.wikipedia.org/wiki/Diabetes",
    "https://en.wikipedia.org/wiki/Hypertension",
    "https://en.wikipedia.org/wiki/Asthma",
    "https://en.wikipedia.org/wiki/Obesity",
    "https://en.wikipedia.org/wiki/Chronic_kidney_disease",
    "https://en.wikipedia.org/wiki/Cardiovascular_disease",
    "https://en.wikipedia.org/wiki/Stroke",
    "https://en.wikipedia.org/wiki/Chronic_obstructive_pulmonary_disease",

    # Common symptoms / acute conditions
    "https://en.wikipedia.org/wiki/Common_cold",
    "https://en.wikipedia.org/wiki/Influenza",
    "https://en.wikipedia.org/wiki/Migraine",
    "https://en.wikipedia.org/wiki/Fever",
    "https://en.wikipedia.org/wiki/Gastroenteritis",
    "https://en.wikipedia.org/wiki/Headache",
    "https://en.wikipedia.org/wiki/Cough",
    "https://en.wikipedia.org/wiki/Low_back_pain",
    "https://en.wikipedia.org/wiki/Allergy",
    "https://en.wikipedia.org/wiki/Dehydration",

    # Medications
    "https://en.wikipedia.org/wiki/Paracetamol",
    "https://en.wikipedia.org/wiki/Ibuprofen",
    "https://en.wikipedia.org/wiki/Antibiotic",
    "https://en.wikipedia.org/wiki/Insulin_(medication)",
    "https://en.wikipedia.org/wiki/Aspirin",
    "https://en.wikipedia.org/wiki/Statin",
    "https://en.wikipedia.org/wiki/Antidepressant",

    # Nutrition
    "https://en.wikipedia.org/wiki/Nutrition",
    "https://en.wikipedia.org/wiki/Vitamin_D",
    "https://en.wikipedia.org/wiki/Dietary_fiber",
    "https://en.wikipedia.org/wiki/Healthy_diet",
    "https://en.wikipedia.org/wiki/Carbohydrate",
    "https://en.wikipedia.org/wiki/Detoxification_(alternative_medicine)",
    "https://en.wikipedia.org/wiki/Exercise",

    # Mental health
    "https://en.wikipedia.org/wiki/Major_depressive_disorder",
    "https://en.wikipedia.org/wiki/Anxiety_disorder",
    "https://en.wikipedia.org/wiki/Sleep",
    "https://en.wikipedia.org/wiki/Stress_(biology)",
    "https://en.wikipedia.org/wiki/Mental_disorder",
    "https://en.wikipedia.org/wiki/Psychotherapy",
    "https://en.wikipedia.org/wiki/Insomnia",
    "https://en.wikipedia.org/wiki/Meditation",
]

custom_sentences = [
    "Regular exercise can help reduce the risk of cardiovascular diseases.",
    "Drinking enough water is essential for maintaining good health.",
    "Meditation and mindfulness help improve mental health.",
    "A diet rich in fruits and vegetables supports a healthy immune system.",
]

CUSTOM_SOURCE = "custom_health_tips"
CHUNK_PARAMS = {"target_words": 150, "overlap_sentences": 2}


def get_sources_fingerprint():
    """Hash of everything that determines the index contents: the sources, the
    embedding model, the chunking settings and the cleaning/chunking code itself.
    Any change makes load_or_build_index() rebuild instead of silently serving
    chunks produced by the old logic."""
    pipeline_code = "".join(
        inspect.getsource(fn)
        for fn in (extract_article, clean_text, split_into_sentences, chunk_by_sentences, indexable_text)
    )
    raw = json.dumps([
        wiki_urls, custom_sentences, EMBEDDING_MODEL, CHUNK_PARAMS, NOISE_SELECTORS,
        sorted(SKIPPED_SECTIONS), SENTENCE_ABBREVIATIONS, SENTENCE_BOUNDARY,
        MAX_SENTENCE_WORDS, pipeline_code,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()


def indexable_text(chunk):
    """The text that gets embedded and keyword-indexed. Prefixing the article and
    section tells search what a chunk is about even when its sentences only say
    "it" or "this condition"; the LLM still sees just chunk["text"]."""
    return " - ".join(p for p in (chunk["title"], chunk["section"], chunk["text"]) if p)


def build_corpus():
    """Clean and chunk every source. Returns (chunks, failed_urls), where chunks
    are {"text", "source", "title", "section"} dicts. Only called when there's no
    valid cache on disk."""
    all_chunks, failed = [], []

    print("\n--- Building corpus (scraping + chunking) ---")
    for url in wiki_urls:
        html = fetch_html(url)
        article = extract_article(html) if html else None
        if not article or not article["sections"]:
            print(f"[SKIP] {url} produced no usable text.")
            failed.append(url)
            continue
        title = article["title"] or url.rsplit("/", 1)[-1].replace("_", " ")
        seen = set()
        for section, text in article["sections"]:
            # Chunks stop at section boundaries so one chunk doesn't mix, say,
            # the end of "Treatment" with the start of "History".
            for chunk in chunk_by_sentences(text, **CHUNK_PARAMS):
                if chunk in seen or len(chunk.split()) < 12:
                    continue
                seen.add(chunk)
                all_chunks.append({"text": chunk, "source": url, "title": title, "section": section})
        print(f"[OK] {url} -> {len(seen)} chunks")

    # One chunk per tip: they're unrelated to each other, so joining them would
    # make every tip compete with the other three for the same embedding.
    for sentence in custom_sentences:
        all_chunks.append({"text": sentence, "source": CUSTOM_SOURCE, "title": "Health tips", "section": ""})

    print(f"--- Total chunks built: {len(all_chunks)} ---\n")

    if len(failed) == len(wiki_urls):
        print("*** WARNING: none of the Wikipedia URLs produced content. "
              f"Your corpus is effectively just the {len(custom_sentences)} custom sentences. "
              "Check the [WARN] messages above. ***\n")

    return all_chunks, failed


def read_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_index(manifest):
    """Returns (index, chunks), or None if the files on disk are inconsistent,
    which happens if an earlier build was interrupted part way through saving."""
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        return None
    cached_index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        cached_chunks = json.load(f)
    if cached_index.ntotal != len(cached_chunks) or len(cached_chunks) != manifest.get("total_chunks"):
        print("--- Cached files don't match each other — rebuilding index ---")
        return None
    print(f"--- Loaded {len(cached_chunks)} chunks from cache (built {manifest.get('built_at')}) ---\n")
    return cached_index, cached_chunks


def load_or_build_index():
    """Load the FAISS index + chunk metadata from disk if a valid cache exists
    for the current sources and pipeline; otherwise scrape/embed fresh and cache
    the result so the next restart can skip straight to loading."""
    current_hash = get_sources_fingerprint()
    manifest = read_manifest()

    if manifest and manifest.get("fingerprint") == current_hash:
        failed = manifest.get("failed_sources", [])
        # A page that failed last time (say, a rate limit) shouldn't stay missing
        # forever. Retry it, but if it still fails, the existing index is the
        # best we have and re-embedding wouldn't change anything.
        if not any([fetch_html(url) for url in failed]):
            if failed:
                print(f"*** WARNING: {len(failed)} sources are missing from the index: {failed} ***")
            cached = load_cached_index(manifest)
            if cached:
                return cached
        else:
            print("--- A previously failed source is now available — rebuilding index ---")
    elif manifest:
        print("--- Sources or pipeline changed since last run — rebuilding index ---")

    chunks, failed = build_corpus()
    embeddings = embedder.encode(
        [indexable_text(c) for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )
    embeddings = np.array(embeddings).astype("float32")

    new_index = faiss.IndexFlatIP(embeddings.shape[1])
    new_index.add(embeddings)

    # Persist to disk so the next startup can skip scraping + embedding entirely.
    # The manifest goes last: it's what marks the other two files as complete.
    faiss.write_index(new_index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "fingerprint": current_hash,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": EMBEDDING_MODEL,
            "total_chunks": len(chunks),
            "failed_sources": failed,
        }, f, indent=2)

    print(f"--- Cached index + chunks to '{CACHE_DIR}' for future restarts ---\n")
    return new_index, chunks


# ----------- Force a cache rebuild -----------
# Run `python app.py --rebuild` to wipe the cache and re-scrape everything from
# scratch — useful if a source's content has changed on the web even though your
# wiki_urls list hasn't, since the fingerprint alone wouldn't catch that.
# This has to happen before the index loads so that BM25 below is built from the
# new chunks. WERKZEUG_RUN_MAIN marks the debug reloader's child process, which
# gets the same argv and would otherwise wipe and rebuild a second time.

if "--rebuild" in sys.argv and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    shutil.rmtree(CACHE_DIR)
    os.makedirs(HTML_DIR)
    print("--- Cache cleared, rebuilding now ---")

index, all_chunks = load_or_build_index()

# ----------- Keyword (BM25) index -----------
# Built fresh every startup regardless of whether the FAISS index was loaded
# from cache — tokenizing the corpus is cheap, so there's no need to persist
# this separately.

print("--- Building BM25 keyword index ---")
tokenized_corpus = [tokenize(indexable_text(c)) for c in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)
print(f"--- BM25 index built over {len(all_chunks)} chunks ---\n")

# Lowered from 0.35 — all-MiniLM-L6-v2 is a general-purpose symmetric similarity
# model, not tuned for question-vs-passage retrieval, so genuine matches often
# score in the 0.15-0.30 range. Use the /debug endpoint below to see real scores
# for your own questions and adjust this number accordingly. Note this now
# applies to the COMBINED hybrid score, not raw cosine similarity — recheck it
# after switching to hybrid_search().
RELEVANCE_THRESHOLD = 0.15

# Weight given to semantic (embedding) score vs keyword (BM25) score when
# combining. 0.6 means 60% semantic, 40% keyword. Semantic search is generally
# better at understanding paraphrased/conceptual questions, while BM25 is
# better at exact terms (drug names, specific numbers/dosages) that embeddings
# can blur past — so this is a starting point, not a fixed answer. Tune it
# using /debug: if BM25-favoring matches (exact term hits) are getting buried
# by semantically-similar-but-wrong chunks, lower this; if keyword noise is
# winning over genuinely relevant chunks, raise it.
SEMANTIC_WEIGHT = 0.6

# How many chunks /ask passes to the LLM: about what fits in the context budget
# (see truncate_to_word_budget), so every source shown was actually read by the
# model. /debug and run_eval.py use the same value so the eval measures what users
# actually get.
TOP_K = 2


def hybrid_search(question, k=TOP_K, candidate_pool=20):
    """
    Combines semantic (embedding/FAISS) search with keyword (BM25) search.

    Why: pure semantic search can miss questions that hinge on an exact term —
    a specific drug name, a dosage number, a precise medical term — because
    the embedding model cares about overall meaning, not exact tokens. BM25
    is the opposite: it's strong on exact term overlap but has no notion of
    meaning, so it misses paraphrased questions. Combining both catches more
    of both cases than either alone.

    Returns the top-k combined results, plus the full candidate pool (used by
    /debug to show the score breakdown for tuning SEMANTIC_WEIGHT/threshold).
    """
    pool = min(candidate_pool, len(all_chunks))

    # --- Semantic candidates ---
    q_embed = embedder.encode([question], normalize_embeddings=True)
    q_embed = np.array(q_embed).astype("float32")
    sem_scores, sem_idxs = index.search(q_embed, pool)
    sem_scores, sem_idxs = sem_scores[0], sem_idxs[0]
    semantic_score_map = {int(i): float(s) for i, s in zip(sem_idxs, sem_scores)}

    # --- Keyword (BM25) candidates ---
    tokenized_query = tokenize(question)
    bm25_scores = bm25.get_scores(tokenized_query)  # one score per chunk in the corpus
    max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) else 0.0
    # Normalize BM25 scores to roughly [0, 1] so they're comparable to cosine
    # similarity. BM25 scores are unbounded and query-dependent, so this is a
    # per-query max-normalization rather than a fixed global scale.
    bm25_norm = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores
    bm25_top_idxs = np.argsort(bm25_scores)[-pool:][::-1]

    # --- Combine ---
    candidate_indices = set(semantic_score_map.keys()) | set(int(i) for i in bm25_top_idxs)

    combined = []
    for i in candidate_indices:
        sem_s = semantic_score_map.get(i, 0.0)
        bm25_s = float(bm25_norm[i])
        combined_score = SEMANTIC_WEIGHT * sem_s + (1 - SEMANTIC_WEIGHT) * bm25_s
        combined.append({
            "index": i,
            "semantic_score": sem_s,
            "bm25_score": bm25_s,
            "combined_score": combined_score,
        })

    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    return combined[:k], combined


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

    top_k, all_candidates = hybrid_search(user_question, k=5, candidate_pool=20)

    results = [
        {
            "rank": rank,
            "used_by_ask": rank <= TOP_K,
            "combined_score": round(c["combined_score"], 4),
            "semantic_score": round(c["semantic_score"], 4),
            "bm25_score": round(c["bm25_score"], 4),
            "source": all_chunks[c["index"]]["source"],
            "section": all_chunks[c["index"]]["section"],
            "text_preview": all_chunks[c["index"]]["text"][:200],
            "passes_current_threshold": c["combined_score"] >= RELEVANCE_THRESHOLD,
        }
        for rank, c in enumerate(top_k, start=1)
    ]

    return jsonify({
        "question": user_question,
        "total_chunks_in_index": len(all_chunks),
        "current_threshold": RELEVANCE_THRESHOLD,
        "top_k": TOP_K,
        "semantic_weight": SEMANTIC_WEIGHT,
        "top_matches": results,
    })


# ----------- API Endpoint -----------

def unique_sources(chunks):
    """Sources for the UI, best match first, each listed once. Custom tips have
    no page to link to, so their url is None."""
    sources = {}
    for c in chunks:
        url = None if c["source"] == CUSTOM_SOURCE else c["source"]
        sources.setdefault(c["source"], {"title": c["title"], "url": url})
    return list(sources.values())


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        top_k, _ = hybrid_search(user_question, candidate_pool=20)

        retrieved = [
            (all_chunks[c["index"]], c["combined_score"])
            for c in top_k
            if c["combined_score"] >= RELEVANCE_THRESHOLD
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
                },
                "sources": [],
            })

        context = "\n\n".join(f"[{c['title']}] {c['text']}" for c, _ in retrieved)
        context = truncate_to_word_budget(context)

        system_prompt = (
            "You are a helpful health assistant. Answer using ONLY the provided "
            "context. If the context does not fully answer the question, say so "
            "honestly instead of guessing. Do not include references, links, or "
            "markdown formatting in your answer."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_question}"

        with _llm_lock:
            llm = get_llm()
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

        with _llm_lock:
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
            "sources": unique_sources(c for c, _ in retrieved),
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
