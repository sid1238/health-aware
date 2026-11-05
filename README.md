# HealthBot (health-aware)

HealthBot is a web portal for health awareness and myth-busting. It pairs a React site with a chatbot that answers health questions from a curated set of Wikipedia articles and lists the sources it used.

The chatbot runs on your own machine. A small language model writes the answers, so asking questions needs no API key and sends nothing to a paid service.

> **Not medical advice.** HealthBot provides general information and can be wrong. It isn't a substitute for a healthcare professional. In an emergency, call your local emergency number. See [Known limitations](#known-limitations).

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Using the API](#using-the-api)
- [Working with the knowledge base](#working-with-the-knowledge-base)
- [Tuning retrieval](#tuning-retrieval)
- [Measuring quality](#measuring-quality)
- [Running the tests](#running-the-tests)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Features

**The website**

- **Home:** common health conditions, prevention habits and popular health myths, each linking straight to a question for the chatbot
- **Purpose:** the project's mission, vision and values
- **Help:** the chat interface, with links to sources, plus a list of frequently asked questions
- Works at phone widths and supports dark mode

**The chatbot**

- **Knowledge base:** 41 Wikipedia articles on chronic diseases, symptoms, medications, nutrition and mental health, plus a few short health tips
- **Hybrid search:** combines meaning-based search with keyword search, so exact terms like drug names aren't missed
- **Re-ranking:** a second model picks the passages that best answer the question
- **Fallback answer:** says it doesn't have enough information when nothing relevant is found

**The evaluation tools**

- A set of 48 test questions, each noting which article should answer it
- A retrieval check that runs locally for free
- An answer judge that uses a stronger model to catch made-up claims and unsafe advice

## How it works

```
                     Question
                        |
        +---------------+---------------+
        |                               |
  Semantic search                 Keyword search
  (FAISS + MiniLM)                    (BM25)
        |                               |
        +---------------+---------------+
                        |
             Combined score: top 10 chunks
                        |
             Cross-encoder re-ranks them
                        |
        Is the best chunk relevant enough?
             |                     |
             no                   yes
             |                     |
      Fallback answer      Top 2 chunks go to Gemma 3 1B,
                           which answers from them only
                                   |
                           Answer + source links
```

| Part | Technology |
|---|---|
| Frontend | React 19, React Router, plain CSS |
| Backend | Flask |
| Answer model | Gemma 3 1B (`gemma-3-1b-it-q4_0.gguf`), run with llama.cpp |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Keyword search | BM25 (`rank_bm25`) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Answer judge (optional) | Claude Opus 5.5, through the Claude API |

## Project structure

```
health-aware/
├── backend/
│   ├── app.py                  # Flask API, scraping, chunking and retrieval
│   ├── eval_questions.json     # Test questions and their expected sources
│   ├── run_eval.py             # Checks that retrieval finds the right article
│   ├── judge_eval.py           # Has a stronger model judge the answers
│   ├── test_judge_eval.py      # Tests for the judge (no API key needed)
│   ├── rag_cache/              # Generated: saved pages, search index, chunks
│   ├── eval_runs/              # Generated: captured answers and verdicts
│   └── eval_results.json       # Generated: latest run_eval.py results
├── public/                     # Static files and the HTML template
├── src/
│   ├── App.js                  # Layout, navigation and the Home page
│   ├── Help.js                 # Chat page and FAQ
│   ├── Purpose.js              # Purpose page
│   ├── App.css                 # Site styles
│   └── App.test.js             # Frontend tests
└── package.json
```

## Getting started

### What you need

- **Node.js and npm.** Run on Node 24.
- **Python.** Run on Python 3.11.
- **About 2 GB of free disk space** for the models and cache.
- **A free [Hugging Face](https://huggingface.co) account**, because the Gemma model requires accepting a license.

`llama-cpp-python` may need a C/C++ compiler to install, depending on your system.

### 1. Get access to the Gemma model

1. Open the [model page](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf), then review and accept the license.
2. Create an access token in your Hugging Face settings. Read access is enough.
3. You'll log in with it in the next step.

### 2. Set up the backend

From the project root, in PowerShell:

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install flask flask-cors sentence-transformers faiss-cpu llama-cpp-python beautifulsoup4 requests rank_bm25
huggingface-cli login
```

On macOS or Linux, activate the environment with `source venv/bin/activate`.

### 3. Start the backend

```
python app.py
```

The API runs at `http://localhost:5000`. The first start takes longer because it has to:

- download the embedding and re-ranking models
- download the Wikipedia articles
- build the search index

Later starts load the saved index, which took about 16 seconds on a modest laptop. The Gemma model loads when the first question arrives, so that answer is slower too.

### 4. Start the website

In a second terminal, from the project root:

```
npm install
npm start
```

The site opens at `http://localhost:3000`. If that port is taken, choose another:

```powershell
$env:PORT=3100; npm start
```

If the backend runs somewhere other than `http://localhost:5000`, set `REACT_APP_API_URL` before starting the site.

## Using the API

### `POST /ask`

Answers a question.

```json
{ "question": "What are the side effects of ibuprofen?" }
```

```json
{
  "question": "What are the side effects of ibuprofen?",
  "answer": "Heartburn, nausea, indigestion, abdominal pain...",
  "sources": [
    { "title": "Ibuprofen", "url": "https://en.wikipedia.org/wiki/Ibuprofen" }
  ]
}
```

`url` is `null` for the built-in health tips, which have no page to link to. `sources` is empty when the fallback answer is used.

Two optional fields exist for the evaluation tools:

| Field | Default | Effect |
|---|---|---|
| `rerank` | `true` | Set to `false` to skip the re-ranker |
| `include_context` | `false` | Set to `true` to also return the text given to the model |

Answers took about 10 to 30 seconds each on a laptop CPU.

### `POST /debug`

Takes the same `question` field and shows how retrieval scored the top 5 chunks, without generating an answer. This is the quickest way to see why a question got a poor answer.

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5000/debug -ContentType 'application/json' -Body '{"question": "How do statins lower cholesterol?"}'
```

Each match includes its source, section, text preview and scores. `used_by_ask` shows whether it would reach the model.

## Working with the knowledge base

The search index is saved in `backend/rag_cache/`, so restarts don't have to download or process anything again. It rebuilds by itself when:

- you change the source list or the health tips
- you change the embedding model, the chunking settings or the cleaning code
- a page that failed to download earlier becomes available

### Adding a topic

Add the article's URL to `wiki_urls` in `backend/app.py` and restart. Downloaded pages are kept, so only the new one is fetched. Then add a question for it to `eval_questions.json` so the eval covers it.

COVID-19, bone fractures and anaphylaxis are left out on purpose, because the eval uses them as out-of-scope tests (questions 26, 28 and 29). If you add them, update those questions too.

### Refreshing the articles

The automatic checks can't tell when a Wikipedia article itself has been edited. To download everything again:

```
python app.py --rebuild
```

## Tuning retrieval

These settings are near the retrieval code in `backend/app.py`:

| Setting | Current value | What it does |
|---|---|---|
| `SEMANTIC_WEIGHT` | `0.6` | Balance of semantic search (60%) to keyword search (40%) |
| `RERANK_CANDIDATES` | `10` | How many chunks the re-ranker scores |
| `TOP_K` | `2` | How many chunks the model receives |
| `RELEVANCE_THRESHOLD` | `-4.0` | Best re-ranker score below this triggers the fallback answer |

Test any changes with the [quality tools](#measuring-quality).

**Be careful raising the context limit.** `truncate_to_word_budget` caps the context at 300 words because longer prompts made this model's answers fall apart: most answers turned to gibberish at 500 words. Try a few real questions before raising it or `TOP_K`.

## Measuring quality

The two tools answer different questions, so a good pipeline needs both.

| | `run_eval.py` | `judge_eval.py` |
|---|---|---|
| **Question it answers** | Did we find the right article? | Is the answer actually good? |
| **Runs on** | Your machine, free | The Claude API, paid |
| **Speed** | Quick, no answers generated | Slower, judges full answers |

### The test questions

Each entry in `backend/eval_questions.json` looks like this:

```json
{
  "id": 39,
  "question": "Can I take Tylenol if I have been drinking alcohol?",
  "expected_source": "https://en.wikipedia.org/wiki/Paracetamol",
  "topic": "medication",
  "note": "Brand name test: the article title is the generic name"
}
```

- `expected_source` is the article that should answer it. `null` means the question should find nothing and trigger the fallback answer.
- `also_acceptable` optionally lists other articles that could answer it.
- `note` optionally explains what the question is testing.

### Checking retrieval

With the backend running, from the `backend` folder:

```
python run_eval.py
```

This checks each question against the chunks `/ask` would really use, prints a summary by topic, and saves the details to `eval_results.json`.

### Judging answers

Judging an answer with the small model that wrote it is circular, because a model can't reliably catch its own mistakes. `judge_eval.py` uses a stronger one instead.

**Setup:** run `pip install anthropic`, then set `ANTHROPIC_API_KEY`. Capturing answers needs neither a key nor a running backend. Judging costs money, so add `--dry-run` first to see how many calls a command will make.

From the `backend` folder:

```
# 1. Save answers from each pipeline version (free and local)
python judge_eval.py capture rerank
python judge_eval.py capture hybrid --no-rerank

# 2. Find what's wrong with one version
python judge_eval.py score rerank

# 3. Choose between two versions
python judge_eval.py compare hybrid rerank
```

**`score`** rates each answer on four separate measures:

| Measure | What it checks |
|---|---|
| **Faithfulness** | Is each claim supported by the retrieved text? This catches made-up facts, so it matters most for a health bot. |
| **Relevance** | Does the answer address the question? |
| **Completeness** | Does it cover what the retrieved text offers? |
| **Safety** | Does it avoid diagnosing, risky instructions, and delaying urgent care? |

The report lists the specific answers with unsupported claims or safety problems. There's deliberately no single overall score, because a blended number could let a complete answer hide a made-up fact.

**`compare`** asks which of two answers is better for each question. Prefer it when choosing between versions, since judges are more reliable at relative comparisons than absolute scores. It also guards against known judge biases:

| Bias | How it's handled |
|---|---|
| **Position** (favouring the answer shown first) | Each pair is judged in both orders. If the verdict changes with the order, neither side gets the win. |
| **Verbosity** (favouring longer answers) | The judge is told length isn't quality, and the report checks how often longer answers win and shows similar-length pairs separately. |
| **Self-preference** (favouring its own style) | The judge never sees pipeline names, and the tool refuses to run if the judge and the answer model are the same family. |

The report ends with a significance test, which tells you whether a lead could just be noise.

**Good to know:**

- After changing the pipeline, capture under a **new label**. Re-running `capture` on an existing label keeps its saved answers.
- Commands can be stopped and re-run. Finished work is kept, and only new or changed answers are judged.
- Results are saved in `backend/eval_runs/`.

## Running the tests

Frontend, from the project root:

```
npm test
```

Answer judge, from the `backend` folder (needs `anthropic` installed, but no API key):

```
python -m unittest test_judge_eval
```

## Known limitations

- **Answers can be wrong.** The answer model is small and sometimes misreads its sources, even when retrieval finds the right text.
- **There's no safety routing yet.** Emergency, diagnosis and dosing questions go through the normal pipeline. They do match the articles, so the relevance check doesn't stop them, and the eval's out-of-scope group shows this. A severe allergic reaction question, for example, currently gets an unhelpful answer rather than advice to call emergency services.
- **The fallback answer rarely fires.** The threshold is deliberately loose and rests on very few examples.
- **Answers are slow** because the model runs on the CPU.
- **Each question stands alone.** The chat shows earlier questions, but the bot doesn't use them as context.
- **The knowledge base is limited to Wikipedia,** so it's only as current and accurate as the saved articles.

## Troubleshooting

| Problem | What to try |
|---|---|
| The first question fails with a 401 or "gated repo" error | Accept the [Gemma license](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf) and run `huggingface-cli login` |
| The site says it couldn't reach the server | Check that `python app.py` is running on port 5000 |
| `npm start` says port 3000 is in use | Choose another port, as shown in [Start the website](#4-start-the-website) |
| Startup warns that sources are missing | A page failed to download. It's retried at the next start, and the bot works without it until then |
| Answers look like gibberish | The prompt is probably too long. See [Tuning retrieval](#tuning-retrieval) |
| `judge_eval.py` says no credentials were found | Set `ANTHROPIC_API_KEY`. Answers you've already captured are kept |
| `judge_eval.py` reports `No module named 'anthropic'` | Run `pip install anthropic` in the backend environment |
