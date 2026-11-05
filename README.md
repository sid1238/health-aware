# health-aware
## This is a portal that improves awareness with regards to health

# The installation procedures are

1. Install node.js and npm latest version.

2. From the project root, run `npm install`, then `npm start`. If port 3000 is taken, use `PORT=3100 npm start`.

3. Create a virtual environment within the backend folder
    Use the command python3 -m venv <environment name>
    Move till the environment path and Use the command environment path/Scripts/Activate

4. Run the command pip install flask flask-cors sentence-transformers faiss-cpu llama-cpp-python beautifulsoup4 requests rank_bm25

5. Run the command python app.py

# Working with the knowledge base

The first start downloads the articles in `wiki_urls` and builds the search index, which takes a few minutes. It is saved to `backend/rag_cache/`, so later starts load it straight away. It rebuilds automatically when the sources or the cleaning/chunking code change.

To pick up edits made to the Wikipedia articles themselves, download everything again:

    python app.py --rebuild

# Measuring retrieval quality

With the backend running, from the backend folder:

    python run_eval.py

This checks each question in `backend/eval_questions.json` against the chunks that `/ask` would give the model, and writes the details to `backend/eval_results.json`. To see the scores behind one question, POST `{"question": "..."}` to `http://localhost:5000/debug`.

Each question lists the page that should answer it in `expected_source`, plus any other valid pages in `also_acceptable`. `expected_source: null` means the question should find nothing and trigger the fallback answer.

# Measuring answer quality

`run_eval.py` only checks that the right page was found. `judge_eval.py` has a stronger model judge the answers themselves, because the small model that writes them can't reliably catch its own mistakes.

It needs `pip install anthropic` and an `ANTHROPIC_API_KEY`. Generating answers is free and local. Judging calls the Claude API and costs money, so add `--dry-run` first to see how many calls it will make.

From the backend folder:

    python judge_eval.py capture rerank
    python judge_eval.py capture hybrid --no-rerank
    python judge_eval.py score rerank
    python judge_eval.py compare hybrid rerank

- `capture` saves each pipeline's answers, and the context behind them, to `backend/eval_runs/`. After changing the pipeline, capture under a new label, so you can compare the old and new versions.
- `score` rates one pipeline on faithfulness, relevance, completeness and safety. Use it to find what's wrong: it lists the answers with unsupported claims or safety problems.
- `compare` asks which of two pipelines' answers is better for each question. Use it to choose between versions, since judges are more reliable at relative comparisons than absolute scores. Each pair is shown in both orders, and the report checks for position and length bias.

Commands can be re-run: finished work is kept, and only new or changed answers are judged. To run the tests for the judge, which need no API key, use `python -m unittest test_judge_eval`.
