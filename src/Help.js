import React, { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import axios from "axios";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

const SUGGESTIONS = [
  "What are the early signs of a stroke?",
  "What are the side effects of ibuprofen?",
  "How does dietary fiber support digestive health?",
  "How does chronic stress affect the body?",
];

const FAQS = [
  {
    q: "How much water should I drink daily?",
    a: "Generally, 2 to 3 liters per day is recommended depending on your activity level.",
  },
  {
    q: "Can regular exercise help reduce stress?",
    a: "Yes, physical activity releases endorphins which help improve mood and reduce stress.",
  },
  {
    q: "Are mental health and physical health related?",
    a: "Absolutely. Poor mental health can lead to physical issues and vice versa.",
  },
  {
    q: "Is 8 hours of sleep necessary for everyone?",
    a: "Most adults need between 7–9 hours for optimal health and brain function.",
  },
  {
    q: "What are simple ways to stay healthy at home?",
    a: "Eat a balanced diet, stay active, hydrate well, and get enough rest.",
  },
];

// The answer comes from a model running on the local CPU, so a long wait is
// normal. Say so before anyone assumes the page has hung.
const SLOW_NOTICE_MS = 8000;

const errorMessage = (error) => {
  if (!error.response) {
    return "Couldn't reach the HealthBot server. Check that the backend is running (python app.py), then try again.";
  }
  return "Something went wrong while generating that answer. Please try again.";
};

function Sources({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources">
      <span className="sources__label">Sources</span>
      <ul className="sources__list">
        {sources.map(({ title, url }) => (
          <li key={title}>
            {url ? (
              <a href={url} target="_blank" rel="noreferrer" className="source-chip">{title}</a>
            ) : (
              <span className="source-chip">{title}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

const Help = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [slow, setSlow] = useState(false);
  const inFlight = useRef(false);
  const threadEnd = useRef(null);

  const ask = useCallback(async (text) => {
    const question = text.trim();
    if (!question || inFlight.current) return;

    inFlight.current = true;
    setLoading(true);
    setSlow(false);
    setQuery("");
    setTurns((prev) => [...prev, { question }]);
    const slowTimer = setTimeout(() => setSlow(true), SLOW_NOTICE_MS);

    let result;
    try {
      const res = await axios.post(`${API_URL}/ask`, { question });
      result = { answer: res.data.answer, sources: res.data.sources };
    } catch (error) {
      result = { error: errorMessage(error) };
    }

    clearTimeout(slowTimer);
    inFlight.current = false;
    setLoading(false);
    setTurns((prev) => [...prev.slice(0, -1), { question, ...result }]);
  }, []);

  // Questions linked from the home page arrive as ?q=. The param is cleared
  // straight away so that refreshing doesn't ask the same question again.
  const incoming = searchParams.get("q");
  useEffect(() => {
    if (!incoming) return;
    setSearchParams({}, { replace: true });
    ask(incoming);
  }, [incoming, ask, setSearchParams]);

  useEffect(() => {
    if (turns.length > 0) threadEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [turns, slow]);

  const handleSubmit = (e) => {
    e.preventDefault();
    ask(query);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) handleSubmit(e);
  };

  return (
    <main className="page-width help">
      <div className="help__intro">
        <h1 className="page-title">Ask HealthBot</h1>
        <p className="lede">Answers are drawn from a set of health articles, with the sources listed underneath.</p>
      </div>

      <section className="chat" aria-label="Conversation">
        {turns.length === 0 ? (
          <div className="chat__empty">
            <p>Try one of these, or ask your own question below.</p>
            <ul className="suggestions">
              {SUGGESTIONS.map((s) => (
                <li key={s}>
                  <button type="button" className="suggestion" onClick={() => ask(s)}>{s}</button>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <ol className="thread" aria-live="polite">
            {turns.map((turn, i) => (
              <li key={i} className="turn">
                <p className="bubble bubble--question">{turn.question}</p>
                {turn.answer && (
                  <div className="bubble bubble--answer">
                    <p>{turn.answer}</p>
                    <Sources sources={turn.sources} />
                  </div>
                )}
                {turn.error && <p className="bubble bubble--error" role="alert">{turn.error}</p>}
                {!turn.answer && !turn.error && (
                  <div className="bubble bubble--answer bubble--pending">
                    <span className="dots" aria-hidden="true"><i /><i /><i /></span>
                    <span>{slow ? "Still working. Answers can take up to a minute on this model." : "Looking through sources…"}</span>
                  </div>
                )}
              </li>
            ))}
            <li ref={threadEnd} aria-hidden="true" />
          </ol>
        )}

        <form className="composer" onSubmit={handleSubmit}>
          <label htmlFor="question" className="visually-hidden">Your health question</label>
          <textarea
            id="question"
            rows={2}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your health question here…"
          />
          <button type="submit" className="button" disabled={loading || !query.trim()}>
            {loading ? "Asking…" : "Ask"}
          </button>
        </form>
        <p className="chat__notice">
          General information, not a diagnosis. If this may be an emergency, call your local emergency number now.
        </p>
      </section>

      <section className="faq">
        <h2 className="section-title">Frequently Asked Questions</h2>
        {FAQS.map(({ q, a }) => (
          <details key={q} className="faq__item">
            <summary>{q}</summary>
            <p>{a}</p>
          </details>
        ))}
      </section>
    </main>
  );
};

export default Help;
