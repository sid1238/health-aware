import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const Help = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const handleChange = (e) => setQuery(e.target.value);

  const handleSubmit = async () => {
    if (!query.trim()) return;

    try {
      const res = await axios.post("http://localhost:5000/ask", {
        question: query,
      });
      setResponse(res.data.answer);
    } catch (error) {
      setResponse("There was an error fetching the answer.");
    }
  };

  return (
    <div className="help-page">
      <h2>Health Chatbot - Ask a Question</h2>

      <input
        type="text"
        value={query}
        onChange={handleChange}
        className="query-input"
        placeholder="Type your health question here..."
      />

      <button onClick={handleSubmit} className="submit-btn">
        Ask
      </button>

      {response && (
        <div className="response-box">
          <h3>Answer:</h3>
          <p>{response}</p>
        </div>
      )}

      <div className="faq-section">
        <h2>Frequently Asked Questions</h2>
        <ul>
          <li><strong>Q:</strong> How much water should I drink daily?<br />
              <strong>A:</strong> Generally, 2 to 3 liters per day is recommended depending on your activity level.</li>

          <li><strong>Q:</strong> Can regular exercise help reduce stress?<br />
              <strong>A:</strong> Yes, physical activity releases endorphins which help improve mood and reduce stress.</li>

          <li><strong>Q:</strong> Are mental health and physical health related?<br />
              <strong>A:</strong> Absolutely. Poor mental health can lead to physical issues and vice versa.</li>

          <li><strong>Q:</strong> Is 8 hours of sleep necessary for everyone?<br />
              <strong>A:</strong> Most adults need between 7–9 hours for optimal health and brain function.</li>

          <li><strong>Q:</strong> What are simple ways to stay healthy at home?<br />
              <strong>A:</strong> Eat a balanced diet, stay active, hydrate well, and get enough rest.</li>
        </ul>
      </div>
    </div>
  );
};

export default Help;