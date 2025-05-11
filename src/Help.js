import React, { useState } from "react";
import "./App.css";

const Help = () => {
  const [query, setQuery] = useState("");

  const handleChange = (e) => {
    setQuery(e.target.value);
  };

  return (
    <div className="help-container">
      <h2>Need Help?</h2>
      <p>Enter your query below and we’ll try to assist you:</p>

      <input
        type="text"
        value={query}
        onChange={handleChange}
        className="query-input"
        placeholder="Type your question here..."
      />

      <div className="faq-section">
        <h3>Frequently Asked Questions</h3>
        <ul className="faq-list">
          <li>
            <strong>1. What is this health chatbot for?</strong><br />
            Our chatbot helps answer common health-related queries and busts popular myths.
          </li>
          <li>
            <strong>2. Is the chatbot a replacement for a doctor?</strong><br />
            No, it offers general information and suggestions, but it’s not a substitute for medical advice.
          </li>
          <li>
            <strong>3. Can I use this service for mental health questions?</strong><br />
            Yes, we cover both physical and mental health awareness and common concerns.
          </li>
          <li>
            <strong>4. How can I contact human support?</strong><br />
            You can scroll to the footer and use the provided email or phone number.
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Help;
