import logo from './logo.svg';
import './App.css';

import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";

function App() {
  return (
    <div className="app-container">
      <header className="header">
        <div className="logo">HealthBot</div>
        <nav>
          <a href="#details">Details</a>
          <a href="#footer">Contact</a>
        </nav>
      </header>

      <main>
        <section className="section green">
          <h1>Welcome to HealthBot</h1>
          <p>Your AI-powered assistant for health awareness and myth-busting.</p>
        </section>

        <section id="details" className="section red">
          <h2>Most Prevalent Health Conditions</h2>
          <ul>
            <li>Cardiovascular Diseases</li>
            <li>Diabetes</li>
            <li>Respiratory Diseases</li>
            <li>Mental Health Disorders</li>
          </ul>
        </section>

        <section className="section green">
          <h2>Home Remedies for Prevention</h2>
          <ul>
            <li>Regular exercise and hydration</li>
            <li>Balanced diet rich in vegetables and fruits</li>
            <li>Daily mindfulness or meditation</li>
            <li>Good sleep hygiene</li>
          </ul>
        </section>

        <section className="section red">
          <h2>Myths About Physical Health</h2>
          <ul>
            <li>“You need to sweat to get a good workout.”</li>
            <li>“Detox diets cleanse your body.”</li>
            <li>“Carbs are always bad.”</li>
          </ul>
        </section>

        <section className="section green">
          <h2>Myths About Mental Health</h2>
          <ul>
            <li>“Mental health issues are a sign of weakness.”</li>
            <li>“Therapy is only for serious problems.”</li>
            <li>“Children don’t experience mental health problems.”</li>
          </ul>
        </section>
      </main>

      <footer id="footer" className="footer">
        <p><strong>Email:</strong> support@healthbot.ai</p>
        <p><strong>Contact:</strong> +1-800-HEALTHBOT</p>
        <p>© 2025 HealthBot. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
