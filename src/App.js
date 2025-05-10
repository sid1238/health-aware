import logo from './logo.svg';
import './App.css';

import React from "react";
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from "react-router-dom";
import "./App.css";
import Purpose from "./Purpose";
import Help from "./Help";

function App() {
  return (
    <Router>
      <div className="app-container">
        <Header />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/purpose" element={<Purpose />} />
          <Route path="/help" element={<Help />} />
        </Routes>
      </div>
    </Router>
  );
}

function Header() {
  const navigate = useNavigate();

  const scrollTo = (id) => {
    navigate("/");
    setTimeout(() => {
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
      }
    }, 100); // Delay ensures page loads first
  };

  return (
    <header className="header">
      <div className="logo">HealthBot</div>
      <nav>
        <button onClick={() => scrollTo("details")} className="nav-btn">Details</button>
        <button onClick={() => scrollTo("footer")} className="nav-btn">Contact</button>
        <Link to="/purpose">Purpose</Link>
        <Link to="/help">Help</Link>
      </nav>
    </header>
  );
}

function Home() {
  return (
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

      <footer id="footer" className="footer">
        <p><strong>Email:</strong> support@healthbot.ai</p>
        <p><strong>Contact:</strong> +1-800-HEALTHBOT</p>
        <p>© 2025 HealthBot. All rights reserved.</p>
      </footer>
    </main>
  );
}

export default App;