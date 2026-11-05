import React, { useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  NavLink,
  useLocation,
  useNavigate,
} from "react-router-dom";
import "./App.css";
import Purpose from "./Purpose";
import Help from "./Help";

const CONDITIONS = [
  { name: "Cardiovascular Diseases", question: "What are the main risk factors for cardiovascular disease?" },
  { name: "Diabetes", question: "What is the difference between type 1 and type 2 diabetes?" },
  { name: "Respiratory Diseases", question: "What are the symptoms of chronic obstructive pulmonary disease?" },
  { name: "Mental Health Disorders", question: "What are the most common mental health disorders?" },
];

const REMEDIES = [
  "Regular exercise and hydration",
  "Balanced diet rich in vegetables and fruits",
  "Daily mindfulness or meditation",
  "Good sleep hygiene",
];

const MYTHS = [
  {
    heading: "Myths About Physical Health",
    items: [
      { myth: "You need to sweat to get a good workout.", question: "Do you need to sweat to get a good workout?" },
      { myth: "Detox diets cleanse your body.", question: "Do detox diets actually remove toxins from the body?" },
      { myth: "Carbs are always bad.", question: "Are carbohydrates always bad for you?" },
    ],
  },
  {
    heading: "Myths About Mental Health",
    items: [
      { myth: "Mental health issues are a sign of weakness.", question: "Are mental health issues a sign of weakness?" },
      { myth: "Therapy is only for serious problems.", question: "Is therapy only for people with serious mental health problems?" },
      { myth: "Children don’t experience mental health problems.", question: "Can children have mental health problems?" },
    ],
  },
];

const askLink = (question) => `/help?q=${encodeURIComponent(question)}`;

function App() {
  return (
    <Router>
      <ScrollToTop />
      <a className="skip-link" href="#main">Skip to content</a>
      <Header />
      <div id="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/purpose" element={<Purpose />} />
          <Route path="/help" element={<Help />} />
        </Routes>
      </div>
      <Footer />
    </Router>
  );
}

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

function Header() {
  const navigate = useNavigate();
  const { pathname } = useLocation();

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  const goToDetails = () => {
    if (pathname !== "/") navigate("/");
    // On another page, wait a tick for Home to render before scrolling.
    setTimeout(() => scrollTo("details"), pathname === "/" ? 0 : 100);
  };

  return (
    <header className="site-header">
      <div className="site-header__inner">
        <Link to="/" className="wordmark">HealthBot</Link>
        <nav className="site-nav" aria-label="Main">
          <button type="button" onClick={goToDetails} className="site-nav__item">Details</button>
          <button type="button" onClick={() => scrollTo("footer")} className="site-nav__item">Contact</button>
          <NavLink to="/purpose" className="site-nav__item">Purpose</NavLink>
          <NavLink to="/help" className="site-nav__item">Help</NavLink>
        </nav>
      </div>
    </header>
  );
}

function Home() {
  const navigate = useNavigate();
  const [question, setQuestion] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) navigate(askLink(question.trim()));
  };

  return (
    <main>
      <section className="hero">
        <div className="page-width">
          <p className="eyebrow">Welcome to HealthBot</p>
          <h1 className="hero__title">Your AI-powered assistant for health awareness and myth-busting.</h1>
          <form className="hero__form" onSubmit={handleSubmit} role="search">
            <label htmlFor="home-question" className="visually-hidden">Ask a health question</label>
            <input
              id="home-question"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a health question…"
              autoComplete="off"
            />
            <button type="submit" className="button">Ask</button>
          </form>
        </div>
      </section>

      <section id="details" className="band">
        <div className="page-width split">
          <h2 className="split__heading">Most Prevalent Health Conditions</h2>
          <ul className="link-rows">
            {CONDITIONS.map(({ name, question: q }) => (
              <li key={name}>
                <Link to={askLink(q)} className="link-row">
                  <span>{name}</span>
                  <span className="link-row__cue">Ask <span aria-hidden="true">→</span></span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="band band--tinted">
        <div className="page-width split">
          <h2 className="split__heading">Home Remedies for Prevention</h2>
          <ol className="habit-grid">
            {REMEDIES.map((remedy, i) => (
              <li key={remedy} className="habit">
                <span className="habit__number" aria-hidden="true">{i + 1}</span>
                {remedy}
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="band">
        <div className="page-width myth-columns">
          {MYTHS.map(({ heading, items }) => (
            <div key={heading}>
              <h2 className="myth-columns__heading">{heading}</h2>
              <ul className="myth-list">
                {items.map(({ myth, question: q }) => (
                  <li key={myth} className="myth">
                    <span className="myth__tag">Myth</span>
                    <p className="myth__text">“{myth}”</p>
                    <Link to={askLink(q)} className="myth__link">
                      Check this with HealthBot <span aria-hidden="true">→</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function Footer() {
  return (
    <footer id="footer" className="site-footer">
      <div className="page-width site-footer__inner">
        <div>
          <p className="wordmark">HealthBot</p>
          <p className="site-footer__note">
            General health information, not medical advice. In an emergency, call your local emergency number.
          </p>
        </div>
        <div className="site-footer__contact">
          <p><strong>Email:</strong> <a href="mailto:support@healthbot.ai">support@healthbot.ai</a></p>
          <p><strong>Contact:</strong> +1-800-HEALTHBOT</p>
          <p className="site-footer__legal">© {new Date().getFullYear()} HealthBot. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}

export default App;
