import React from "react";
import "./App.css";

const SECTIONS = [
  {
    title: "Our Mission",
    points: [
      { title: "Empowerment", desc: "Equip people with clear, trustworthy health information." },
      { title: "Accessibility", desc: "Make healthcare knowledge available anytime, anywhere." },
      { title: "Awareness", desc: "Promote informed decisions about physical and mental health." },
    ],
  },
  {
    title: "Our Vision",
    points: [
      { title: "Innovation", desc: "Lead the future of AI-driven health education." },
      { title: "Support", desc: "Be a 24/7 companion for health concerns and myths." },
      { title: "Trust", desc: "Build confidence in self-care and healthy lifestyles." },
    ],
  },
  {
    title: "Our Values",
    points: [
      { title: "Empathy", desc: "Think about other people's needs and care for them." },
      { title: "Equality", desc: "Make health care accessible to everyone irrespective of their caste, country or religion." },
      { title: "Open Minded", desc: "Being open to diverse point of views." },
    ],
  },
];

const Purpose = () => {
  return (
    <main>
      <div className="hero hero--compact">
        <div className="page-width">
          <p className="eyebrow">Purpose</p>
          <h1 className="hero__title">Why HealthBot exists.</h1>
        </div>
      </div>

      {SECTIONS.map(({ title, points }, i) => (
        <section key={title} className={i % 2 === 1 ? "band band--tinted" : "band"}>
          <div className="page-width split">
            <h2 className="split__heading">{title}</h2>
            <ol className="points">
              {points.map((point, n) => (
                <li key={point.title} className="point">
                  <span className="point__number" aria-hidden="true">{n + 1}</span>
                  <div>
                    <h3 className="point__title">{point.title}</h3>
                    <p className="point__desc">{point.desc}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </section>
      ))}
    </main>
  );
};

export default Purpose;
