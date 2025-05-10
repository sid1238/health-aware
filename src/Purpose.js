import React from "react";
import "./App.css";

const Purpose = () => {
  return (
    <>
      <div className="mission-section">
        <h2>Our Mission</h2>
        <ul>
          <li><strong>Empowerment:</strong> Equip people with clear, trustworthy health information.</li>
          <li><strong>Accessibility:</strong> Make healthcare knowledge available anytime, anywhere.</li>
          <li><strong>Awareness:</strong> Promote informed decisions about physical and mental health.</li>
        </ul>
      </div>

      <div className="vision-section">
        <h2>Our Vision</h2>
        <ul>
          <li><strong>Innovation:</strong> Lead the future of AI-driven health education.</li>
          <li><strong>Support:</strong> Be a 24/7 companion for health concerns and myths.</li>
          <li><strong>Trust:</strong> Build confidence in self-care and healthy lifestyles.</li>
        </ul>
      </div>
    </>
  );
};

export default Purpose;