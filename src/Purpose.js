import React from "react";
import "./App.css";

const Purpose = () => {
  return (
    <>
      <div className="mission-section">
        <h2>Our Mission</h2>
        <ol className="numbered-list">
          <li>
            <div className="point-title">Empowerment</div>
            <div className="point-desc">Equip people with clear, trustworthy health information.</div>
          </li>
          <li>
            <div className="point-title">Accessibility</div>
            <div className="point-desc">Make healthcare knowledge available anytime, anywhere.</div>
          </li>
          <li>
            <div className="point-title">Awareness</div>
            <div className="point-desc">Promote informed decisions about physical and mental health.</div>
          </li>
        </ol>
      </div>

      <div className="vision-section">
  <h2>Our Vision</h2>
  <div className="vision-list-wrapper">
    <ol className="numbered-list">
      <li>
        <div className="point-title">Innovation</div>
        <div className="point-desc">Lead the future of AI-driven health education.</div>
      </li>
      <li>
        <div className="point-title">Support</div>
        <div className="point-desc">Be a 24/7 companion for health concerns and myths.</div>
      </li>
      <li>
        <div className="point-title">Trust</div>
        <div className="point-desc">Build confidence in self-care and healthy lifestyles.</div>
      </li>
    </ol>
  </div>
</div>

      <div className="values-section">
        <h2>Our Values</h2>
        <ol className="numbered-list">
          <li>
            <div className="point-title">Empathy</div>
            <div className="point-desc">Think about other people's needs and care for them.</div>
          </li>
          <li>
            <div className="point-title">Equality</div>
            <div className="point-desc">Make health care accessible to everyone irrespective of their caste, country or religion.</div>
          </li>
          <li>
            <div className="point-title">Open Minded</div>
            <div className="point-desc">Being open to diverse point of views.</div>
          </li>
        </ol>
      </div>
    </>
  );
};

      
export default Purpose;