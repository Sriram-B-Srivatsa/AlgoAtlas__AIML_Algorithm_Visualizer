import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-section">
          <h3>AlgoAtlas</h3>
          <p>The Interactive Machine Learning Explorer.</p>
        </div>

        <div className="footer-section">
          <h3>Navigation</h3>
          <ul>
            <li><Link to="/">Algorithm Hub</Link></li>
            <li><Link to="/docs">Documentation</Link></li>
            <li><Link to="/about">System Architecture</Link></li>
          </ul>
        </div>

        <div className="footer-section">
          <h3>Developers</h3>
          <ul>
            <li>
              <a href="https://github.com/Sriram-B-Srivatsa/AlgoAtlas__AIML_Algorithm_Visualizer" target="_blank" rel="noreferrer">
                Source Code (GitHub)
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div className="footer-bottom">
        <p>&copy; {new Date().getFullYear()} AlgoAtlas. All rights reserved.</p>
      </div>
    </footer>
  );
}

export default Footer;
