import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ marginRight: '10px', filter: 'drop-shadow(0px 2px 4px rgba(59,130,246,0.3))' }}>
            {/* Translucent background mountain / The 'A' Shape */}
            <path d="M12 3L4 19H20L12 3Z" fill="url(#algo-grad)" fillOpacity="0.15" stroke="#3b82f6" strokeWidth="2" strokeLinejoin="round"/>

            {/* Internal Neural Connections */}
            <path d="M8 11L16 11" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 3L12 19" stroke="#3b82f6" strokeWidth="2" strokeDasharray="3 3" strokeLinecap="round" opacity="0.6"/>

            {/* Multi-Colored Data Nodes */}
            <circle cx="12" cy="3" r="3" fill="#f59e0b" />  {/* Apex (Gold) */}
            <circle cx="4" cy="19" r="2.5" fill="#3b82f6" /> {/* Left Base (Blue) */}
            <circle cx="20" cy="19" r="2.5" fill="#3b82f6" /> {/* Right Base (Blue) */}
            <circle cx="8" cy="11" r="2.5" fill="#10b981" /> {/* Left Mid (Green) */}
            <circle cx="16" cy="11" r="2.5" fill="#8b5cf6" /> {/* Right Mid (Purple) */}
            <circle cx="12" cy="19" r="2.5" fill="#ef4444" /> {/* Center Base (Red) */}

            <defs>
              <linearGradient id="algo-grad" x1="12" y1="3" x2="12" y2="19" gradientUnits="userSpaceOnUse">
                <stop stopColor="#3b82f6" />
                <stop offset="1" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
          </svg>
          <span className="logo-text">AlgoAtlas</span>
        </Link>

        <div className="navbar-links">
          <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
            Algorithm Hub
          </Link>
          <Link to="/docs" className={location.pathname.startsWith('/docs') ? 'active' : ''}>
            Documentation
          </Link>
          <Link to="/about" className={location.pathname === '/about' ? 'active' : ''}>
            About Architecture
          </Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
