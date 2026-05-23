import React from 'react';
import { useNavigate } from 'react-router-dom';

function InfoButton({ algoId }) {
  const navigate = useNavigate();

  const handleClick = (e) => {
    e.stopPropagation(); // Prevents clicking the button from triggering other events

    navigate('/docs', { state: { activeId: algoId } });
  };

  return (
    <button
      onClick={handleClick}
      style={{
        background: 'none',
        border: 'none',
        color: '#3b82f6',
        cursor: 'pointer',
        marginLeft: '10px',
        padding: '0',
        display: 'inline-flex',
        alignItems: 'center',
        verticalAlign: 'middle',
        transition: 'transform 0.2s ease, color 0.2s ease'
      }}
      onMouseOver={(e) => { e.currentTarget.style.color = '#2563eb'; e.currentTarget.style.transform = 'scale(1.1)'; }}
      onMouseOut={(e) => { e.currentTarget.style.color = '#3b82f6'; e.currentTarget.style.transform = 'scale(1)'; }}
      title="Read the Mathematical Documentation"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>
    </button>
  );
}

export default InfoButton;
