import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { docsData } from '../data/docsData';
import './ModelPage.css';

function Docs() {
  const navigate = useNavigate();
  const location = useLocation();
  const [activeAlgo, setActiveAlgo] = useState(docsData[0].algorithms[0].id);

  // If a user clicks an <InfoButton> on a model page, it passes the algoId in the URL state!
  useEffect(() => {
    if (location.state && location.state.activeId) {
      setActiveAlgo(location.state.activeId);
    }
  }, [location]);

  // Find the currently active algorithm data
  const currentDoc = docsData.flatMap(cat => cat.algorithms).find(a => a.id === activeAlgo) || docsData[0].algorithms[0];

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ maxWidth: '1400px' }}>
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <span>&larr; Back to Hub</span>
        </button>
        <h1 className="model-title">Algorithm Documentation</h1>
      </div>

      <div style={{ display: 'flex', gap: '2rem', marginTop: '2rem', alignItems: 'flex-start' }}>

        {/* SIDEBAR NAVIGATION */}
        <div style={{ width: '300px', flexShrink: 0, backgroundColor: 'white', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', position: 'sticky', top: '2rem', maxHeight: '80vh', overflowY: 'auto' }}>
          {docsData.map((category, idx) => (
            <div key={idx} style={{ marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: '#9ca3af', fontWeight: 'bold', marginBottom: '0.75rem', letterSpacing: '0.05em' }}>
                {category.category}
              </h3>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {category.algorithms.map(algo => (
                  <li key={algo.id} style={{ marginBottom: '0.25rem' }}>
                    <button
                      onClick={() => setActiveAlgo(algo.id)}
                      style={{
                        width: '100%', textAlign: 'left', padding: '0.5rem 0.75rem', border: 'none', borderRadius: '4px', cursor: 'pointer',
                        backgroundColor: activeAlgo === algo.id ? '#eff6ff' : 'transparent',
                        color: activeAlgo === algo.id ? '#2563eb' : '#4b5563',
                        fontWeight: activeAlgo === algo.id ? '600' : '400',
                        transition: 'all 0.2s'
                      }}
                    >
                      {algo.title}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* MAIN CONTENT AREA */}
        <div style={{ flex: 1, backgroundColor: 'white', padding: '3rem', borderRadius: '8px', border: '1px solid #e5e7eb', minHeight: '80vh' }}>
          <h1 style={{ fontSize: '2.5rem', color: '#0f172a', marginBottom: '1rem' }}>{currentDoc.title}</h1>

          <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
            <span style={{ backgroundColor: '#f1f5f9', color: '#475569', padding: '4px 12px', borderRadius: '20px', fontSize: '0.85rem', fontWeight: '600' }}>
              {docsData.find(c => c.algorithms.some(a => a.id === currentDoc.id)).category}
            </span>
            <button onClick={() => navigate(currentDoc.path)} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '20px', padding: '4px 16px', fontSize: '0.85rem', fontWeight: 'bold', cursor: 'pointer' }}>
              Launch Visualizer &rarr;
            </button>
          </div>

          <div className='docs-content' dangerouslySetInnerHTML={{ __html: currentDoc.content }} />
        </div>

      </div>
    </motion.div>
  );
}

export default Docs;
