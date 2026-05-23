import React from 'react';
import { motion } from 'framer-motion';
import './ModelPage.css';

function About() {
  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <div className="model-header">
        <h1 className="model-title">About AlgoAtlas Architecture</h1>
      </div>

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

        <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
          <h2 style={{ fontSize: '1.5rem', color: '#1e293b', marginBottom: '1rem' }}>Project Mission</h2>
          <p style={{ color: '#4b5563', lineHeight: '1.6', fontSize: '1.05rem' }}>
            AlgoAtlas is designed to demystify the mathematics behind modern Artificial Intelligence. By bridging the gap between theoretical textbook equations and real-time visual feedback, this platform provides an interactive sandbox for 29 distinct algorithms spanning from baseline Supervised models all the way to complex Deep Reinforcement Learning and Transformer architectures.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
          <div style={{ backgroundColor: '#eff6ff', padding: '2rem', borderRadius: '8px', border: '1px solid #bfdbfe' }}>
            <h3 style={{ fontSize: '1.25rem', color: '#1e40af', marginBottom: '1rem' }}>The Frontend (React)</h3>
            <p style={{ color: '#1e3a8a', lineHeight: '1.6' }}>
              Built natively in React, the frontend utilizes HTML5 Canvas for high-performance physics rendering and interactive data painting. <strong>Recharts</strong> is integrated heavily to supply dynamic, multi-axis visualizations for algorithm validation curves and loss history.
            </p>
          </div>

          <div style={{ backgroundColor: '#f0fdf4', padding: '2rem', borderRadius: '8px', border: '1px solid #bbf7d0' }}>
            <h3 style={{ fontSize: '1.25rem', color: '#166534', marginBottom: '1rem' }}>The Backend (Python Flask)</h3>
            <p style={{ color: '#14532d', lineHeight: '1.6' }}>
              The API layer is powered by Flask. The mathematical engines are driven by <strong>NumPy</strong>, <strong>Scikit-Learn</strong>, and custom from-scratch implementations (e.g., the CartPole DQN and the Nano-GPT Transformer). Visual frames are generated server-side using <strong>Matplotlib</strong> headless rendering.
            </p>
          </div>
        </div>

      </div>
    </motion.div>
  );
}

export default About;
