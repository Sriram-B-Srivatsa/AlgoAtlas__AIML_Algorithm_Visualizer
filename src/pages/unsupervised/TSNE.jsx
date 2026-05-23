import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function TSNE() {
  const navigate = useNavigate();
  const[loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [datasetType, setDatasetType] = useState('swiss');
  const[sampleCount, setSampleCount] = useState(300);
  const [perplexity, setPerplexity] = useState(30);
  const [results, setResults] = useState(null);

  const [playback, setPlayback] = useState({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    let interval;
    if (playback.active && playback.isPlaying) {
        interval = setInterval(() => {
            setPlayback(prev => {
                if (prev.currentIndex >= prev.frames.length - 1) {
                    clearInterval(interval); return { ...prev, isPlaying: false };
                }
                return { ...prev, currentIndex: prev.currentIndex + 1 };
            });
        }, 800);
    }
    return () => clearInterval(interval);
  },[playback.active, playback.isPlaying]);

  const togglePlayback = () => {
      setPlayback(p => {
          if (!p.isPlaying && p.currentIndex >= p.frames.length - 1) return { ...p, currentIndex: 0, isPlaying: true };
          return { ...p, isPlaying: !p.isPlaying };
      });
  };

  const trainModel = () => {
    setLoading(true); setError(null);
    axios.post(`${apiUrl}/tsne/train`, { parameters: { dataset_type: datasetType, count: sampleCount, perplexity: perplexity } })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
    })
    .catch(err => setError("Failed to run t-SNE."))
    .finally(() => setLoading(false));
  };

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">t-SNE </h1>
      </div>

      <p className="model-description">t-SNE takes incredibly complex, tangled 3D (or 100D) data and mathematically crushes it down into a beautiful, separated 2D map.<InfoButton algoId="tsne" /></p>
      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          {/* LEFT COLUMN: Input Explanations */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">High-Dimensional Data</h2>
            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}>
                <p style={{ color: '#4b5563', lineHeight: '1.5', marginBottom: '1.5rem' }}>
                    <strong>Why is there no interactive drawing canvas here?</strong><br/>
                    t-SNE is a Dimensionality Reduction tool meant for data with 3, 10, or even 100 dimensions! Since you can only click in 2D on a flat screen, running t-SNE on it wouldn't do anything meaningful. Instead, select a complex <strong>3D</strong> structure below to generate mathematically on the backend.
                </p>

                <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>3D Dataset Shape</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
                    <button onClick={() => setDatasetType('swiss')} style={{ padding: '0.75rem', backgroundColor: datasetType === 'swiss' ? '#3b82f6' : '#eff6ff', color: datasetType === 'swiss' ? 'white' : '#1e40af', border: '1px solid #bfdbfe', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>Swiss Roll</button>
                    <button onClick={() => setDatasetType('scurve')} style={{ padding: '0.75rem', backgroundColor: datasetType === 'scurve' ? '#10b981' : '#f0fdf4', color: datasetType === 'scurve' ? 'white' : '#166534', border: '1px solid #bbf7d0', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>S-Curve</button>
                    <button onClick={() => setDatasetType('blobs')} style={{ padding: '0.75rem', backgroundColor: datasetType === 'blobs' ? '#8b5cf6' : '#faf5ff', color: datasetType === 'blobs' ? 'white' : '#6b21a8', border: '1px solid #e9d5ff', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>3D Blobs</button>
                </div>
                <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>
                    <strong>Swiss Roll & S-Curve:</strong> Complex 3D manifolds (like a rolled-up piece of paper). Standard PCA fails here, but t-SNE can unroll them!<br/>
                    <strong>3D Blobs:</strong> Standard clusters floating in 3D space.
                </p>

                <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Number of Samples: {sampleCount}</h3>
                <input type="range" min="100" max="600" step="50" value={sampleCount} onChange={(e) => setSampleCount(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                    <span>100 (Fast, sparse shape)</span><span>600 (Slow, dense shape)</span>
                </div>
                <p style={{fontSize: '0.85rem', color: '#6b7280'}}>More points create a clearer 3D shape, but the algorithm will take longer to compute the physics for every point pair!</p>
            </div>
          </div>

          {/* RIGHT COLUMN: Controls */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Algorithm Controls</h2>
            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Perplexity: {perplexity}</h3>
              <input type="range" min="5" max="50" step="5" value={perplexity} onChange={(e) => setPerplexity(parseInt(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem', marginTop: '0.5rem' }}>
                  <span>5 (Focus on local neighbors)</span><span>50 (Focus on global picture)</span>
              </div>
              <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>Perplexity is like a "knob" that balances attention between local details and the big global picture.</p>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
                <button onClick={trainModel} disabled={loading} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', marginBottom: '1rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}>
                    {loading ? (
                        <>
                            <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle>
                                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path>
                            </svg>
                            Crunching Dimensions...
                        </>
                    ) : 'Run t-SNE Unrolling'}
                </button>

                {/* EXPLANATION BOX */}
                <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                    <h4 style={{ margin: '0 0 0.5rem 0', color: '#166534' }}>How t-SNE Works</h4>
                    <p style={{ fontSize: '0.9rem', color: '#15803d', margin: 0, lineHeight: '1.5' }}>
                        t-SNE does not learn a mathematical formula (like PCA). It acts like a physics simulation! It calculates the gravitational "pull" between every point and its nearest neighbors in high-dimensional space, and then tries to replicate that exact same pull on a flat 2D map.
                    </p>
                </div>
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Perplexity Animation</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>Perplexity: {playback.frames[playback.currentIndex]?.perplexity}</span>
                        </div>
                        {/* FIX: Set height to auto, width to 100%, object-fit contain to eliminate whitespace */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e5e7eb', padding: '0.5rem', marginBottom: '1rem' }}>
                            <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="t-SNE" style={{ width: '100%', height: 'auto', objectFit: 'contain' }}/>
                        </div>
                        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                            <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer' }}>{playback.isPlaying ? '⏸' : '▶'}</button>
                            <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1 }} />
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Final KL Divergence: {results.final_kl.toFixed(4)}</p>
                        </div>
                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Original 3D Shape</h4>
                            <img src={`data:image/png;base64,${results.original_image}`} alt="3D Space" style={{ width: '100%', borderRadius: '4px' }}/>
                            <p style={{ color: '#4b5563', fontSize: '0.85rem', lineHeight: '1.6', marginTop: '1rem' }}>
                                Look at the dataset. It is impossible to draw a flat 2D line through it. But the t-SNE animation (Left) perfectly unrolled it!
                            </p>
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>KL Divergence vs Perplexity</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <LineChart data={results.loss_history}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="perplexity" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line type="monotone" dataKey="kl_divergence" stroke="#ef4444" name="KL Divergence (Loss)" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center'}}>KL Divergence measures how much information was lost when crushing 3D to 2D. Lower is better!</p>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Data Distribution</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <BarChart data={results.cluster_sizes}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="cluster" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Bar dataKey="size" name="Number of Points" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default TSNE;
