/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function Diffusion() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [timesteps, setTimesteps] = useState(50);
  const [noiseAmount, setNoiseAmount] = useState(1.0);
  const [datasetType, setDatasetType] = useState('s_curve');
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
        }, 150); // Very fast for smooth denoising video
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
    axios.post(`${apiUrl}/diffusion/train`, {
        parameters: { timesteps, noise: noiseAmount, dataset_type: datasetType }
    })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
    })
    .catch(err => setError("Failed to run Diffusion Model."))
    .finally(() => setLoading(false));
  };

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Diffusion Models </h1>
      </div>

      <p className="model-description">Diffusion Models power Midjourney, DALL-E, and Sora. The math works by taking a perfectly good image and destroying it with pure TV static. The AI then acts as a "Denoising Engine", mathematically reversing the process to generate brand new images out of thin air!<InfoButton algoId="diffusion" /></p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Generative AI Process</h2>
            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1rem', height: '100%' }}>
                <p style={{ color: '#4b5563', lineHeight: '1.6', fontSize: '0.95rem', marginBottom: '1.5rem' }}>
                    <strong>1. The Forward Process:</strong> We take a perfect dataset (like an S-Curve) and mathematically destroy it by adding pure static noise.<br/><br/>
                    <strong>2. The Reverse Process:</strong> The AI acts as a "Denoising Engine". It looks at pure static noise, calculates the "Score Function", and physically pushes the pixels back into the original shape!
                </p>
                <div style={{ padding: '1.5rem', backgroundColor: '#eff6ff', borderRadius: '8px', border: '1px dashed #3b82f6', textAlign: 'center' }}>
                    <h4 style={{ color: '#1d4ed8', margin: '0 0 10px 0', fontSize: '1.2rem' }}>Markov Chain Denoising</h4>
                    <span style={{ fontFamily: 'serif', fontSize: '1.3rem', fontStyle: 'italic', fontWeight: 'bold' }}>x_{'{t-1}'} = x_t - ε_θ(x_t, t)</span>
                </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Generation Controls</h2>
            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>

              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Target Shape</h3>
              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <button onClick={() => setDatasetType('s_curve')} style={{ padding: '0.5rem', backgroundColor: datasetType === 's_curve' ? '#3b82f6' : '#eff6ff', color: datasetType === 's_curve' ? 'white' : '#1e40af', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>S-Curve</button>
                  <button onClick={() => setDatasetType('moons')} style={{ padding: '0.5rem', backgroundColor: datasetType === 'moons' ? '#3b82f6' : '#eff6ff', color: datasetType === 'moons' ? 'white' : '#1e40af', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Moons</button>
                  <button onClick={() => setDatasetType('circles')} style={{ padding: '0.5rem', backgroundColor: datasetType === 'circles' ? '#3b82f6' : '#eff6ff', color: datasetType === 'circles' ? 'white' : '#1e40af', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Circles</button>
              </div>

              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Markov Timesteps: {timesteps}</h3>
              <input type="range" min="10" max="200" step="10" value={timesteps} onChange={(e) => setTimesteps(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}>
                  <span>10 (Fast, rough generation)</span><span>200 (Slow, perfect generation)</span>
              </div>

              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Initial Noise Level: {noiseAmount.toFixed(1)}</h3>
              <input type="range" min="0.5" max="3.0" step="0.1" value={noiseAmount} onChange={(e) => setNoiseAmount(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>0.5 (Easy)</span><span>3.0 (Absolute Chaos)</span>
              </div>
            </div>

            <button onClick={trainModel} disabled={loading} style={{ width: '100%', backgroundColor: '#8b5cf6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}>
                {loading ? (
                    <>
                        <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle>
                            <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path>
                        </svg>
                        Denoising Data...
                    </>
                ) : 'Run Denoising Process'}
            </button>
          </div>
        </div>

        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Denoising Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #8b5cf6', borderRadius: '8px', backgroundColor: '#f9fafb', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', width: '100%' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Reverse Process</h4>
                            <span style={{ backgroundColor: '#f3e8ff', color: '#8b5cf6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>Timestep: {playback.frames[playback.currentIndex]?.step} / {results.timesteps}</span>
                        </div>
                        <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="Boundary" style={{ width: '100%', maxWidth: '400px', borderRadius: '8px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}/>
                        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', width: '100%' }}>
                            <button onClick={togglePlayback} style={{ backgroundColor: '#8b5cf6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer' }}>{playback.isPlaying ? '⏸' : '▶'}</button>
                            <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1 }} />
                        </div>
                    </div>

                    {/* (0,1) The Before/After Comparison */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>The Forward Process (T=0)</h4>
                            <img src={`data:image/png;base64,${results.pure_noise_img}`} alt="Noise" style={{ width: '100%', maxWidth: '250px', borderRadius: '4px' }}/>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6', marginTop: '1rem' }}>
                                Look at the image above. The AI starts with absolutely nothing but pure static noise (like a TV on an empty channel).
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Noise Schedule Graph */}
                    <div style={{ display: 'flex', flexDirection: 'column', gridColumn: '1 / 3' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Beta Noise Schedule (Signal vs Noise Ratio)</h4>
                        <div style={{ height: 300, width: '100%' }}>
                            <ResponsiveContainer>
                                <AreaChart data={results.noise_schedule}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="step" />
                                    <YAxis domain={[0, 100]} />
                                    <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
                                    <Legend />
                                    <Area type="monotone" dataKey="Noise" stroke="#ef4444" fill="#ef4444" stackId="1" fillOpacity={0.6} />
                                    <Area type="monotone" dataKey="Signal" stroke="#10b981" fill="#10b981" stackId="1" fillOpacity={0.6} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        <p style={{fontSize: '0.9rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                            This is the secret to DALL-E! The AI mathematically schedules the noise removal. In early timesteps, it removes huge chunks of noise (setting the global shape). In late timesteps, it removes tiny fractions of noise (fine-tuning the details).
                        </p>
                    </div>

                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default Diffusion;
