/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function Autoencoder() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);

  const[showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleDataType, setSampleDataType] = useState('moons');
  const [sampleCount, setSampleCount] = useState(60);
  const[sampleVariance, setSampleVariance] = useState(0.1);

  const [epochs, setEpochs] = useState(150);
  const [learningRate, setLearningRate] = useState(0.05);
  const[results, setResults] = useState(null);

  const [playback, setPlayback] = useState({ active: false, frames:[], currentIndex: 0, isPlaying: false });

  const canvasDimensions = useMemo(() => ({ width: 600, height: 600 }),[]);
  const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -8, max: 8 } }),[]);
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
        }, 400);
    }
    return () => clearInterval(interval);
  },[playback.active, playback.isPlaying]);

  const togglePlayback = () => {
      setPlayback(p => {
          if (!p.isPlaying && p.currentIndex >= p.frames.length - 1) return { ...p, currentIndex: 0, isPlaying: true };
          return { ...p, isPlaying: !p.isPlaying };
      });
  };

  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const dataX = scale.x.min + ((x * (canvas.width / rect.width)) / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - ((y * (canvas.height / rect.height)) / canvas.height) * (scale.y.max - scale.y.min);
    return { x: dataX, y: dataY };
  };

  const handleCanvasClick = (e) => {
    if (loading) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);
    setPoints([...points, { x: dataPoint.x, y: dataPoint.y }]);
    setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const generateSampleData = () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);
    axios.post(`${apiUrl}/ae/sample_data`, {
      dataset_type: sampleDataType,
      count: sampleCount,
      variance: sampleVariance,
      n_clusters: 3 // Add this!
    })
    .then(response => {
        if(response.data.error) throw new Error(response.data.error);
        setPoints(response.data.points);
        setResults(null); setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    })
    .catch(err => setError("Failed to load sample data."))
    .finally(() => setLoading(false));
  };

  const trainModel = () => {
    if (points.length < 3) { setError("Place at least 3 points."); return; }
    setLoading(true); setError(null);
    axios.post(`${apiUrl}/ae/train`, { points: points, parameters: { epochs, learningRate } })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
    })
    .catch(err => setError("Failed to run Autoencoder."))
    .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
    const stepX = canvas.width / 16; const stepY = canvas.height / 16;
    for (let i = 0; i <= 16; i++) {
        ctx.beginPath(); ctx.moveTo(0, i * stepY); ctx.lineTo(canvas.width, i * stepY); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(i * stepX, 0); ctx.lineTo(i * stepX, canvas.height); ctx.stroke();
    }
    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, canvas.height/2); ctx.lineTo(canvas.width, canvas.height/2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) {
        ctx.fillText((scale.x.min + (i/16)*(scale.x.max - scale.x.min)).toFixed(0), i*stepX - 8, canvas.height - 5);
        ctx.fillText((scale.y.max - (i/16)*(scale.y.max - scale.y.min)).toFixed(0), 5, i*stepY + 4);
    }

    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#3b82f6'; ctx.fill(); ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.stroke();
    });
  }, [points]);

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', width: '90%', maxWidth: '500px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '600' }}>Generate Data</h2>
          <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Dataset Type</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            <button onClick={() => setSampleDataType('blobs')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'blobs' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'blobs' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Blobs<span style={{ fontSize: '0.75rem', marginTop: '0.25rem', display:'block' }}>Distinct Clusters</span></button>
            <button onClick={() => setSampleDataType('moons')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'moons' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'moons' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Moons<span style={{ fontSize: '0.75rem', marginTop: '0.25rem', display:'block' }}>Curved Boundaries</span></button>
            <button onClick={() => setSampleDataType('circles')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'circles' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'circles' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Circles<span style={{ fontSize: '0.75rem', marginTop: '0.25rem', display:'block' }}>Concentric Circles</span></button>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Samples: {sampleCount}</label>
          <input type="range" min="20" max="200" step="10" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}>
            <span>20 (Sparse data)</span><span>200 (Dense data)</span>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Noise Dispersion: {sampleVariance.toFixed(2)}</label>
          <input type="range" min="0.0" max="0.5" step="0.05" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}>
            <span>0.0 (Clean shape)</span><span>0.5 (Messy/Scattered)</span>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleData} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', cursor: 'pointer', borderRadius: '0.5rem', border: 'none', fontWeight: '500' }}>Generate Data</button>
        </div>
      </div>
    </div>
  );

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {showSampleDataModal && <SampleDataModal />}

      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Autoencoders </h1>
      </div>

      <p className="model-description">Autoencoders are unsupervised Neural Networks shaped like an hourglass. They are forced to compress data into a tiny bottleneck, and then mathematically rebuild the original data from scratch!<InfoButton algoId="ae" /></p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Data Sandbox</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} style={{backgroundColor:'#3b82f6', color:'white'}}>Load Samples</button>
                <button className="sample-data-button" onClick={() => {setPoints([]); setResults(null);}} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
              </div>
            </div>

            <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>Click on the canvas below to add data points.</p>

            <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', width: '100%', paddingBottom: '100%' }}>
              <canvas ref={canvasRef} width={600} height={600} onClick={handleCanvasClick} style={{ position: 'absolute', top: 0, left: 0, display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Algorithm Controls</h2>
            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Training Epochs: {epochs}</h3>
              <input type="range" min="50" max="500" step="50" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}><span>50 (Underfits)</span><span>500 (Learns pattern)</span></div>

              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Learning Rate: {learningRate}</h3>
              <input type="range" min="0.01" max="0.2" step="0.01" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}><span>0.01 (Slow)</span><span>0.2 (Erratic)</span></div>

              <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px dashed #cbd5e1' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#334155' }}>Network Architecture</h4>
                  <p style={{ margin: 0, fontSize: '0.85rem', color: '#64748b', lineHeight: '1.5' }}>
                      <strong>Encoder:</strong> 2D Input &rarr; 8 Nodes &rarr; <strong>1D Bottleneck</strong><br/>
                      <strong>Decoder:</strong> 1D Bottleneck &rarr; 8 Nodes &rarr; 2D Output
                  </p>
              </div>
            </div>

            <button onClick={trainModel} disabled={loading || points.length === 0} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', marginBottom: '1rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}>
                {loading ? (
                    <>
                        <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle>
                            <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path>
                        </svg>
                        Compressing Data...
                    </>
                ) : 'Run Autoencoder'}
            </button>
          </div>
        </div>

        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Reconstruction Animation</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>Epoch: {playback.frames[playback.currentIndex]?.epoch}</span>
                        </div>
                        <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="Boundary" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}/>
                        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                            <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer' }}>{playback.isPlaying ? '⏸' : '▶'}</button>
                            <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1 }} />
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Final Loss (MSE): {results.final_loss.toFixed(4)}</p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>Compression Ratio: 2:1</p>
                        </div>
                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                Watch the animation! The Red dots are the "Reconstructed" points. Because the Network was forced to crush 2D (X,Y) coordinates into a 1D bottleneck, it lost information.<br/><br/>
                                To minimize the error, the AI mathematically learned the principal structure of the shape (e.g., the curve of the moon) to guess where the original dots were!
                            </p>
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Reconstruction Loss</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <LineChart data={results.loss_history}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="epoch" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line type="monotone" dataKey="loss" stroke="#ef4444" name="Mean Squared Error" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center'}}>Loss drops as the Decoder learns how to unpack the 1D bottleneck back into 2D space.</p>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>The 1D Latent Space (Bottleneck)</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis type="number" dataKey="latent_val" name="Latent Value" domain={['auto', 'auto']} />
                                    <YAxis type="number" dataKey="zero" domain={[-1, 1]} hide />
                                    <Tooltip cursor={{strokeDasharray: '3 3'}} />
                                    <Scatter name="Compressed Points" data={results.latent_data.map(d => ({ ...d, zero: 0 }))} fill="#8b5cf6" />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                        <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center'}}>
                            This is the physical representation of the Neural Network's memory. All your 2D points were mapped onto this single 1D number line!
                        </p>
                    </div>
                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default Autoencoder;
