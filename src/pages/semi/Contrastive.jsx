/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function Contrastive() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const[error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");

  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [brushMode, setBrushMode] = useState(0);

  // Sample Data Modal
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleDataType, setSampleDataType] = useState('blobs');
  const[sampleCount, setSampleCount] = useState(100);
  const [sampleVariance, setSampleVariance] = useState(0.5);
  const [sampleClusters, setSampleClusters] = useState(3);

  // Parameters
  const[epochs, setEpochs] = useState(100);
  const [margin, setMargin] = useState(2.0);
  const [learningRate, setLearningRate] = useState(0.05);
  const [results, setResults] = useState(null);

  // Playback
  const[playback, setPlayback] = useState({
      active: false,
      frames:[],
      currentIndex: 0,
      isPlaying: false
  });

  const canvasDimensions = useMemo(() => ({ width: 600, height: 600 }),[]);
  const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -8, max: 8 } }),[]);
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    axios.get(`${apiUrl}/health`).then(res => setBackendStatus("connected")).catch(() => setBackendStatus("disconnected"));
  }, [apiUrl]);

  useEffect(() => {
    let interval;
    if (playback.active && playback.isPlaying) {
        interval = setInterval(() => {
            setPlayback(prev => {
                if (prev.currentIndex >= prev.frames.length - 1) {
                    clearInterval(interval);
                    return { ...prev, isPlaying: false };
                }
                return { ...prev, currentIndex: prev.currentIndex + 1 };
            });
        }, 300); // 0.3s per frame
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
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);

    setPoints([...points, { x: dataPoint.x, y: dataPoint.y, class: brushMode }]);
    setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const generateSampleData = () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);
    axios.post(`${apiUrl}/cl/sample_data`, {
        dataset_type: sampleDataType, count: sampleCount, variance: sampleVariance, n_clusters: sampleClusters
    })
    .then(response => {
        if(response.data.error) throw new Error(response.data.error);
        setPoints(response.data.points);
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    })
    .catch(err => setError("Failed to generate sample data."))
    .finally(() => setLoading(false));
  };

  const trainModel = () => {
    if (points.length < 5) {
      setError("Place at least 5 points to perform Contrastive Learning.");
      return;
    }

    setLoading(true); setError(null);

    axios.post(`${apiUrl}/cl/train`, {
      points: points,
      parameters: { epochs: epochs, margin: margin, learningRate: learningRate }
    })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) {
            setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
        }
    })
    .catch(err => setError("Failed to run Contrastive Learning."))
    .finally(() => setLoading(false));
  };

  // Draw Canvas
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid & Axes
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
    const stepX = canvas.width / 16; const stepY = canvas.height / 16;
    for (let i = 0; i <= 16; i++) {
        ctx.beginPath(); ctx.moveTo(0, i * stepY); ctx.lineTo(canvas.width, i * stepY); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(i * stepX, 0); ctx.lineTo(i * stepX, canvas.height); ctx.stroke();
    }
    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    const yAxisPos = canvas.height / 2; ctx.beginPath(); ctx.moveTo(0, yAxisPos); ctx.lineTo(canvas.width, yAxisPos); ctx.stroke();
    const xAxisPos = canvas.width / 2; ctx.beginPath(); ctx.moveTo(xAxisPos, 0); ctx.lineTo(xAxisPos, canvas.height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) { const x = i * stepX; const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min); ctx.fillText(value.toFixed(0), x - 8, canvas.height - 5); }
    for (let i = 0; i <= 16; i += 2) { const y = i * stepY; const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min); ctx.fillText(value.toFixed(0), 5, y + 4); }

    const colors =['#3b82f6', '#ef4444', '#22c55e', '#f59e0b'];

    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;

      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = colors[point.class % colors.length];
      ctx.fill(); ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.stroke();
    });
  }, [points]);

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', width: '90%', maxWidth: '550px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', margin: 0 }}>Generate Sample Data</h2>
          <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Dataset Type</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            <button onClick={() => setSampleDataType('blobs')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'blobs' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'blobs' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Blobs</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Distinct clusters</span>
            </button>
            <button onClick={() => setSampleDataType('moons')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'moons' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'moons' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Moons</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Curved boundaries</span>
            </button>
            <button onClick={() => setSampleDataType('circles')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'circles' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'circles' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Circles</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Concentric circles</span>
            </button>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
          <input type="range" min="30" max="200" step="10" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>30 (Fewer Points)</span><span>99 (More points)</span>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Variance: {sampleVariance.toFixed(1)}</label>
          <input type="range" min="0.1" max="1.5" step="0.1" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>0.1 (Less Variance)</span><span>1.5 (More Variance)</span>
          </div>
        </div>

        {sampleDataType === 'blobs' && (
          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Clusters: {sampleClusters}</label>
            <input type="range" min="2" max="4" step="1" value={sampleClusters} onChange={(e) => setSampleClusters(Number(e.target.value))} style={{ width: '100%' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>2 (Fewer Clusters)</span><span>4 (More Clusters)</span>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleData} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Generate Data</button>
        </div>
      </div>
    </div>
  );

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {showSampleDataModal && <SampleDataModal />}

      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Contrastive Learning (Self-Supervised) </h1>
      </div>

      <p className="model-description">
        Contrastive Learning trains a model without human labels. It artificially creates "Positives" (by jittering a point) and forces them together, while pushing "Negatives" (random other points) far away.
        <InfoButton algoId="cl" />
      </p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">The Data Sandbox</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{backgroundColor: '#3b82f6', color:'white'}}>Load Sample Data</button>
                <button className="sample-data-button" onClick={() => setPoints([])} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <button onClick={() => setBrushMode(0)} style={{ padding: '0.5rem', backgroundColor: brushMode === 0 ? '#3b82f6' : '#eff6ff', color: brushMode === 0 ? 'white' : '#1e40af', border: '1px solid #bfdbfe', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Class 0</button>
                <button onClick={() => setBrushMode(1)} style={{ padding: '0.5rem', backgroundColor: brushMode === 1 ? '#ef4444' : '#fef2f2', color: brushMode === 1 ? 'white' : '#b91c1c', border: '1px solid #fecaca', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Class 1</button>
                <button onClick={() => setBrushMode(2)} style={{ padding: '0.5rem', backgroundColor: brushMode === 2 ? '#22c55e' : '#f0fdf4', color: brushMode === 2 ? 'white' : '#15803d', border: '1px solid #bbf7d0', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Class 2</button>
                <button onClick={() => setBrushMode(3)} style={{ padding: '0.5rem', backgroundColor: brushMode === 3 ? '#f59e0b' : '#fffbeb', color: brushMode === 3 ? 'white' : '#b45309', border: '1px solid #fde68a', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Class 3</button>
            </div>

            <div style={{ backgroundColor: '#fef2f2', border: '1px solid #fca5a5', padding: '1rem', borderRadius: '0.5rem', marginBottom: '1rem' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#b91c1c', fontSize: '0.95rem' }}>Important Note on Colors</h4>
                <p style={{ margin: 0, fontSize: '0.85rem', color: '#991b1b', lineHeight: '1.4' }}>
                    You are placing colored labels. <strong>However, the AI is completely blind to these colors!</strong> It only sees gray dots. The colors are strictly for YOU to see if the AI successfully grouped them together in the Latent Space below.
                </p>
            </div>

            <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>Click on the canvas below to add data points.</p>

            <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', width: '100%', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={600} height={600} onClick={handleCanvasClick} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Algorithm Controls</h2>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Training Epochs: {epochs}</h3>
              <input type="range" min="10" max="300" step="10" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem', marginTop: '0.5rem' }}>
                  <span>10 (Quick preview)</span><span>300 (Deep convergence)</span>
              </div>

              <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Repulsion Margin: {margin.toFixed(1)}</h3>
              <input type="range" min="0.5" max="5.0" step="0.5" value={margin} onChange={(e) => setMargin(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>0.5 (Weak separation)</span><span>5.0 (Strong separation)</span>
              </div>
              <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>How forcefully the AI shoves unrelated dots away from each other in the Latent Space.</p>

              <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Learning Rate: {learningRate.toFixed(2)}</h3>
              <input type="range" min="0.01" max="0.5" step="0.01" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  <span>0.01 (Slow & Stable)</span><span>0.50 (Fast & Erratic)</span>
              </div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <button onClick={trainModel} disabled={loading || points.length < 5} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Learning Representations...</>
                ) : 'Run Contrastive Learning'}
              </button>
            </div>

            <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#166534' }}>Note: </h4>
                <p style={{ fontSize: '0.9rem', color: '#15803d', margin: 0 }}>
                    Self-Supervised Learning doesn't predict "Dog" or "Cat". It builds a <strong>Latent Dictionary</strong>. Once this dictionary is built, companies freeze it, attach a simple classifier on top, and use it for FaceID or Image Search!
                </p>
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Latent Space Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Latent Space Evolution</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>
                                Epoch: {playback.frames[playback.currentIndex]?.epoch}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                            <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="Latent Space" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb' }}/>
                        </div>
                        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                            <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer', fontSize: '1.2rem' }}>
                                {playback.isPlaying ? '⏸' : '▶'}
                            </button>
                            <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1 }} />
                        </div>
                    </div>

                    {/* (0,1) Final Stats */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Final Loss: {results.final_loss.toFixed(4)}</p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>Epochs: {results.epochs} | Margin Penalty: {results.margin}</p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                <strong>Watch the Animation:</strong> All points start randomly squished together at the origin (0,0). <br/><br/>
                                Because we mathematically penalized the AI when random points were closer than the <strong>Margin of {results.margin}</strong>, the dots literally "explode" outwards, ripping themselves into perfect clusters!
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Contrastive Loss Curve */}
                    {results.loss_history && results.loss_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Contrastive Margin Loss</h4>
                            <div style={{ height: 250, width: '100%' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.loss_history}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="epoch" />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line type="monotone" dataKey="loss" stroke="#ef4444" name="InfoNCE / Margin Loss" strokeWidth={2} dot={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                The loss drops as the clusters physically push each other away in the latent space.
                            </p>
                        </div>
                    )}

                    {/* (1,1) Original Space Image */}
                    {results.original_plot && (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', width: '100%' }}>Original Input Space</h4>
                            <div style={{ height: 300, width: '100%', display: 'flex', justifyContent: 'center' }}>
                                <img src={`data:image/png;base64,${results.original_plot}`} alt="Original Space" style={{ maxHeight: '100%', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }} />
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Compare this to the final Latent Space frame above. The AI pulled the scattered groups into tight balls!
                            </p>
                        </div>
                    )}
                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default Contrastive;
