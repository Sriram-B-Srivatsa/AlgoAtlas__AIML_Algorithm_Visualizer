/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function Regularization() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const[error, setError] = useState(null);

  const canvasRef = useRef(null);
  const[points, setPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const[predictions, setPredictions] = useState([]);
  const [pointsMode, setPointsMode] = useState('train');

  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const[sampleDataType, setSampleDataType] = useState('sine');
  const [sampleCount, setSampleCount] = useState(20);
  const [sampleNoise, setSampleNoise] = useState(0.5);

  const[regType, setRegType] = useState('Ridge');
  const [maxAlpha, setMaxAlpha] = useState(10.0);
  const [degree, setDegree] = useState(8);
  const[results, setResults] = useState(null);

  const [playback, setPlayback] = useState({ active: false, frames:[], currentIndex: 0, isPlaying: false });

  const canvasDimensions = useMemo(() => ({ width: 600, height: 600 }),[]);
  const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -5, max: 5 } }),[]);
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

    if (pointsMode === 'train') {
        setPoints([...points, { x: dataPoint.x, y: dataPoint.y }]);
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    } else {
        setPredictPoints([...predictPoints, { x: dataPoint.x, y: 0 }]); // y is 0 for unpredicted regression points
    }
  };

  const generateSampleData = () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);
    axios.post(`${apiUrl}/regularization/sample_data`, { dataset_type: sampleDataType, count: sampleCount, noise: sampleNoise })
    .then(response => {
        if(response.data.error) throw new Error(response.data.error);
        setPoints(response.data.points);
        setPredictPoints([]); setPredictions([]); setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    })
    .catch(err => setError("Failed to generate sample data"))
    .finally(() => setLoading(false));
  };

  const trainModel = () => {
    if (points.length < 5) { setError("Please place at least 5 points."); return; }
    setLoading(true); setError(null);
    axios.post(`${apiUrl}/regularization/train`, { points: points, parameters: { type: regType, alpha: maxAlpha, degree: degree } })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
    })
    .catch(err => setError("Failed to run Regularization."))
    .finally(() => setLoading(false));
  };

  const handlePrediction = () => {
    if (predictPoints.length === 0) { setError("Add points to predict first!"); return; }
    if (!results) { setError("Train the model first!"); return; }
    setLoading(true); setError(null);
    axios.post(`${apiUrl}/regularization/predict`, { trained_points: points, predict_points: predictPoints, parameters: { type: regType, alpha: maxAlpha, degree } })
    .then(response => {
      if (response.data.error) throw new Error(response.data.error);
      const formattedResults = response.data.predictions.map((r, idx) => ({
        x: predictPoints[idx].x, predictedValue: r
      }));
      setPredictions(formattedResults);
    })
    .catch(err => setError("Prediction failed."))
    .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Detailed Numbered Grid
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
    const stepX = canvas.width / 16; const stepY = canvas.height / 10; // -5 to 5 is 10 units
    for (let i = 0; i <= 10; i++) {
        ctx.beginPath(); ctx.moveTo(0, i * stepY); ctx.lineTo(canvas.width, i * stepY); ctx.stroke();
    }
    for (let i = 0; i <= 16; i++) {
        ctx.beginPath(); ctx.moveTo(i * stepX, 0); ctx.lineTo(i * stepX, canvas.height); ctx.stroke();
    }

    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, canvas.height/2); ctx.lineTo(canvas.width, canvas.height/2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) {
        ctx.fillText((scale.x.min + (i/16)*(scale.x.max - scale.x.min)).toFixed(0), i*stepX - 8, canvas.height - 5);
    }
    for (let i = 0; i <= 10; i += 2) {
        ctx.fillText((scale.y.max - (i/10)*(scale.y.max - scale.y.min)).toFixed(0), 5, i*stepY + 4);
    }

    // Training Points
    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#9ca3af'; ctx.fill(); ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.stroke();
    });

    // Unpredicted Points
    if (predictions.length === 0) {
        predictPoints.forEach(point => {
          const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
          const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;
          ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
          ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
        });
    }

    // Predicted Points
    predictions.forEach(pred => {
        const x = ((pred.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = ((scale.y.max - pred.predictedValue) / (scale.y.max - scale.y.min)) * canvas.height;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = results?.reg_type === 'Ridge' ? '#3b82f6' : '#ef4444';
        ctx.fill();
        ctx.fillStyle = '#000000'; ctx.font = '10px Arial'; ctx.fillText(pred.predictedValue.toFixed(2), x + 10, y - 10);
        ctx.setLineDash([2, 2]); ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    });

  }, [points, predictPoints, predictions]);

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
        <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', width: '90%', maxWidth: '550px' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '1.5rem' }}>Generate Data</h2>

            <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Dataset Pattern</label>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
                    <button onClick={() => setSampleDataType('linear')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'linear' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'linear' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Linear</button>
                    <button onClick={() => setSampleDataType('nonlinear')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'nonlinear' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'nonlinear' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Polynomial</button>
                    <button onClick={() => setSampleDataType('sine')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'sine' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'sine' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Sine Wave</button>
                </div>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Points (Keep low to force overfitting): {sampleCount}</label>
                <input type="range" min="10" max="100" step="5" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>10 (Sparse)</span><span>100 (Dense)</span></div>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Noise Level: {sampleNoise.toFixed(1)}</label>
                <input type="range" min="0.1" max="2.0" step="0.1" value={sampleNoise} onChange={(e) => setSampleNoise(Number(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.1 (Clean)</span><span>2.0 (Messy)</span></div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
                <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', cursor: 'pointer', borderRadius: '0.5rem', border: 'none' }}>Cancel</button>
                <button onClick={generateSampleData} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', cursor: 'pointer', borderRadius: '0.5rem', border: 'none' }}>Generate</button>
            </div>
        </div>
    </div>
  );

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {showSampleDataModal && <SampleDataModal />}

      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Ridge & Lasso Regularization </h1>
      </div>

      <p className="model-description">Regularization (L1 Lasso and L2 Ridge) artificially penalizes complex math equations to prevent Overfitting. It literally forces an over-complicated, squiggly line to smooth out into a gentle curve!<InfoButton algoId="regularization" /></p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          {/* LEFT: CANVAS */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Data Sandbox</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} style={{backgroundColor:'#3b82f6', color:'white'}}>Load Samples</button>
                <button className="sample-data-button" onClick={() => {setPoints([]); setPredictPoints([]); setPredictions([]); setResults(null);}} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
              </div>
            </div>

            <p style={{ fontSize: '0.85rem', color: '#6b7280', fontWeight: '500', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              Click on the grid below to plot your data points.
            </p>

            <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', width: '100%', paddingBottom: '100%' }}>
              <canvas ref={canvasRef} width={600} height={600} onClick={handleCanvasClick} style={{ position: 'absolute', top: 0, left: 0, display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              <div style={{ position: 'absolute', bottom: '10px', right: '10px', padding: '4px 8px', backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '4px', fontSize: '0.8rem', color: '#4b5563', pointerEvents: 'none' }}>
                {pointsMode === 'train' ? 'Click to add training point' : 'Click to add prediction point'}
              </div>
            </div>
          </div>

          {/* RIGHT: CONTROLS */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Algorithm Controls</h2>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <button onClick={() => setRegType('Ridge')} style={{ padding: '0.75rem', backgroundColor: regType === 'Ridge' ? '#3b82f6' : '#eff6ff', color: regType === 'Ridge' ? 'white' : '#1e40af', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Ridge (L2 Penalty)</button>
                <button onClick={() => setRegType('Lasso')} style={{ padding: '0.75rem', backgroundColor: regType === 'Lasso' ? '#ef4444' : '#fef2f2', color: regType === 'Lasso' ? 'white' : '#b91c1c', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Lasso (L1 Penalty)</button>
              </div>

              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Max Alpha (Penalty Strength): {maxAlpha}</h3>
              <input type="range" min="0.1" max="50.0" step="0.1" value={maxAlpha} onChange={(e) => setMaxAlpha(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}><span>0.1 (No penalty)</span><span>50.0 (High penalty)</span></div>

              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Polynomial Degree: {degree}</h3>
              <input type="range" min="3" max="12" step="1" value={degree} onChange={(e) => setDegree(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}><span>3 (Simple)</span><span>12 (Overfitted)</span></div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Interaction Mode</h3>
                <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button onClick={() => setPointsMode('train')} style={{ padding: '0.75rem', backgroundColor: pointsMode === 'train' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'train' ? 'white' : '#4b5563', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Training Points</button>
                    <button onClick={() => setPointsMode('predict')} style={{ padding: '0.75rem', backgroundColor: pointsMode === 'predict' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'predict' ? 'white' : '#4b5563', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Prediction Points</button>
                </div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                <button onClick={trainModel} disabled={loading || points.length === 0} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', marginBottom: '1rem' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Applying Penalties...</>
                ) : 'Run Regularization'}
                </button>
                <button onClick={handlePrediction} disabled={loading || !results || predictPoints.length === 0} style={{ width: '100%', backgroundColor: '#8b5cf6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: (!results || predictPoints.length === 0) ? 'not-allowed' : 'pointer' }}>
                Predict Y-Values
                </button>
            </div>
          </div>
        </div>

        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    <div style={{ padding: '1.5rem', border: `2px solid ${results.reg_type === 'Ridge' ? '#3b82f6' : '#ef4444'}`, borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Penalty Animation</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>Alpha: {playback.frames[playback.currentIndex]?.alpha.toFixed(3)}</span>
                        </div>
                        <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="Boundary" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}/>
                        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                            <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer' }}>{playback.isPlaying ? '⏸' : '▶'}</button>
                            <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1 }} />
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Final R² Score: {(results.final_r2 * 100).toFixed(2)}%</p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>MSE: {results.final_mse.toFixed(4)}</p>
                        </div>
                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                The Red dashed line is an overfitted polynomial (Degree {results.degree}) with no penalty. It hits every dot but draws crazy zig-zags.<br/><br/>
                                By increasing Alpha, we mathematically punish the AI for drawing complex curves. The Blue line perfectly smoothens out into the true, generalized trend!
                            </p>
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Final Coefficient Magnitudes</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <BarChart data={results.final_weights}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="feature" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Bar dataKey="weight" name="Weight Value" fill={results.reg_type === 'Ridge' ? '#3b82f6' : '#ef4444'} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center'}}>
                            {results.reg_type === 'Lasso'
                                ? "Notice how Lasso (L1) aggressively forces useless weights to EXACTLY zero, performing Feature Selection!"
                                : "Notice how Ridge (L2) shrinks weights close to zero, but rarely exactly zero."}
                        </p>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Coefficient Shrinkage Path</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <LineChart data={results.coef_history}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="alpha" scale="log" domain={['auto', 'auto']} type="number" />
                                    <YAxis />
                                    <Tooltip />
                                    <Line type="monotone" dataKey="w1" stroke="#3b82f6" dot={false} />
                                    <Line type="monotone" dataKey="w2" stroke="#ef4444" dot={false} />
                                    <Line type="monotone" dataKey="w3" stroke="#10b981" dot={false} />
                                    <Line type="monotone" dataKey="w4" stroke="#f59e0b" dot={false} />
                                    <Line type="monotone" dataKey="w5" stroke="#8b5cf6" dot={false} />
                                </LineChart>
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

export default Regularization;
