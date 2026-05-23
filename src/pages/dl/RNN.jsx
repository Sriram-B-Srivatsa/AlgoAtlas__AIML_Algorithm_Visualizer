/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function RNN() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");

  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);

  // Sample Data Modal
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleDataType, setSampleDataType] = useState('sine');
  const[sampleCount, setSampleCount] = useState(60);
  const [sampleNoise, setSampleNoise] = useState(0.1);

  // RNN Parameters
  const [epochs, setEpochs] = useState(100);
  const [lookback, setLookback] = useState(5);
  const[learningRate, setLearningRate] = useState(0.05);
  const [results, setResults] = useState(null);

  // 2x2 Playback State
  const [playback, setPlayback] = useState({
      active: false,
      frames:[],
      currentIndex: 0,
      isPlaying: false
  });

  const canvasDimensions = useMemo(() => ({ width: 600, height: 600 }),[]);
  const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -8, max: 8 } }),[]);
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    axios.get(`${apiUrl}/health`).then(() => setBackendStatus("connected")).catch(() => setBackendStatus("disconnected"));
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
        }, 400); // 0.4s per sliding window frame
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

    // Sort points automatically so it remains a strict Time-Series (left to right)
    const newPoints =[...points, { x: dataPoint.x, y: dataPoint.y }].sort((a, b) => a.x - b.x);
    setPoints(newPoints);
    setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const generateSampleDataWithOptions = () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);
    axios.post(`${apiUrl}/rnn/sample_data`, {
        dataset_type: sampleDataType, count: sampleCount, noise: sampleNoise
    })
    .then(response => {
        if(response.data.error) throw new Error(response.data.error);
        setPoints(response.data.points);
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    })
    .catch(err => setError("Failed to generate sequence."))
    .finally(() => setLoading(false));
  };

  const trainModel = () => {
    if (points.length <= lookback + 1) {
      setError(`Need at least ${lookback + 2} points for a lookback window of ${lookback}.`);
      return;
    }

    setLoading(true); setError(null);

    axios.post(`${apiUrl}/rnn/train`, {
      points: points,
      parameters: { epochs: epochs, lookback: lookback, learningRate: learningRate }
    })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) {
            setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
        }
    })
    .catch(err => setError("Failed to train RNN. Make sure backend is running."))
    .finally(() => setLoading(false));
  };

  // Draw Canvas (Connected Lines for Time Series)
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

    // Draw Sequence Line
    if (points.length > 1) {
        ctx.beginPath();
        points.forEach((point, idx) => {
            const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
            const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = '#9ca3af';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Draw Points
    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#3b82f6';
      ctx.fill(); ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.stroke();
    });
  }, [points]);

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', width: '90%', maxWidth: '550px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', margin: 0 }}>Generate Sequence Data</h2>
          <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Sequence Pattern</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            <button onClick={() => setSampleDataType('sine')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'sine' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'sine' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: 'bold' }}>Sine Wave</button>
            <button onClick={() => setSampleDataType('stock')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'stock' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'stock' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: 'bold' }}>Stock Trend</button>
            <button onClick={() => setSampleDataType('triangle')} style={{ padding: '0.5rem', backgroundColor: sampleDataType === 'triangle' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'triangle' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: 'bold' }}>Triangle Wave</button>
          </div>
        </div>

        {/* ADDED MIN/MAX LABELS TO SLIDERS */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="sample-count" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Time Steps (Points): {sampleCount}</label>
          <input id="sample-count" type="range" min="20" max="100" step="5" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.25rem' }}>
              <span>20 (Short)</span><span>100 (Long)</span>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="noise" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Noise / Volatility: {sampleNoise.toFixed(2)}</label>
          <input id="noise" type="range" min="0.0" max="0.5" step="0.05" value={sampleNoise} onChange={(e) => setSampleNoise(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.25rem' }}>
              <span>0.0 (Clean Pattern)</span><span>0.5 (Highly Volatile)</span>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleDataWithOptions} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Generate Sequence</button>
        </div>
      </div>
    </div>
  );

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {showSampleDataModal && <SampleDataModal />}

      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Recurrent Neural Networks (RNN) </h1>
      </div>

      <p className="model-description">
        RNNs process data sequentially. Unlike standard networks that look at everything at once, an RNN reads a "Time-Series" step-by-step, updating a hidden "Memory Cell" so it can predict the future based on the past!
        <InfoButton algoId="rnn" />
      </p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Time-Series Sandbox</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{backgroundColor:'#3b82f6',color:'white'}}>Load Timeline</button>
                <button className="sample-data-button" onClick={() => setPoints([])} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
              </div>
            </div>

            <p style={{ color: '#4b5563', marginBottom: '1rem', lineHeight: '1.5' }}>
              Click across the canvas from left to right to draw a stock trend or sound wave. The AI will learn the pattern!
            </p>

            <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', width: '100%', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={600} height={600} onClick={handleCanvasClick} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              </div>
            </div>

            {/* ADDED NEW EXPLANATION BOX */}
            <div style={{ width: '100%', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', fontSize: '0.85rem', marginTop: '1rem' }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>How Recurrent Networks Work</h3>
              <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                <p style={{ marginBottom: '0.5rem' }}>Instead of drawing a static boundary line, RNNs slide a "Reading Window" over time:</p>
                <ol style={{ paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                  <li>It looks at the past few days (Lookback Window).</li>
                  <li>It runs those values through its hidden Memory state.</li>
                  <li>It outputs a prediction for tomorrow (The Target).</li>
                  <li>It steps forward one day, updating its memory with the new reality, and predicts again!</li>
                </ol>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Network Architecture</h2>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>

              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Lookback Window: {lookback} steps</h3>
              <input type="range" min="2" max="15" step="1" value={lookback} onChange={(e) => setLookback(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>2 (Amnesia)</span><span>15 (Deep Memory)</span>
              </div>
              <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>
                  How many steps into the past the RNN is allowed to read before guessing the next point.
              </p>

              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Training Epochs: {epochs}</h3>
              <input type="range" min="10" max="300" step="10" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}>
                  <span>10 (Underfits)</span><span>300 (Learns pattern)</span>
              </div>

              {/* ADDED MIN/MAX TO LEARNING RATE */}
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Learning Rate: {learningRate.toFixed(2)}</h3>
              <input type="range" min="0.01" max="0.2" step="0.01" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}>
                  <span>0.01 (Slow)</span><span>0.20 (Aggressive)</span>
              </div>

              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Inside the RNN Cell</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ padding: '0.75rem', backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#1e40af' }}>Input Gate (Wx)</div>
                  <div style={{ fontSize: '0.8rem', color: '#3b82f6', marginTop: '4px' }}>Takes the current time step's value.</div>
                </div>
                <div style={{ padding: '0.75rem', backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#166534' }}>Hidden State / Memory (Wh)</div>
                  <div style={{ fontSize: '0.8rem', color: '#22c55e', marginTop: '4px' }}>Mixes the new Input with what it remembers from the previous steps using Tanh math!</div>
                </div>
                <div style={{ padding: '0.75rem', backgroundColor: '#fffbeb', border: '1px solid #fde68a', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#b45309' }}>Output Gate (Wy)</div>
                  <div style={{ fontSize: '0.8rem', color: '#f59e0b', marginTop: '4px' }}>Spits out the prediction for the future.</div>
                </div>
              </div>

            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <button onClick={trainModel} disabled={loading || points.length <= lookback + 1} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Unrolling Time...</>
                ) : 'Train Recurrent Network'}
              </button>
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Sliding Window Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Sequence Prediction</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>
                                Timestep: {playback.frames[playback.currentIndex]?.timestep}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                            <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="Sequence" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb' }}/>
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
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Final MSE: {results.final_loss.toFixed(4)}</p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>Epochs Trained: {results.epochs} | Lookback: {results.lookback}</p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                <strong>Watch the Animation:</strong> The thick blue line is the <strong>Lookback Window</strong>. The RNN reads those values, updates its internal memory state, and fires out the Green Dot as its prediction for the future! <br/><br/>If you set Lookback to 2, it acts like a goldfish. If you set it to 15, it acts like an elephant.
                            </p>
                        </div>
                    </div>

                    {/* (1,0) BPTT Loss Curve */}
                    {results.loss_history && results.loss_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Backpropagation Through Time (Loss)</h4>
                            <div style={{ height: 250, width: '100%' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.loss_history}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="epoch" />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line type="monotone" dataKey="loss" stroke="#ef4444" name="MSE Error" strokeWidth={2} dot={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                The RNN calculates its error at the end of the sequence, and rolls time backwards to update its weights!
                            </p>
                        </div>
                    )}

                    {/* (1,1) NEW MULTI-AXIS COMPOSED CHART */}
                    {results.hidden_state_activity && results.hidden_state_activity.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Memory Activation vs Prediction Error</h4>
                            <div style={{ height: 250, width: '100%' }}>
                                <ResponsiveContainer>
                                    <ComposedChart data={results.hidden_state_activity}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="timestep" />
                                        <YAxis yAxisId="left" orientation="left" stroke="#8b5cf6" />
                                        <YAxis yAxisId="right" orientation="right" stroke="#ef4444" />
                                        <Tooltip />
                                        <Legend />
                                        {/* Bar for Memory Activity */}
                                        <Bar yAxisId="left" dataKey="memory_activation" name="Memory Cell Activity" fill="#8b5cf6" opacity={0.6} />
                                        {/* Line for Error */}
                                        <Line yAxisId="right" type="step" dataKey="prediction_error" name="Prediction Error" stroke="#ef4444" strokeWidth={2} dot={false} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Notice how high Prediction Errors (Red Spikes) force the Memory Cell to activate intensely (Purple Bars) to memorize the mistake!
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

export default RNN;
