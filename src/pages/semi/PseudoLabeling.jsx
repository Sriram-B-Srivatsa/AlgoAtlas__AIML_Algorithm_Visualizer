/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function PseudoLabeling() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const[backendStatus, setBackendStatus] = useState("connected");

  const canvasRef = useRef(null);
  const[points, setPoints] = useState([]);
  const [brushMode, setBrushMode] = useState(-1); // 0, 1, or -1 (unlabeled)

  // Sample Data Modal
  const[showSampleDataModal, setShowSampleDataModal] = useState(false);
  const[sampleDataType, setSampleDataType] = useState('moons');
  const [sampleCount, setSampleCount] = useState(100);
  const [sampleVariance, setSampleVariance] = useState(0.15);
  const [sampleClusters, setSampleClusters] = useState(2);

  // Parameters
  const[confidenceThreshold, setConfidenceThreshold] = useState(0.80);
  const [maxIterations, setMaxIterations] = useState(15);
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

  const generateSampleDataWithOptions = () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);
    axios.post(`${apiUrl}/pl/sample_data`, {
        dataset_type: sampleDataType,
        count: sampleCount,
        variance: sampleVariance,
        n_clusters: sampleClusters
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
    const labeledClasses = new Set(points.filter(p => p.class !== -1).map(p => p.class));
    if (labeledClasses.size < 2) {
      setError("You must place at least one Blue dot (Class 0) and one Red dot (Class 1) to teach the model.");
      return;
    }

    setLoading(true); setError(null);

    axios.post(`${apiUrl}/pl/train`, {
      points: points,
      parameters: { confidence: confidenceThreshold, max_iter: maxIterations }
    })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) {
            setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
        }
    })
    .catch(err => setError("Failed to run Pseudo-Labeling."))
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

    // Points
    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;

      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      if (point.class === 0) ctx.fillStyle = '#3b82f6';
      else if (point.class === 1) ctx.fillStyle = '#ef4444';
      else ctx.fillStyle = '#9ca3af'; // Unlabeled

      ctx.fill(); ctx.strokeStyle = point.class === -1 ? 'white' : 'black'; ctx.lineWidth = 1; ctx.stroke();
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
          <label htmlFor="sample-count" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
          <input id="sample-count" type="range" min="20" max="200" step="10" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>20 (Fewer points)</span><span>200 (More points)</span></div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="variance" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Variance: {sampleVariance.toFixed(2)}</label>
          <input id="variance" type="range" min="0.05" max="0.5" step="0.05" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.05 (Compact)</span><span>0.5 (Spread out)</span></div>
        </div>

        {sampleDataType === 'blobs' && (
          <div style={{ marginBottom: '1.25rem' }}>
            <label htmlFor="clusters" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Clusters: {sampleClusters}</label>
            <input id="clusters" type="range" min="2" max="4" step="1" value={sampleClusters} onChange={(e) => setSampleClusters(Number(e.target.value))} style={{ width: '100%' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>2 clusters</span><span>4 clusters</span></div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleDataWithOptions} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Generate Data</button>
        </div>
      </div>
    </div>
  );

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {showSampleDataModal && <SampleDataModal />}

      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <span>&larr; Back to Hub</span>
        </button>
        <h1 className="model-title">Pseudo-Labeling (Proxy-Labeling)</h1>
      </div>

      <p className="model-description">
        In the real world, labeling data is expensive. Semi-Supervised Learning solves this by training a model on a tiny amount of Labeled data, and then having the model "Pseudo-Label" the massive pile of Unlabeled data!
        <InfoButton algoId="pl" />
      </p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">The Data Sandbox</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{ backgroundColor: '#3b82f6', color: 'white'}}>Load Sample Data</button>
                <button className="sample-data-button" onClick={() => setPoints([])} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
                <button onClick={() => setBrushMode(0)} style={{ padding: '0.5rem', backgroundColor: brushMode === 0 ? '#3b82f6' : '#eff6ff', color: brushMode === 0 ? 'white' : '#1e40af', border: '1px solid #bfdbfe', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Blue (Class 0)</button>
                <button onClick={() => setBrushMode(1)} style={{ padding: '0.5rem', backgroundColor: brushMode === 1 ? '#ef4444' : '#fef2f2', color: brushMode === 1 ? 'white' : '#b91c1c', border: '1px solid #fecaca', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Red (Class 1)</button>
                <button onClick={() => setBrushMode(-1)} style={{ padding: '0.5rem', backgroundColor: brushMode === -1 ? '#4b5563' : '#f3f4f6', color: brushMode === -1 ? 'white' : '#1f2937', border: '1px solid #d1d5db', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Gray (Unlabeled)</button>
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
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Confidence Threshold: {(confidenceThreshold*100).toFixed(0)}%</h3>
              <input type="range" min="0.51" max="0.99" step="0.01" value={confidenceThreshold} onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>51% (Risky, labels fast)</span><span>99% (Strict, labels slowly)</span>
              </div>
              <p style={{fontSize: '0.85rem', color: '#6b7280'}}>The model must be &gt;{(confidenceThreshold*100).toFixed(0)}% sure before it turns a gray dot into a colored dot.</p>

              <h3 style={{ margin: '1.5rem 0 1rem 0', fontSize: '1.1rem', fontWeight: '500' }}>Max Iterations: {maxIterations}</h3>
              <input type="range" min="1" max="30" step="1" value={maxIterations} onChange={(e) => setMaxIterations(parseInt(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  <span>1 (Stops early)</span><span>30 (Thorough propagation)</span>
              </div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <button onClick={trainModel} disabled={loading || points.length === 0} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Propagating Labels...</>
                ) : 'Run Pseudo-Labeling'}
              </button>
            </div>

            <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#166534' }}>How to play with this:</h4>
                <p style={{ fontSize: '0.9rem', color: '#15803d', margin: 0 }}>Load the different datasets. Notice there are 100 gray dots, but only 6 colored dots. Hit Train! Watch the AI use those 6 dots to figure out the colors of all 100 dots!</p>
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Label Propagation Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Label Propagation</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>
                                Iteration: {playback.frames[playback.currentIndex]?.iteration}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                            <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="Boundary" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb' }}/>
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
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Successfully Guessed: {results.final_pseudo_labeled} labels</p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>Remaining Unlabeled: {results.remaining_unlabeled} | Started with: {results.initial_labeled}</p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                The dashed lines in the animation represent the <strong>Confidence Margins</strong>. Any gray dot that falls outside the dashed lines is considered &gt;{(confidenceThreshold*100).toFixed(0)}% safe to label. <br/><br/>As gray dots get colored, they pull the boundary closer to the true shape of the data!
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Confidence Curve */}
                    {results.confidence_history && results.confidence_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Model Confidence Growth</h4>
                            <div style={{ height: 250, width: '100%' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.confidence_history}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="iteration" />
                                        <YAxis domain={[50, 100]} />
                                        <Tooltip />
                                        <Legend />
                                        <Line type="monotone" dataKey="avg_confidence" stroke="#10b981" name="Avg Confidence (%)" strokeWidth={2} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}

                    {/* (1,1) Data Utilization Bar Chart */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Data Utilization</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <BarChart data={[{name: 'Start', Labeled: results.initial_labeled, Unlabeled: results.total_points - results.initial_labeled}, {name: 'End', Labeled: results.initial_labeled + results.final_pseudo_labeled, Unlabeled: results.remaining_unlabeled}]}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="name" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Bar dataKey="Labeled" fill="#3b82f6" stackId="a" />
                                    <Bar dataKey="Unlabeled" fill="#9ca3af" stackId="a" />
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

export default PseudoLabeling;
