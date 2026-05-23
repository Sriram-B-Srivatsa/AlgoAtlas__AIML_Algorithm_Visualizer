/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function GMM() {
  const navigate = useNavigate();
  const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
  const safeDataPairs = dataPairs || [];
  const[loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const[backendStatus, setBackendStatus] = useState("checking");

  const [components, setComponents] = useState(3);
  const [maxIterations, setMaxIterations] = useState(100);
  const [results, setResults] = useState(null);

  // Sample Data Modal
  const[showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleDataType, setSampleDataType] = useState('anisotropic');
  const [sampleCount, setSampleCount] = useState(150);
  const [sampleClusters, setSampleClusters] = useState(3);
  const [sampleVariance, setSampleVariance] = useState(0.5);

  // --- PLAYBACK STATE FOR 2x2 GRID ---
  const [playback, setPlayback] = useState({
      active: false,
      frames:[],
      currentIndex: 0,
      isPlaying: false
  });

  const canvasRef = useRef(null);
  const [canvasDimensions] = useState({ width: 600, height: 600 });
  const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -8, max: 8 } }),[]);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${apiUrl}/health`);
        setBackendStatus(response.data.status === "healthy" ? "connected" : "disconnected");
      } catch (err) {
        setBackendStatus("disconnected");
      }
    };
    checkBackendHealth();
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
        }, 500); // 0.5s per EM step
    }
    return () => clearInterval(interval);
  },[playback.active, playback.isPlaying]);

  const togglePlayback = () => {
      setPlayback(p => {
          if (!p.isPlaying && p.currentIndex >= p.frames.length - 1) {
              return { ...p, currentIndex: 0, isPlaying: true };
          }
          return { ...p, isPlaying: !p.isPlaying };
      });
  };

  const getValidPoints = () => {
    return safeDataPairs
      .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
      .map(pair => ({ x: parseFloat(pair.x), y: parseFloat(pair.y) }));
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f9f9f9";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Grid
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
    const stepX = canvas.width / 16; const stepY = canvas.height / 16;
    for (let i = 0; i <= 16; i++) { const y = i * stepY; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
    for (let i = 0; i <= 16; i++) { const x = i * stepX; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }

    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    const yAxisPos = canvas.height / 2; ctx.beginPath(); ctx.moveTo(0, yAxisPos); ctx.lineTo(canvas.width, yAxisPos); ctx.stroke();
    const xAxisPos = canvas.width / 2; ctx.beginPath(); ctx.moveTo(xAxisPos, 0); ctx.lineTo(xAxisPos, canvas.height); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) {
        const x = i * stepX; const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min);
        ctx.fillText(value.toFixed(0), x - 8, canvas.height - 5);
    }
    for (let i = 0; i <= 16; i += 2) {
        const y = i * stepY; const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min);
        ctx.fillText(value.toFixed(0), 5, y + 4);
    }

    // Draw Points
    getValidPoints().forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = canvas.height - ((point.y - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
      ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
    });
  }, [dataPairs]);

  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleFactorX = canvas.width / rect.width;
    const scaleFactorY = canvas.height / rect.height;
    const dataX = scale.x.min + ((x * scaleFactorX) / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - ((y * scaleFactorY) / canvas.height) * (scale.y.max - scale.y.min);
    return { x: dataX, y: dataY };
  };

  const handleCanvasClick = (e) => {
    if (loading) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);
    const emptyPairIndex = safeDataPairs.findIndex(pair => pair.x === '' && pair.y === '');
    const newPoint = { x: dataPoint.x.toFixed(2), y: dataPoint.y.toFixed(2) };

    if (emptyPairIndex >= 0) {
      const newPairs = [...safeDataPairs];
      newPairs[emptyPairIndex] = newPoint;
      setDataPairs(newPairs);
    } else {
      setDataPairs([...safeDataPairs, newPoint]);
    }
    setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const resetData = () => {
    setDataPairs([{ x: '', y: '' }]);
    setResults(null); setError(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const generateSampleData = async () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);
    try {
      const response = await axios.post(`${apiUrl}/gmm/sample_data`, {
        dataset_type: sampleDataType, count: sampleCount, n_clusters: sampleClusters, variance: sampleVariance
      });
      if (response.data.points) {
        const pairs = response.data.points.map(p => ({ x: p.x.toString(), y: p.y.toString() }));
        setDataPairs(pairs);
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
      } else {
        setError(`Backend Error: ${response.data.error}`);
      }
    } catch (err) {
      setError(`Network Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunModel = async () => {
    setDataPairs((prevPairs) => prevPairs.filter(pair => pair.x !== '' && pair.y !== ''));
    const validPoints = getValidPoints();
    if (validPoints.length < components) { setError(`Please add at least ${components} data points.`); return; }

    setError(null); setLoading(true); setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });

    try {
      const apiData = {
        X: validPoints.map(point => [point.x, point.y]),
        components: components,
        max_iterations: maxIterations
      };
      const response = await axios.post(`${apiUrl}/gmm/train`, apiData);

      if (response.data.error) throw new Error(response.data.error);

      setResults(response.data);
      if (response.data.history && response.data.history.length > 0) {
          setPlayback({
              active: true,
              frames: response.data.history,
              currentIndex: 0,
              isPlaying: true
          });
      }
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

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
            <button onClick={() => setSampleDataType('anisotropic')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'anisotropic' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'anisotropic' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Elongated</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Stretched blobs</span>
            </button>
            <button onClick={() => setSampleDataType('blobs')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'blobs' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'blobs' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Standard</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Circular blobs</span>
            </button>
            <button onClick={() => setSampleDataType('moons')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'moons' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'moons' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Moons</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Curved shapes</span>
            </button>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="sample-count" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
          <input id="sample-count" type="range" min="50" max="300" step="10" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>50 (Fewer points)</span><span>300 (More points)</span></div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="variance" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Variance (Noise): {sampleVariance.toFixed(1)}</label>
          <input id="variance" type="range" min="0.1" max="1.5" step="0.1" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.1 (Tight)</span><span>1.5 (Spread out)</span></div>
        </div>

        {(sampleDataType === 'anisotropic' || sampleDataType === 'blobs') && (
          <div style={{ marginBottom: '1.25rem' }}>
            <label htmlFor="clusters" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of True Clusters: {sampleClusters}</label>
            <input id="clusters" type="range" min="2" max="6" step="1" value={sampleClusters} onChange={(e) => setSampleClusters(Number(e.target.value))} style={{ width: '100%' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>2 clusters</span><span>6 clusters</span></div>
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
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      {showSampleDataModal && <SampleDataModal />}

      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Hub</span>
        </button>
        <h1 className="model-title">Gaussian Mixture Models (GMM) </h1>
      </div>

      <p className="model-description">
        GMM is a probabilistic clustering model. Unlike K-Means (which assumes circular clusters), GMM fits elliptical Gaussians that can stretch and rotate, making it powerful for complex data distributions.
        <InfoButton algoId="gmm" />
      </p>

      {backendStatus === "disconnected" && (
        <div className="backend-status error">
          <span>Backend service is not responding. Please make sure the Flask server is running on port 5000.</span>
        </div>
      )}

      <div className="content-container" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

        {/* TOP ROW: Input & Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

          {/* Left column: Input Plot */}
          <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
            <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 className="section-title">Interactive Data Input</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Load Sample Data</button>
                <button className="reset-button" onClick={resetData} disabled={loading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Reset Data</button>
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <p style={{ color: '#4b5563', marginBottom: '1rem', lineHeight: '1.5' }}>
              Click on the graph below to add data points.
            </p>

            <div style={{ marginBottom: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', height: '0', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={canvasDimensions.width} height={canvasDimensions.height} onClick={handleCanvasClick} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              </div>
            </div>
          </div>

          {/* Right column: Controls */}
          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title" style={{marginBottom: '1rem'}}>Algorithm Controls</h2>
            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%', marginBottom: '1.5rem' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Parameters</h3>

              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Components (Gaussians): {components}</label>
                <input type="range" min="1" max="8" step="1" value={components} onChange={(e) => setComponents(parseInt(e.target.value))} style={{ width: '100%' }} disabled={loading}/>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}>
                    <span>1 (Underfitting, High Bias)</span><span>8 (Overfitting, High Variance)</span>
                </div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Max Iterations (EM Steps): {maxIterations}</label>
                <input type="range" min="10" max="300" step="10" value={maxIterations} onChange={(e) => setMaxIterations(parseInt(e.target.value))} style={{ width: '100%' }} disabled={loading}/>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}>
                    <span>10 (Underfit)</span><span>300 (Slow Execution)</span>
                </div>
              </div>

              <button onClick={handleRunModel} disabled={loading || backendStatus === "disconnected" || getValidPoints().length < components} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#10b981', color: 'white', padding: '12px', fontSize: '1.1rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1 }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Running EM Algorithm...</>
                ) : 'Train Gaussian Mixture'}
              </button>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Why GMM over K-Means?</h3>
                <p style={{ color: '#4b5563', fontSize: '0.9rem', lineHeight: '1.5' }}>
                    K-Means is a "hard clustering" algorithm that draws strict circular boundaries. If your data is stretched into ovals (like the Elongated Blobs dataset), K-Means fails.<br/><br/>
                    GMM assigns a <strong>probability</strong> to each point, allowing its Gaussians (the yellow contours) to rotate, stretch, and perfectly match the real covariance of the data!
                </p>
            </div>
          </div>
        </div>

        {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>GMM Analytical Dashboard</h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) EM Algorithm Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #10b981', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>EM Convergence Animation</h4>
                            <span style={{ backgroundColor: '#d1fae5', color: '#059669', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Iteration: {playback.frames.length > 0 ? playback.frames[playback.currentIndex].iteration : results.iterations}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1.5rem', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                                src={`data:image/png;base64,${playback.frames.length > 0 ? playback.frames[playback.currentIndex].image : results.history[results.history.length-1].image}`}
                                alt="EM Evolution"
                                style={{ maxHeight: '100%', maxWidth: '100%', borderRadius: '8px', border: '1px solid #e5e7eb' }}
                            />
                        </div>
                        {playback.frames.length > 0 && (
                            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', justifyContent: 'center', maxWidth: '400px', margin: '0 auto' }}>
                                <button onClick={togglePlayback} style={{ backgroundColor: '#10b981', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '1.2rem' }}>
                                    {playback.isPlaying ? '⏸' : '▶'}
                                </button>
                                <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1, cursor: 'pointer' }} />
                            </div>
                        )}
                    </div>

                    {/* (0,1) Final Mathematical Stats */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                Final Log-Likelihood: {results.final_log_likelihood.toFixed(2)}
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0', fontSize: '0.95rem' }}>
                                Converged: {results.converged ? 'Yes' : 'No'} | Total Iterations: {results.iterations}
                            </p>
                        </div>

                        <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', flex: 1 }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Model Evaluation Scores</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '4px', border: '1px dashed #cbd5e1' }}>
                                    <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '0 0 4px 0' }}>AIC (Akaike Info Criterion)</p>
                                    <p style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', margin: 0 }}>{results.final_aic.toFixed(2)}</p>
                                </div>
                                <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '4px', border: '1px dashed #cbd5e1' }}>
                                    <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '0 0 4px 0' }}>BIC (Bayesian Info Criterion)</p>
                                    <p style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', margin: 0 }}>{results.final_bic.toFixed(2)}</p>
                                </div>
                                <div style={{ gridColumn: '1 / 3' }}>
                                    <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: 0, fontStyle: 'italic' }}>* Lower AIC and BIC scores mean a better model that penalizes unnecessary complexity.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* (1,0) Log-Likelihood Convergence Curve */}
                    {results.log_likelihood_history && results.log_likelihood_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Log-Likelihood Convergence</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.log_likelihood_history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="iteration" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line type="monotone" dataKey="log_likelihood" stroke="#10b981" name="Log-Likelihood" strokeWidth={2} dot={false} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                The EM algorithm seeks to maximize the Log-Likelihood. When the curve goes completely flat, the Gaussians have locked into their optimal positions.
                            </p>
                        </div>
                    )}

                    {/* (1,1) AIC/BIC Model Selection Curve */}
                    {results.aic_bic_history && results.aic_bic_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Optimal Components (AIC/BIC)</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.aic_bic_history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="k" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line type="monotone" dataKey="AIC" stroke="#f59e0b" name="AIC Score" strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                        <Line type="monotone" dataKey="BIC" stroke="#6366f1" name="BIC Score" strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Look for the "elbow" or the lowest point on the BIC curve. That mathematically indicates the true number of clusters in the data!
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

export default GMM;
