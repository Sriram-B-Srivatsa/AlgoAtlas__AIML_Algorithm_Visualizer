/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import { runPCA, getPCASampleData, checkHealth } from '../../api';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function PCA() {
  const navigate = useNavigate();
  const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
  const safeDataPairs = dataPairs || [];
  const [loading, setLoading] = useState(false);
  const[results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const[sampleLoading, setSampleLoading] = useState(false);

  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleCount, setSampleCount] = useState(30);
  const[sampleNoise, setSampleNoise] = useState(5.0);
  const [projectedPoints, setProjectedPoints] = useState([]);

  // --- STANDARD PLAYBACK STATE FOR 2x2 GRID ---
  const[playback, setPlayback] = useState({
      active: false,
      frames:[],
      currentIndex: 0,
      isPlaying: false
  });

  const canvasRef = useRef(null);
  const [canvasDimensions] = useState({ width: 600, height: 600 });
  const[isProjectionActive, setIsProjectionActive] = useState(false);

  const scale = {
    x: { min: -8, max: 8 },
    y: { min: -8, max: 8 }
  };

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${apiUrl}/health`);
        setBackendStatus(response.data.status === "healthy" ? "connected" : "disconnected");
      } catch (err) {
        setBackendStatus("disconnected");
        setError("Backend connection error: " + (err.message || "Unknown error"));
      }
    };
    checkBackendHealth();
  },[apiUrl]);

  // Playback Loop Effect
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
          }, 400); // 0.4s per projection frame
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
      .map(pair => ({
        x: parseFloat(pair.x),
        y: parseFloat(pair.y)
      }));
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f9f9f9";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    drawGrid(ctx, canvas);
    const points = getValidPoints();
    drawPoints(ctx, canvas, points);

    if (projectedPoints.length > 0) {
      drawPoints(ctx, canvas, projectedPoints, 'rgba(239, 51, 18, 0.7)');
    }

    if (results && results.components && results.original_mean && !playback.active) {
        drawPrincipalComponents(ctx, canvas, results.components, results.original_mean, points);
    }
  },[dataPairs, projectedPoints, results, isProjectionActive, playback.active]);

  const drawGrid = (ctx, canvas) => {
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;

    const stepX = canvas.width / 16;
    const stepY = canvas.height / 16;

    for (let i = 0; i <= 16; i++) {
        const y = i * stepY;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    for (let i = 0; i <= 16; i++) {
        const x = i * stepX;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }

    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    const yAxisPos = canvas.height / 2; ctx.beginPath(); ctx.moveTo(0, yAxisPos); ctx.lineTo(canvas.width, yAxisPos); ctx.stroke();
    const xAxisPos = canvas.width / 2; ctx.beginPath(); ctx.moveTo(xAxisPos, 0); ctx.lineTo(xAxisPos, canvas.height); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) {
        const x = i * stepX;
        const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min);
        ctx.fillText(value.toFixed(0), x - 8, canvas.height - 5);
    }
    for (let i = 0; i <= 16; i += 2) {
        const y = i * stepY;
        const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min);
        ctx.fillText(value.toFixed(0), 5, y + 4);
    }
  };

  const drawPoints = (ctx, canvas, points, color = 'rgba(59, 130, 246, 0.7)') => {
    if (!points || points.length === 0) return;
    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = canvas.height - ((point.y - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  };

  const drawPrincipalComponents = (ctx, canvas, components, originalMean, points) => {
    const meanX = ((originalMean[0] - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
    const meanY = canvas.height - ((originalMean[1] - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;

    const pc1 = components[0];
    const pc2 = components[1];

    const scaleFactor = Math.min(canvas.width, canvas.height) * 0.4;

    const pc1x = meanX + pc1[0] * scaleFactor;
    const pc1y = meanY - pc1[1] * scaleFactor;
    ctx.beginPath(); ctx.moveTo(meanX, meanY); ctx.lineTo(pc1x, pc1y); ctx.strokeStyle = 'black'; ctx.lineWidth = 3; ctx.stroke();
    drawArrow(ctx, pc1x, pc1y, pc1[0], pc1[1], 10);

    const pc2x = meanX + pc2[0] * scaleFactor;
    const pc2y = meanY - pc2[1] * scaleFactor;
    ctx.beginPath(); ctx.moveTo(meanX, meanY); ctx.lineTo(pc2x, pc2y); ctx.strokeStyle = 'indigo'; ctx.lineWidth = 3; ctx.stroke();
    drawArrow(ctx, pc2x, pc2y, pc2[0], pc2[1], 10);

    ctx.beginPath(); ctx.arc(meanX, meanY, 6, 0, Math.PI * 2); ctx.fillStyle = 'rgba(251, 191, 36, 0.7)'; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();

    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillStyle = 'black'; ctx.fillText('PC1', pc1x + 5, pc1y - 5);
    ctx.fillStyle = 'indigo'; ctx.fillText('PC2', pc2x + 5, pc2y - 5);
    ctx.fillStyle = '#b45309'; ctx.fillText('Mean', meanX + 10, meanY);

    if (isProjectionActive) {
      points.forEach(point => {
        const dx = point.x - originalMean[0];
        const dy = point.y - originalMean[1];
        const projectionLength = dx * pc1[0] + dy * pc1[1];
        const projectedX = originalMean[0] + projectionLength * pc1[0];
        const projectedY = originalMean[1] + projectionLength * pc1[1];

        const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = canvas.height - ((point.y - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
        const projX = ((projectedX - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const projY = canvas.height - ((projectedY - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;

        ctx.beginPath(); ctx.setLineDash([5, 5]); ctx.moveTo(x, y); ctx.lineTo(projX, projY); ctx.strokeStyle = 'rgba(107, 114, 128, 0.7)'; ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(projX, projY, 4, 0, Math.PI * 2); ctx.fillStyle = 'rgba(239, 51, 18, 0.7)'; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
      });
    }
  };

  const drawArrow = (ctx, x, y, dx, dy, size) => {
    const length = Math.sqrt(dx*dx + dy*dy);
    const dirX = dx / length; const dirY = dy / length;
    const perpX = -dirY; const perpY = dirX;

    ctx.beginPath(); ctx.moveTo(x, y);
    ctx.lineTo(x - size * dirX + size * 0.5 * perpX, y + size * dirY - size * 0.5 * perpY);
    ctx.lineTo(x - size * dirX - size * 0.5 * perpX, y + size * dirY + size * 0.5 * perpY);
    ctx.closePath(); ctx.fillStyle = "black"; ctx.fill();
  };

  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleFactorX = canvas.width / rect.width;
    const scaleFactorY = canvas.height / rect.height;
    const adjustedX = x * scaleFactorX;
    const adjustedY = y * scaleFactorY;
    const dataX = scale.x.min + (adjustedX / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - (adjustedY / canvas.height) * (scale.y.max - scale.y.min);
    return { x: dataX, y: dataY };
  };

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const dataPoint = screenToData(x, y);

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

  const handleAddPair = () => setDataPairs([...safeDataPairs, { x: '', y: '' }]);

  const handleRemovePair = (index) => {
    const newPairs = [...safeDataPairs];
    newPairs.splice(index, 1);
    if (newPairs.length === 0) newPairs.push({ x: '', y: '' });
    setDataPairs(newPairs);
  };

  const handleInputChange = (index, field, value) => {
    const newPairs = [...safeDataPairs];
    if (field === 'x' || field === 'y') {
      value = value.replace(/[^0-9.-]/g, '');
      const parts = value.split('.');
      if (parts.length > 2) value = parts[0] + '.' + parts.slice(1).join('');
      if (value.indexOf('-') > 0) {
        value = value.replace(/-/g, '');
        value = '-' + value;
      }
    }
    newPairs[index][field] = value;
    setDataPairs(newPairs);
  };

  // FIXED: Restored handleProjectPoints
  const handleProjectPoints = () => {
    if (!results || !results.components || !results.original_mean) {
      console.error("No results available for projecting points.");
      return;
    }

    const pc1 = results.components[0]; // First principal component
    const mean = results.original_mean; // Mean vector

    // Calculate projections for each point
    const projectedPairs = safeDataPairs
      .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(pair.x) && !isNaN(pair.y))
      .map(pair => {
        const x = parseFloat(pair.x);
        const y = parseFloat(pair.y);

        // Calculate projection onto PCA1
        const dx = x - mean[0];
        const dy = y - mean[1];
        const projectionLength = dx * pc1[0] + dy * pc1[1];
        const projectedX = mean[0] + projectionLength * pc1[0];
        const projectedY = mean[1] + projectionLength * pc1[1];

        return {
          x: projectedX.toFixed(2),
          y: projectedY.toFixed(2)
        };
      });

    setProjectedPoints(projectedPairs);
    setIsProjectionActive(true);
    setPlayback(p => ({ ...p, active: false })); // Hide video player if manual projection clicked
  };

  const generateSampleDataWithOptions = async () => {
    setSampleLoading(true); setShowSampleDataModal(false); setError(null);
    try {
      const response = await axios.get(`${apiUrl}/pca/sample-data?n_samples=${sampleCount}&noise=${sampleNoise / 10}`);
      const pairs = response.data.X.map((point) => ({
        x: point[0]?.toString() || '',
        y: point[1]?.toString() || ''
      }));
      setDataPairs(pairs);
      setResults(null);
      setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    } catch (err) {
      setError('Failed to load sample data.');
    } finally {
      setSampleLoading(false);
    }
  };

  const resetData = () => {
    setDataPairs([{ x: '', y: '' }]);
    setResults(null); setError(null);
    setProjectedPoints([]); setIsProjectionActive(false);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const handleRunModel = async () => {
    setIsProjectionActive(false);
    setDataPairs((prevPairs) => prevPairs.filter(pair => pair.x !== '' && pair.y !== ''));
    setProjectedPoints([]);

    const validPoints = getValidPoints();
    if (validPoints.length < 2) { setError('Please add at least 2 data points for meaningful PCA'); return; }

    setError(null); setLoading(true); setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });

    try {
      const apiData = { X: validPoints.map(point => [point.x, point.y]) };
      const response = await axios.post(`${apiUrl}/pca`, apiData);

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
      setError(`Error: ${err.message || 'An unknown error occurred'}. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  // Pre-calculate scree plot data for Recharts if results exist
  const screeData = results ?[
      { component: 'PC1', variance: results.explained_variance_ratio[0] * 100 },
      { component: 'PC2', variance: results.explained_variance_ratio[1] * 100 }
  ] :[];

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '8px', width: '90%', maxWidth: '500px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '600' }}>Generate Sample Data</h2>
          <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
        </div>
        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="sample-count" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
          <input id="sample-count" type="range" min="10" max="100" step="5" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
            <span>10 (Fewer Points)</span><span>100 (More points)</span>
          </div>
        </div>
        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="noise" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Noise Dispersion: {sampleNoise.toFixed(1)}</label>
          <input id="noise" type="range" min="1.0" max="10.0" step="0.5" value={sampleNoise} onChange={(e) => setSampleNoise(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
            <span>1 (Fewer Spread Samples)</span><span>10 (More Spread Samples)</span>
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleDataWithOptions} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Generate Data</button>
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
          <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
        </button>
        <h1 className="model-title">Principal Component Analysis (PCA) </h1>
      </div>

      <p className="model-description">
        Principal Component Analysis (PCA) identifies the mathematical directions (principal components) where your data varies the most. It is the core algorithm for Dimensionality Reduction.
        <InfoButton algoId="pca" />
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
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={sampleLoading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Load Sample Data</button>
                <button className="reset-button" onClick={resetData} disabled={loading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Reset Data</button>
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <p style={{ color: '#4b5563', marginBottom: '1rem', lineHeight: '1.5' }}>
              Click on the graph below to add data points, or manually enter values in the table.
            </p>

            <div style={{ marginBottom: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', height: '0', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={canvasDimensions.width} height={canvasDimensions.height} onClick={handleCanvasClick} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              </div>
              <div style={{ position: 'absolute', bottom: '10px', right: '10px', padding: '4px 8px', backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '4px', fontSize: '0.8rem', color: '#4b5563', pointerEvents: 'none' }}>
                Click to add point
              </div>
            </div>
          </div>

          {/* Right column: Data Table and Controls */}
          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Data Points Table</h2>

            <div style={{ marginBottom: '1.5rem', maxHeight: '400px', overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: '0.5rem', padding: '1rem', backgroundColor: 'white' }}>
              <div style={{ marginBottom: '0.5rem', display: 'flex', fontWeight: 'bold', color: '#4b5563' }}>
                <div style={{ flex: 1 }}>Feature X1</div>
                <div style={{ flex: 1 }}>Feature X2</div>
                <div style={{ width: '30px' }}></div>
              </div>

              {safeDataPairs.map((pair, index) => (
                <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                  <div style={{ display: 'flex', flex: 1, gap: '10px' }}>
                    <input type="text" style={{ flex: 1, padding: '8px', border: '1px solid #d1d5db', borderRadius: '4px' }} placeholder="X1 value" value={pair.x} onChange={(e) => handleInputChange(index, 'x', e.target.value)} />
                    <input type="text" style={{ flex: 1, padding: '8px', border: '1px solid #d1d5db', borderRadius: '4px' }} placeholder="X2 value" value={pair.y} onChange={(e) => handleInputChange(index, 'y', e.target.value)} />
                  </div>
                  <button onClick={() => handleRemovePair(index)} style={{ width: '32px', height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#fee2e2', border: 'none', borderRadius: '4px', marginLeft: '10px', cursor: 'pointer', color: '#b91c1c', fontWeight: 'bold' }}>×</button>
                </div>
              ))}

              <button onClick={handleAddPair} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', padding: '10px', backgroundColor: '#f3f4f6', border: 'none', borderRadius: '4px', cursor: 'pointer', marginTop: '10px', color: '#4b5563', fontWeight: '500' }}>
                + Add Data Row
              </button>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Algorithm Actions</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <button onClick={handleRunModel} disabled={loading || backendStatus === "disconnected"} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '12px', fontSize: '1.1rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1 }}>
                    {loading ? (
                      <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Running Math...</>
                    ) : 'Compute PCA'}
                  </button>
                  {/* Keep the manual projection feature as a backup interaction tool */}
                  <button onClick={handleProjectPoints} disabled={!results || loading || backendStatus === "disconnected"} style={{ width: '100%', backgroundColor: '#8b5cf6', color: 'white', padding: '12px', fontSize: '1.1rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: !results ? 'not-allowed' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: !results ? 0.7 : 1 }}>
                    Manually Show Projections
                  </button>
              </div>
            </div>

            {/* PCA Legend */}
            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', margin: '10px' }}>
                <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>PCA Visualization Guide:</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><span style={{ width: '12px', height: '12px', backgroundColor: 'rgba(59, 130, 246, 0.7)', borderRadius: '50%', display: 'inline-block' }}></span><span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Original 2D Data Points</span></div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><span style={{ width: '12px', height: '12px', backgroundColor: 'rgba(251, 191, 36, 0.7)', borderRadius: '50%', display: 'inline-block' }}></span><span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Dataset Center (Mean)</span></div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><span style={{ width: '12px', height: '3px', backgroundColor: 'rgb(1, 1, 1)', display: 'inline-block' }}></span><span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Principal Component 1 (Maximum Variance)</span></div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><span style={{ width: '12px', height: '3px', backgroundColor: 'rgba(139, 92, 246, 1)', display: 'inline-block' }}></span><span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Principal Component 2 (Orthogonal)</span></div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><span style={{ width: '12px', height: '12px', backgroundColor: 'rgba(239, 51, 18, 0.7)', borderRadius: '50%', display: 'inline-block' }}></span><span style={{ color: '#4b5563', fontSize: '0.9rem' }}>1D Projected Data Points</span></div>
                </div>
            </div>
          </div>
        </div>

        {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>PCA Analytical Dashboard</h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Dimensionality Reduction Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>1D Projection Animation</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Progress: {playback.frames.length > 0 ? playback.frames[playback.currentIndex].progress : 100}%
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1.5rem', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                                src={`data:image/png;base64,${playback.frames.length > 0 ? playback.frames[playback.currentIndex].image : results.history[results.history.length-1].image}`}
                                alt="Projection Evolution"
                                style={{ maxHeight: '100%', maxWidth: '100%', borderRadius: '8px', border: '1px solid #e5e7eb' }}
                            />
                        </div>
                        {playback.frames.length > 0 && (
                            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', justifyContent: 'center', maxWidth: '400px', margin: '0 auto' }}>
                                <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '1.2rem' }}>
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
                                Top Component Variance: {(results.explained_variance_ratio[0] * 100).toFixed(2)}%
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0', fontSize: '0.95rem' }}>
                                If you crush this 2D data down to a 1D line, you retain {(results.explained_variance_ratio[0] * 100).toFixed(2)}% of the original information.
                            </p>
                        </div>

                        <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', flex: 1 }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Eigen-Decomposition</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                <div>
                                    <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '0 0 4px 0' }}>Eigenvalue 1 (Magnitude)</p>
                                    <p style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', margin: 0 }}>{results.explained_variance[0].toFixed(4)}</p>
                                </div>
                                <div>
                                    <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '0 0 4px 0' }}>Eigenvalue 2</p>
                                    <p style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', margin: 0 }}>{results.explained_variance[1].toFixed(4)}</p>
                                </div>
                                <div style={{ gridColumn: '1 / 3', padding: '10px', backgroundColor: '#f8fafc', borderRadius: '4px', border: '1px dashed #cbd5e1' }}>
                                    <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '0 0 4px 0' }}>Eigenvector 1 (PC1 Direction)</p>
                                    <p style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', margin: 0, fontFamily: 'monospace' }}>[{results.components[0][0].toFixed(3)}, {results.components[0][1].toFixed(3)}]</p>
                                </div>
                                <div style={{ gridColumn: '1 / 3', padding: '10px', backgroundColor: '#f8fafc', borderRadius: '4px', border: '1px dashed #cbd5e1' }}>
                                    <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '0 0 4px 0' }}>Mean Center (Origin)</p>
                                    <p style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', margin: 0, fontFamily: 'monospace' }}>[{results.original_mean[0].toFixed(3)}, {results.original_mean[1].toFixed(3)}]</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* (1,0) The Scree Plot (BarChart) */}
                    {results.explained_variance_ratio && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Variance Scree Plot</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <BarChart data={screeData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="component" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} domain={[0, 100]} label={{ value: 'Variance %', angle: -90, position: 'insideLeft', style: {textAnchor: 'middle'} }} />
                                        <Tooltip cursor={{fill: 'transparent'}} formatter={(value) => `${value.toFixed(2)}%`} />
                                        <Bar dataKey="variance" name="Information Captured" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                The "Scree Plot" helps you visually decide how many dimensions to keep. Keep the tall bars, throw away the short ones.
                            </p>
                        </div>
                    )}

                    {/* (1,1) Correlation Matrix Heatmap */}
                    {results.corr_heatmap && (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem', width: '100%' }}>Feature Correlation Matrix</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
                                <img
                                    src={`data:image/png;base64,${results.corr_heatmap}`}
                                    alt="Correlation Heatmap"
                                    style={{ maxHeight: '100%', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}
                                />
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Red means X1 and X2 rise together. Blue means they move in opposite directions. High correlation = less dimensions needed!
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

export default PCA;
