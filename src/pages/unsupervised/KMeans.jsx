/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function KMeans() {
  const navigate = useNavigate();
  const [data, setData] = useState({ X: [] });
  const[clusterCount, setClusterCount] = useState(3);
  const [maxIterations, setMaxIterations] = useState(100);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");

  // Sample Data Modal States
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleCount, setSampleCount] = useState(100);
  const[sampleClusters, setSampleClusters] = useState(3);
  const [clusterVariance, setClusterVariance] = useState(0.5);
  const [datasetType, setDatasetType] = useState('blobs');

  const[canvasBounds] = useState({ xMin: -8, xMax: 8, yMin: -8, yMax: 8 });
  const canvasRef = useRef(null);
  const [canvasDimensions] = useState({ width: 600, height: 600 });
  const [results, setResults] = useState(null);

  // --- STANDARD PLAYBACK STATE FOR 2x2 GRID ---
  const[playback, setPlayback] = useState({
      active: false,
      frames:[],
      currentIndex: 0,
      isPlaying: false
  });

  const clusterColors =[
    'rgba(59, 130, 246, 0.7)', 'rgba(239, 68, 68, 0.7)', 'rgba(34, 197, 94, 0.7)',
    'rgba(168, 85, 247, 0.7)', 'rgba(251, 146, 60, 0.7)', 'rgba(236, 72, 153, 0.7)',
    'rgba(14, 165, 233, 0.7)', 'rgba(234, 179, 8, 0.7)', 'rgba(8, 145, 178, 0.7)', 'rgba(124, 58, 237, 0.7)'
  ];

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
        }, 800); // 0.8s per iteration frame
    }
    return () => clearInterval(interval);
  }, [playback.active, playback.isPlaying]);

  const togglePlayback = () => {
      setPlayback(p => {
          if (!p.isPlaying && p.currentIndex >= p.frames.length - 1) {
              return { ...p, currentIndex: 0, isPlaying: true };
          }
          return { ...p, isPlaying: !p.isPlaying };
      });
  };

  const generateSampleData = async () => {
    try {
      setLoading(true); setError(null); setShowSampleDataModal(false);
      const response = await axios.get(
        `${apiUrl}/kmeans/sample?n_samples=${sampleCount}&n_clusters=${sampleClusters}&variance=${clusterVariance}&dataset_type=${datasetType}`
      );
      setData(response.data);
      setResults(null);
      setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    } catch (err) {
      setError('Failed to generate sample data');
    } finally {
      setLoading(false);
    }
  };

  const resetData = () => {
    setData({ X:[] });
    setResults(null);
    setError(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleFactorX = canvas.width / rect.width;
    const scaleFactorY = canvas.height / rect.height;
    const adjustedX = x * scaleFactorX;
    const adjustedY = y * scaleFactorY;
    const dataX = canvasBounds.xMin + (adjustedX / canvas.width) * (canvasBounds.xMax - canvasBounds.xMin);
    const dataY = canvasBounds.yMax - (adjustedY / canvas.height) * (canvasBounds.yMax - canvasBounds.yMin);
    return { x: dataX, y: dataY };
  };

  const handleCanvasClick = (e) => {
    if (loading) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);

    setData({
      ...data,
      X:[...(data.X || []), [dataPoint.x, dataPoint.y]]
    });

    setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const runKMeansModel = async () => {
    try {
      if (!data.X || data.X.length < clusterCount) {
        setError(`Please add at least ${clusterCount} data points`);
        return;
      }

      setLoading(true); setError(null);
      setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });

      const requestData = {
        X: data.X, k: clusterCount, max_iterations: maxIterations
      };

      const response = await axios.post(`${apiUrl}/kmeans`, requestData);
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
      setError(`Failed to run K-means: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Draw Input Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
    const stepX = width / 16; const stepY = height / 16;
    for (let i = 0; i <= 16; i++) { const y = i * stepY; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
    for (let i = 0; i <= 16; i++) { const x = i * stepX; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }

    // Draw axes
    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    const yAxisPos = height / 2; ctx.beginPath(); ctx.moveTo(0, yAxisPos); ctx.lineTo(width, yAxisPos); ctx.stroke();
    const xAxisPos = width / 2; ctx.beginPath(); ctx.moveTo(xAxisPos, 0); ctx.lineTo(xAxisPos, height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) { const x = i * stepX; const value = canvasBounds.xMin + (i / 16) * (canvasBounds.xMax - canvasBounds.xMin); ctx.fillText(value.toFixed(0), x - 8, height - 5); }
    for (let i = 0; i <= 16; i += 2) { const y = i * stepY; const value = canvasBounds.yMax - (i / 16) * (canvasBounds.yMax - canvasBounds.yMin); ctx.fillText(value.toFixed(0), 5, y + 4); }

    // Draw points (always gray in input canvas because it is Unsupervised)
    if (data && data.X && data.X.length > 0) {
      data.X.forEach(point => {
        const x = ((point[0] - canvasBounds.xMin) / (canvasBounds.xMax - canvasBounds.xMin)) * width;
        const y = height - ((point[1] - canvasBounds.yMin) / (canvasBounds.yMax - canvasBounds.yMin)) * height;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(107, 114, 128, 0.7)';
        ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
      });
    } else {
      ctx.fillStyle = 'rgba(100, 100, 100, 0.5)'; ctx.font = '16px Inter, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Click to add data points', width / 2, height / 2);
    }
  }, [data, canvasDimensions, canvasBounds]);

  // Sample Data Modal Component
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
            <button onClick={() => setDatasetType('blobs')} style={{ padding: '0.5rem 0.75rem', backgroundColor: datasetType === 'blobs' ? '#3b82f6' : '#e5e7eb', color: datasetType === 'blobs' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Blobs</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Distinct clusters</span>
            </button>
            <button onClick={() => setDatasetType('moons')} style={{ padding: '0.5rem 0.75rem', backgroundColor: datasetType === 'moons' ? '#3b82f6' : '#e5e7eb', color: datasetType === 'moons' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Moons</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Curved boundaries</span>
            </button>
            <button onClick={() => setDatasetType('circles')} style={{ padding: '0.5rem 0.75rem', backgroundColor: datasetType === 'circles' ? '#3b82f6' : '#e5e7eb', color: datasetType === 'circles' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Circles</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Concentric circles</span>
            </button>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="sample-count" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
          <input id="sample-count" type="range" min="50" max="500" step="50" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>50 (Fewer points)</span><span>500 (More points)</span></div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="variance" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Variance: {clusterVariance.toFixed(1)}</label>
          <input id="variance" type="range" min="0.1" max="2.0" step="0.1" value={clusterVariance} onChange={(e) => setClusterVariance(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.1 (Compact)</span><span>2.0 (Spread out)</span></div>
        </div>

        {datasetType === 'blobs' && (
          <div style={{ marginBottom: '1.25rem' }}>
            <label htmlFor="clusters" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of True Clusters: {sampleClusters}</label>
            <input id="clusters" type="range" min="2" max="8" step="1" value={sampleClusters} onChange={(e) => setSampleClusters(Number(e.target.value))} style={{ width: '100%' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>2 clusters</span><span>8 clusters</span></div>
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
        <h1 className="model-title">K-Means Clustering </h1>
      </div>

      <p className="model-description">
        K-means clustering is an unsupervised learning algorithm that partitions data into K clusters, where each data point belongs to the cluster with the nearest mean.
        <InfoButton algoId="kmeans" />
      </p>

      {backendStatus === "disconnected" && (
        <div className="backend-status error">
          <span>Backend service is not responding. Please make sure the Flask server is running on port 5000.</span>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      <div style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

        {/* TOP ROW: Input & Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

          {/* Left column: Input Plot */}
          <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
            <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 className="section-title">Data Points</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Load Sample Data</button>
                <button className="reset-button" onClick={resetData} disabled={loading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Reset Data</button>
              </div>
            </div>

            <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>Click on the canvas below to add data points manually.</p>

            <div style={{ marginBottom: '1rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', width: '100%', height: '0', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={canvasDimensions.width} height={canvasDimensions.height} onClick={handleCanvasClick} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              </div>
            </div>

            <div style={{ marginTop: '0.5rem', marginBottom: '1rem', backgroundColor: '#f9fafb', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #e5e7eb', width: '100%' }}>
              <p style={{ fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>Statistics:</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div><strong>Data Points:</strong> {data.X ? data.X.length : 0}</div>
                <div><strong>Target Clusters (K):</strong> {clusterCount}</div>
              </div>
            </div>
          </div>

          {/* Right column: Controls */}
          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title" style={{ marginBottom: '1rem' }}>Algorithm Controls</h2>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Clustering Parameters</h3>
              <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Number of Clusters (K): {clusterCount}</label>
                <input type="range" min="2" max="10" step="1" value={clusterCount} onChange={(e) => setClusterCount(Number(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>2 clusters</span><span>10 clusters</span></div>
              </div>
              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Max Iterations: {maxIterations}</label>
                <input type="range" min="10" max="300" step="10" value={maxIterations} onChange={(e) => setMaxIterations(Number(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>10 iterations</span><span>300 iterations</span></div>
              </div>
              <button onClick={runKMeansModel} disabled={loading || backendStatus === "disconnected" || !data.X || data.X.length < clusterCount} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected" || !data.X || data.X.length < clusterCount) ? 0.7 : 1 }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Running...</>
                ) : 'Run K-Means Clustering'}
              </button>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Legend</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', marginBottom: '1.5rem' }}>
                {clusterColors.slice(0, 6).map((color, index) => (
                  <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: color, border: '1px solid #333' }}></div>
                    <span>Cluster {index + 1}</span>
                  </div>
                ))}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(107, 114, 128, 0.7)', border: '1px solid #333' }}></div><span>Unclustered</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><svg width="14" height="14" viewBox="0 0 100 100"><polygon points="50,5 63,38 100,38 69,59 82,96 50,75 18,96 31,59 0,38 37,38" fill="#3b82f6" stroke="#000" strokeWidth="5"/></svg><span>Centroid</span></div>
              </div>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>How K-Means Works</h3>
              <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                <p style={{ marginBottom: '0.5rem' }}>K-Means clusters data in these steps:</p>
                <ol style={{ paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                  <li>Initialize K centroids randomly</li>
                  <li>Assign points to nearest centroid</li>
                  <li>Move centroids to cluster means</li>
                  <li>Repeat until convergence</li>
                </ol>
                <p style={{ fontSize: '0.8rem', fontStyle: 'italic' }}>The interactive plot shows centroids (stars) moving through iterations.</p>
              </div>
            </div>
          </div>
        </div>

        {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>Clustering Results</h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Cluster Evolution Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Cluster Search Evolution</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Iteration: {playback.frames.length > 0 ? playback.frames[playback.currentIndex].iteration : results.iterations}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1.5rem', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                                src={`data:image/png;base64,${playback.frames.length > 0 ? playback.frames[playback.currentIndex].image : results.final_plot}`}
                                alt="Cluster Evolution"
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

                    {/* (0,1) Final Stats */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                Inertia (Sum of Squares): {results.inertia?.toFixed(2)}
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>
                                Silhouette Score: {results.silhouette_score !== null ? results.silhouette_score.toFixed(4) : "N/A"} | Converged at Iteration: {results.iterations}
                            </p>
                        </div>

                        <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center', flex: 1 }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Mathematical Insights</h4>
                            <ul style={{ textAlign: 'left', color: '#4b5563', fontSize: '0.9rem', lineHeight: '1.5', paddingLeft: '1.2rem' }}>
                                <li style={{marginBottom: '10px'}}><strong>Inertia</strong> drops rapidly as centroids move to the center of dense point clouds. Lower inertia means tighter clusters.</li>
                                <li style={{marginBottom: '10px'}}><strong>Silhouette Score</strong> ranges from -1 to 1. A score closer to 1 indicates that points are perfectly mapped to their own cluster and far from neighboring clusters.</li>
                                <li><strong>Convergence</strong> happens when centroids mathematically lock into a local minimum and stop moving.</li>
                            </ul>
                        </div>
                    </div>

                    {/* (1,0) Inertia Convergence Curve */}
                    {results.inertia_history && results.inertia_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Inertia Convergence Curve</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.inertia_history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="iteration" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line type="monotone" dataKey="inertia" stroke="#ef4444" name="Inertia (Error)" strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Inertia plunges as centroids center themselves. If the line is flat, convergence is reached.
                            </p>
                        </div>
                    )}

                    {/* (1,1) Cluster Distribution Bar Chart */}
                    {results.cluster_sizes && results.cluster_sizes.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Cluster Density Distribution</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <BarChart data={results.cluster_sizes} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="cluster" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Bar dataKey="size" name="Number of Points" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Shows how the data points are mathematically distributed across the {clusterCount} clusters.
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

export default KMeans;
