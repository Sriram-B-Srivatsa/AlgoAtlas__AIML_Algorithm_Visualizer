/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo, useLayoutEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function RandomForest() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('classification');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");

  const canvasRef = useRef(null);
  const [selectedClass, setSelectedClass] = useState('1');
  const[pointsMode, setPointsMode] = useState('train');
  const [trainingPoints, setTrainingPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const[hoveredPoint, setHoveredPoint] = useState(null);

  // Sample Data Modal
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const[sampleDataType, setSampleDataType] = useState('blobs');
  const [sampleCount, setSampleCount] = useState(60);
  const [sampleClusters, setSampleClusters] = useState(3);
  const[sampleVariance, setSampleVariance] = useState(0.5);
  const[sampleSparsity, setSampleSparsity] = useState(1.0);

  // Parameters
  const [trees, setTrees] = useState(15);
  const [maxDepth, setMaxDepth] = useState(5);
  const [results, setResults] = useState(null);

  // Playback State
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
        }, 800);
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

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    resetData();
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
    if (loading) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);

    if (pointsMode === 'train') {
        if (activeTab === 'classification') {
          setTrainingPoints([...trainingPoints, { x1: dataPoint.x, x2: dataPoint.y, y: selectedClass }]);
        } else {
          const randomY = ((0.5 * dataPoint.x) + 2 + (Math.random() * 2 - 1)).toFixed(2);
          setTrainingPoints([...trainingPoints, { x1: dataPoint.x, x2: dataPoint.y, y: randomY }]);
        }
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    } else {
        setPredictPoints([...predictPoints, { x1: dataPoint.x, x2: dataPoint.y }]);
    }
  };

  const handleCanvasMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);
    const hoverRadius = 0.5;
    let foundPoint = null;
    for (const point of predictions.length > 0 ? predictions : predictPoints) {
      const dist = Math.sqrt(Math.pow(point.x1 - dataPoint.x, 2) + Math.pow(point.x2 - dataPoint.y, 2));
      if (dist < hoverRadius) { foundPoint = point; break; }
    }
    if (foundPoint !== hoveredPoint) setHoveredPoint(foundPoint);
  };

  const resetData = () => {
    setTrainingPoints([]);
    setPredictPoints([]);
    setPredictions([]);
    setPointsMode('train');
    setResults(null);
    setError(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  // STRICT BACKEND-ONLY DATA GENERATION
  const generateSampleDataWithOptions = () => {
    setLoading(true);
    setShowSampleDataModal(false);
    setError(null);

    axios.post(`${apiUrl}/rf/sample_data`, {
      type: activeTab,
      dataset_type: sampleDataType,
      count: sampleCount,
      n_clusters: sampleClusters,
      variance: sampleVariance,
      sparsity: sampleSparsity
    })
    .then(response => {
      if (response.data && response.data.points) {
        setTrainingPoints(response.data.points);
        setPredictPoints([]);
        setPredictions([]);
        setPointsMode('train');
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
      } else if (response.data.error) {
        setError(`Backend Error: ${response.data.error}`);
      }
    })
    .catch(err => {
        setError(`Network Error: Failed to fetch sample data from backend. (${err.message})`);
    })
    .finally(() => setLoading(false));
  };

  const trainModel = () => {
    if (trainingPoints.length < 2) {
      setError("Need at least 2 training points");
      return;
    }

    setLoading(true);
    setError(null);

    const formattedPoints = trainingPoints.map(point => {
      const x1 = Number(point.x1); const x2 = Number(point.x2);
      if (activeTab === 'classification') return { x1, x2, class: String(point.y || '') };
      else return { x1, x2, value: Number(point.y) };
    });

    const requestData = {
      X: formattedPoints.map(p => [p.x1, p.x2]),
      y: formattedPoints.map(p => activeTab === 'classification' ? p.class : p.value),
      type: activeTab,
      parameters: { trees: trees, max_depth: maxDepth }
    };

    axios.post(`${apiUrl}/rf/train`, requestData)
      .then(response => {
        if (response.data.error) { setError(response.data.error); return; }

        setResults(response.data);
        if (response.data.history && response.data.history.length > 0) {
            setPlayback({
                active: true,
                frames: response.data.history,
                currentIndex: response.data.history.length - 1,
                isPlaying: false
            });
        }
      })
      .catch(err => setError("Failed to train model. Ensure backend has /api/rf/train mapped!"))
      .finally(() => setLoading(false));
  };

  const handlePrediction = () => {
    if (predictPoints.length === 0) { setError("Add points to predict first!"); return; }
    if (!results) { setError("Train the model first!"); return; }

    setLoading(true); setError(null);

    const formattedTrain = trainingPoints.map(point => {
        const x1 = Number(point.x1); const x2 = Number(point.x2);
        return activeTab === 'classification' ? { x1, x2, class: String(point.y) } : { x1, x2, value: Number(point.y) };
    });

    const requestData = {
      trained_points: {
          X: formattedTrain.map(p =>[p.x1, p.x2]),
          y: formattedTrain.map(p => activeTab === 'classification' ? p.class : p.value)
      },
      predict_points: predictPoints.map(p =>[parseFloat(p.x1), parseFloat(p.x2)]),
      type: activeTab,
      parameters: { trees: trees, max_depth: maxDepth }
    };

    axios.post(`${apiUrl}/rf/predict`, requestData)
    .then(response => {
      if (response.data.error) { setError(response.data.error); return; }

      const res = response.data.predictions ||[];
      const formattedResults = res.map((r, idx) => ({
        x1: predictPoints[idx].x1,
        x2: predictPoints[idx].x2,
        predictedClass: r
      }));
      setPredictions(formattedResults);
    })
    .catch(err => setError("Failed to make prediction. Ensure backend has /api/rf/predict mapped!"))
    .finally(() => setLoading(false));
  };

  // Canvas drawing effect
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
    const stepX = canvasWidth / 16; const stepY = canvasHeight / 16;
    for (let i = 0; i <= 16; i++) { const y = i * stepY; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasWidth, y); ctx.stroke(); }
    for (let i = 0; i <= 16; i++) { const x = i * stepX; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvasHeight); ctx.stroke(); }

    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    const yAxisPos = canvasHeight / 2; ctx.beginPath(); ctx.moveTo(0, yAxisPos); ctx.lineTo(canvasWidth, yAxisPos); ctx.stroke();
    const xAxisPos = canvasWidth / 2; ctx.beginPath(); ctx.moveTo(xAxisPos, 0); ctx.lineTo(xAxisPos, canvasHeight); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) { const x = i * stepX; const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min); ctx.fillText(value.toFixed(0), x - 8, canvasHeight - 5); }
    for (let i = 0; i <= 16; i += 2) { const y = i * stepY; const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min); ctx.fillText(value.toFixed(0), 5, y + 4); }

    // Draw Training Points
    trainingPoints.forEach(point => {
      const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
      const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);

      if (activeTab === 'classification') {
        const pointClass = point.y.toString();
        if (pointClass === '0') ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
        else if (pointClass === '1') ctx.fillStyle = 'rgba(239, 68, 68, 0.7)';
        else if (pointClass === '2') ctx.fillStyle = 'rgba(34, 197, 94, 0.7)';
        else ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
      } else {
        ctx.fillStyle = 'rgba(139, 92, 246, 0.7)';
        ctx.fill();
        ctx.fillStyle = '#000000'; ctx.font = '10px Arial'; ctx.fillText(point.y, x + 10, y - 10);
        ctx.fillStyle = 'rgba(139, 92, 246, 0.7)';
      }
      ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
    });

    // Draw Unpredicted Points
    if (predictions.length === 0) {
        predictPoints.forEach(point => {
          const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
          const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
          const isHovered = hoveredPoint && hoveredPoint.x1 === point.x1 && hoveredPoint.x2 === point.x2;
          ctx.beginPath(); ctx.arc(x, y, isHovered ? 8 : 6, 0, Math.PI * 2);
          ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)';
          ctx.fill(); ctx.strokeStyle = isHovered ? '#000' : '#333'; ctx.lineWidth = isHovered ? 2 : 1; ctx.stroke();
        });
    }

    // Draw Predicted Points
    predictions.forEach(pred => {
        if (!pred || pred.x1 === undefined || pred.x2 === undefined || pred.predictedClass === undefined) return;
        const x = ((pred.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
        const y = ((scale.y.max - pred.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
        const isHovered = hoveredPoint && hoveredPoint.x1 === pred.x1 && hoveredPoint.x2 === pred.x2;

        ctx.beginPath(); ctx.arc(x, y, isHovered ? 8 : 6, 0, Math.PI * 2);

        if (activeTab === 'classification') {
          const predClass = String(pred.predictedClass).trim();

          // 1. SET THE BRUSH COLOR FIRST
          if (predClass === '0') ctx.fillStyle = isHovered ? 'rgba(30, 64, 175, 0.9)' : 'rgba(59, 130, 246, 0.7)';
          else if (predClass === '1') ctx.fillStyle = isHovered ? 'rgba(185, 28, 28, 0.9)' : 'rgba(239, 68, 68, 0.7)';
          else if (predClass === '2') ctx.fillStyle = isHovered ? 'rgba(21, 128, 61, 0.9)' : 'rgba(34, 197, 94, 0.7)';
          else ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)';

          // 2. FILL THE CIRCLE BEFORE CHANGING THE BRUSH TO BLACK
          ctx.fill();

          // 3. NOW DRAW THE BLACK TEXT
          ctx.fillStyle = '#000000'; ctx.font = '10px Arial'; ctx.fillText(String(pred.predictedClass), x + 10, y - 10);
        } else {
            ctx.fillStyle = 'rgba(139, 92, 246, 0.7)';
            ctx.fill();
            ctx.fillStyle = '#000000'; ctx.font = '10px Arial'; ctx.fillText(Number(pred.predictedClass).toFixed(2), x + 10, y - 10);
        }

        ctx.setLineDash([2, 2]); ctx.strokeStyle = isHovered ? '#000' : '#333'; ctx.lineWidth = isHovered ? 2 : 1.5; ctx.stroke(); ctx.setLineDash([]);
      });

  },[trainingPoints, predictPoints, predictions, scale, canvasDimensions, activeTab, hoveredPoint]);

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '8px', width: '90%', maxWidth: '500px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '600' }}>
            Generate Sample Data ({activeTab === 'classification' ? 'Classification' : 'Regression'})
          </h2>
          <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
        </div>

        {activeTab === 'classification' ? (
          <>
            {/* Classification options */}
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
              <label htmlFor="variance" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Variance: {sampleVariance.toFixed(1)}</label>
              <input id="variance" type="range" min="0.1" max="1.0" step="0.1" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.1 (Compact)</span><span>1.0 (Spread out)</span></div>
            </div>

            {sampleDataType === 'blobs' && (
              <div style={{ marginBottom: '1.25rem' }}>
                <label htmlFor="clusters" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Clusters: {sampleClusters}</label>
                <input id="clusters" type="range" min="1" max="3" step="1" value={sampleClusters} onChange={(e) => setSampleClusters(Number(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>1 clusters</span><span>3 clusters</span></div>
              </div>
            )}
          </>
        ) : (
          <>
            {/* Regression options */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Regression Function Type</label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                <button onClick={() => setSampleDataType('linear')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'linear' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'linear' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <span style={{ fontWeight: '500' }}>Linear</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Simple linear pattern</span>
                </button>
                <button onClick={() => setSampleDataType('nonlinear')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleDataType === 'nonlinear' ? '#3b82f6' : '#e5e7eb', color: sampleDataType === 'nonlinear' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <span style={{ fontWeight: '500' }}>Nonlinear</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Complex regions</span>
                </button>
              </div>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label htmlFor="reg-sample-count" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
              <input id="reg-sample-count" type="range" min="20" max="200" step="10" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>20 (Fewer points)</span><span>200 (More points)</span></div>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label htmlFor="reg-variance" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Noise Level: {sampleVariance.toFixed(1)}</label>
              <input id="reg-variance" type="range" min="0.1" max="1.5" step="0.1" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.1 (Low noise)</span><span>1.5 (High noise)</span></div>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label htmlFor="reg-sparsity" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Data Sparsity: {sampleSparsity.toFixed(1)}</label>
              <input id="reg-sparsity" type="range" min="0.5" max="3.0" step="0.1" value={sampleSparsity} onChange={(e) => setSampleSparsity(Number(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.5 (Dense points)</span><span>3.0 (Sparse clusters)</span></div>
            </div>
          </>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleDataWithOptions} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Generate Data</button>
        </div>
      </div>
    </div>
  );

  useLayoutEffect(() => {
          window.history.scrollRestoration = "manual";

          const resetScroll = () => {
              window.scrollTo(0, 0);
              document.documentElement.scrollTop = 0;
              document.body.scrollTop = 0;
          };

          resetScroll();

          requestAnimationFrame(() => {
              resetScroll();

              setTimeout(resetScroll, 0);
              setTimeout(resetScroll, 100);
              setTimeout(resetScroll, 500);
          });
  }, []);

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      {showSampleDataModal && <SampleDataModal />}
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Hub</span>
        </button>
        <h1 className="model-title">Random Forest </h1>
      </div>

      <p className="model-description">
        Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees to prevent overfitting.
        <InfoButton algoId="rf" />
      </p>

      {backendStatus === "disconnected" && (
        <div className="backend-status error">
          <span>Backend service is not responding. Please make sure the Flask server is running on port 5000.</span>
        </div>
      )}

      {error && (
        <div className="error-message">
          <span>{error}</span>
        </div>
      )}

      <div className="tabs">
        <div className={`tab ${activeTab === 'classification' ? 'active' : ''}`} onClick={() => handleTabChange('classification')}>Classification</div>
        <div className={`tab ${activeTab === 'regression' ? 'active' : ''}`} onClick={() => handleTabChange('regression')}>Regression</div>
      </div>

      <div className="content-container" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

        {/* ROW 1: INPUT AND PARAMS */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

          <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Random Forest {activeTab === 'classification' ? 'Classification' : 'Regression'}</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{ backgroundColor: '#3b82f6', color: 'white'}}>{loading ? 'Load Sample Data' : 'Load Sample Data'}</button>
                <button className="sample-data-button" onClick={resetData} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }} disabled={loading}>Reset Data</button>
              </div>
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>
                Click on the graph below to add data points. Current mode: <strong>{pointsMode === 'train' ? 'Training' : 'Prediction'}</strong>
              </p>
            </div>

            <div style={{ marginBottom: '1rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', width: '100%', height: '0', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={canvasDimensions.width} height={canvasDimensions.height} onClick={handleCanvasClick} onMouseMove={handleCanvasMouseMove} style={{ display: 'block', cursor: loading ? 'wait' : 'crosshair', width: '100%', height: '100%' }} />
              </div>
              <div style={{ position: 'absolute', bottom: '10px', right: '10px', padding: '4px 8px', backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '4px', fontSize: '0.8rem', color: '#4b5563', pointerEvents: 'none' }}>
                {pointsMode === 'train' ? 'Click to add training point' : 'Click to add prediction point'}
              </div>
            </div>

            {/* Statistics */}
            <div style={{ marginTop: '0.5rem', marginBottom: '1rem', backgroundColor: '#f9fafb', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #e5e7eb', width: '100%' }}>
              <p style={{ fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>Statistics:</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                <div><strong>Training Points:</strong> {trainingPoints.length}</div>
                <div><strong>Points to Predict:</strong> {predictPoints.length}</div>
                <div><strong>Predictions Made:</strong> {predictions.length}</div>
              </div>
            </div>

            <div style={{ width: '100%', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', fontSize: '0.85rem', marginBottom: '1rem' }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>Legend</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.7)', border: '1px solid #333' }}></div><span>Class 0</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(239, 68, 68, 0.7)', border: '1px solid #333' }}></div><span>Class 1</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(34, 197, 94, 0.7)', border: '1px solid #333' }}></div><span>Class 2</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(156, 163, 175, 0.7)', border: '1px solid #333', borderStyle: 'dashed' }}></div><span>Unpredicted</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'white', border: '2px dashed #333' }}></div><span>Predicted</span></div>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Controls & Results</h2>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Ensemble Parameters</h3>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Total Trees in Forest: {trees}</label>
                <input type="range" min="1" max="50" step="1" value={trees} onChange={(e) => setTrees(parseInt(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  <span>1 Tree (Weak)</span><span>50 Trees (Robust)</span>
                </div>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Max Depth per Tree: {maxDepth}</label>
                <input type="range" min="1" max="10" step="1" value={maxDepth} onChange={(e) => setMaxDepth(parseInt(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  <span>1 (Shallow)</span><span>10 (Deep)</span>
                </div>
              </div>

              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Interaction Mode</h3>
              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
                <button onClick={() => setPointsMode('train')} style={{ padding: '0.75rem 1rem', backgroundColor: pointsMode === 'train' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'train' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Training Points</button>
                <button onClick={() => setPointsMode('predict')} style={{ padding: '0.75rem 1rem', backgroundColor: pointsMode === 'predict' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'predict' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Prediction Points</button>
              </div>

              {pointsMode === 'train' && activeTab === 'classification' && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <p style={{ marginBottom: '0.75rem', color: '#4b5563', fontSize: '1rem' }}>Select class for training points:</p>
                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button onClick={() => setSelectedClass('0')} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: selectedClass === '0' ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.1)', color: selectedClass === '0' ? 'white' : '#1e40af', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Class 0</button>
                    <button onClick={() => setSelectedClass('1')} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: selectedClass === '1' ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.1)', color: selectedClass === '1' ? 'white' : '#b91c1c', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Class 1</button>
                    <button onClick={() => setSelectedClass('2')} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: selectedClass === '2' ? 'rgba(34, 197, 94, 1)' : 'rgba(34, 197, 94, 0.1)', color: selectedClass === '2' ? 'white' : '#15803d', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Class 2</button>
                  </div>
                </div>
              )}
            </div>

            <div style={{ marginBottom: '1rem', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Actions</h3>
              <button onClick={trainModel} disabled={loading || backendStatus === "disconnected" || trainingPoints.length < 2} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected" || trainingPoints.length < 2) ? 0.7 : 1, marginBottom: '1.0rem' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Growing Forest...</>
                ) : 'Train Random Forest'}
              </button>

              <button onClick={handlePrediction} disabled={loading || backendStatus === "disconnected" || trainingPoints.length < 2 || predictPoints.length < 1 || !results} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#8b5cf6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected" || trainingPoints.length < 2 || predictPoints.length < 1 || !results) ? 0.7 : 1 }}>
                {loading ? 'Predict Points' : 'Predict Points'}
              </button>
            </div>

            <div style={{ width: '100%', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', fontSize: '0.85rem' }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>Why use a Forest?</h3>
              <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                <p style={{ marginBottom: '0.5rem' }}>A single Decision Tree is prone to <strong>overfitting</strong> (it memorizes the noise in the data). Random Forests fix this by:</p>
                <ol style={{ paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                  <li>Creating many trees on random subsets of data (Bagging).</li>
                  <li>Limiting the features each tree can look at.</li>
                  <li>Averaging the votes of all trees to produce a smooth, robust boundary.</li>
                </ol>
              </div>
            </div>
          </div>
        </div>

        {/* ROW 2 & 3: THE 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Forest Growth Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Boundary Evolution</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Trees in Forest: {playback.frames.length > 0 ? playback.frames[playback.currentIndex].trees : results.trees}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1.5rem', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                                src={`data:image/png;base64,${playback.frames.length > 0 ? playback.frames[playback.currentIndex].boundary : results.history[results.history.length-1].boundary}`}
                                alt="Decision Boundary"
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

                    {/* (0,1) Final Stats with Mathematical Insights explaining 100% Accuracy */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                Validation {activeTab === 'classification' ? 'Accuracy' : 'R² Score'}: {((activeTab === 'classification' ? results.final_val_score : results.r2) * 100).toFixed(2)}%
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>
                                Training Score: {(results.final_train_score * 100).toFixed(2)}% | Total Trees: {results.trees}
                            </p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.90rem', lineHeight: '1.6', textAlign: 'left' }}>
                                <strong>Why is Training Accuracy often 100%?</strong> On simple datasets, Random Forests can perfectly memorize (overfit) every point, hitting 100%. This is an illusion of perfection. <br/><br/>
                                To show real-world performance, this backend automatically performs a <strong>70/30 Train/Validation Split</strong>. Look at the graph below: the Validation line shows how the model *actually* performs on unseen data!
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Ensemble Learning Curve (Train vs Validation) */}
                    {results.history && results.history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                                Ensemble Learning Curve (Train vs Val)
                            </h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="trees" tick={{fontSize: 12}} />
                                        <YAxis domain={activeTab === 'classification' ? [0, 1] : ['auto', 'auto']} tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line type="monotone" dataKey="train_score" stroke="#3b82f6" name={activeTab === 'classification' ? "Training Accuracy" : "Train MSE"} strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                        <Line type="monotone" dataKey="val_score" stroke="#10b981" name={activeTab === 'classification' ? "Validation Accuracy" : "Val MSE"} strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Notice the gap between Train and Validation. Adding more trees helps close this gap by reducing overfitting!
                            </p>
                        </div>
                    )}

                    {/* (1,1) Feature Importance Profile */}
                    {results.feature_importance && results.feature_importance.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Feature Importance</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <BarChart data={results.feature_importance} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="feature" tick={{fontSize: 12}} />
                                        <YAxis domain={[0, 100]} tick={{fontSize: 12}} />
                                        <Tooltip formatter={(value) => `${value.toFixed(2)}%`}/>
                                        <Legend verticalAlign="top" height={36}/>
                                        <Bar dataKey="importance" name="Information Gain (%)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Shows which axis (Feature X1 or X2) provided the most mathematical information for splitting the data.
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

export default RandomForest;
