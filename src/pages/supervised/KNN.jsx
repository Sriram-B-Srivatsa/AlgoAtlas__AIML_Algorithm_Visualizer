import React, { useState, useRef, useEffect, useMemo, useLayoutEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import axios from 'axios';
import { checkHealth, predictKnnPoint, getKnnDecisionBoundary } from '../../api';
import InfoButton from '../../components/InfoButton';

function KNN() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('classification');
  const[neighbors, setNeighbors] = useState(5);
  const [loading, setLoading] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");

  const canvasRef = useRef(null);
  const[selectedClass, setSelectedClass] = useState('1');
  const [pointsMode, setPointsMode] = useState('train');
  const [trainingPoints, setTrainingPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const canvasDimensions = { width: 600, height: 600 };

  const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -8, max: 8 } }), []);

  const [showPointDialog, setShowPointDialog] = useState(false);
  const [dialogPosition, setDialogPosition] = useState({ x: 0, y: 0 });
  const [tempPoint, setTempPoint] = useState(null);
  const[tempValue, setTempValue] = useState('');

  const [results, setResults] = useState(null);
  const[hoveredPoint, setHoveredPoint] = useState(null);
  const [nearestNeighbors, setNearestNeighbors] = useState([]);

  // --- STANDARD PLAYBACK STATE FOR 2x2 GRID ---
  const [playback, setPlayback] = useState({
      active: false,
      frames:[],
      currentIndex: 0,
      isPlaying: false
  });

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    resetData();
  };

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
          }, 1000); // 1 second per K-value jump
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

  const handleClassificationCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || loading) return;
    const rect = canvas.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);

    if (pointsMode === 'train') {
      const newPoint = { x1: dataPoint.x, x2: dataPoint.y, y: selectedClass };
      setTrainingPoints([...trainingPoints, newPoint]);
      setResults(null);
      setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    } else {
      setPredictPoints([...predictPoints, { x1: dataPoint.x, x2: dataPoint.y }]);
    }
  };

  const handleRegressionCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || loading) return;
    const rect = canvas.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);

    if (pointsMode === 'train') {
      setTempPoint({ x1: dataPoint.x, x2: dataPoint.y });
      let dialogX = e.clientX - 100;
      let dialogY = e.clientY - 130;
      if (dialogX < 10) dialogX = 10;
      if (dialogX > window.innerWidth - 210) dialogX = window.innerWidth - 210;
      if (dialogY < 10) dialogY = e.clientY + 10;

      setDialogPosition({ x: dialogX, y: dialogY });
      setTempValue('');
      setShowPointDialog(true);
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

    if (foundPoint !== hoveredPoint) {
      setHoveredPoint(foundPoint);
      if (foundPoint && trainingPoints.length > 0) {
        const k = neighbors;
        const distances = trainingPoints.map(tp => ({
          point: tp,
          distance: Math.sqrt(Math.pow(tp.x1 - foundPoint.x1, 2) + Math.pow(tp.x2 - foundPoint.x2, 2))
        }));
        distances.sort((a, b) => a.distance - b.distance);
        setNearestNeighbors(distances.slice(0, k).map(d => d.point));
      } else {
        setNearestNeighbors([]);
      }
    }
  };

  const handleCanvasMouseLeave = () => { setHoveredPoint(null); setNearestNeighbors([]); };

  const handleDialogConfirm = () => {
    const numValue = parseFloat(tempValue);
    if (tempPoint && !isNaN(numValue)) {
      setTrainingPoints([...trainingPoints, { x1: tempPoint.x1, x2: tempPoint.x2, y: tempValue }]);
      setShowPointDialog(false); setTempPoint(null); setError(null);
    } else {
      setError('Please enter a valid number for the value');
    }
  };

  const handleCanvasClick = (e) => {
    if (activeTab === 'classification') handleClassificationCanvasClick(e);
    else handleRegressionCanvasClick(e);
  };

  const predictAllPoints = async () => {
    if (trainingPoints.length < 1) { setError('Need at least 1 training point'); return; }
    if (predictPoints.length < 1) { setError('No prediction points added.'); return; }

    try {
      setLoading('predict'); setError(null);
      const newPredictions =[];

      for (const point of predictPoints) {
        const apiData = {
          X: trainingPoints.map(p => [p.x1, p.x2]),
          y: trainingPoints.map(p => p.y),
          n_neighbors: parseInt(neighbors),
          predict_point: [point.x1, point.x2],
          mode: activeTab
        };
        const result = await predictKnnPoint(apiData);
        newPredictions.push({ ...point, predictedClass: result.predicted_class });
      }
      setPredictions(newPredictions);
    } catch (err) {
      setError('Error making prediction: ' + err.message);
    } finally {
      setLoading(null);
    }
  };

  const generateDecisionBoundary = async () => {
    if (trainingPoints.length < 5) { setError('Need at least 5 training points'); return; }
    try {
      setLoading('train'); setError(null); setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });

      const apiData = {
        X: trainingPoints.map(p => [p.x1, p.x2]),
        y: trainingPoints.map(p => p.y),
        n_neighbors: parseInt(neighbors),
        mode: activeTab
      };

      const result = await getKnnDecisionBoundary(apiData);

      if (result.error) throw new Error(result.error);

      setResults(result);

      // Filter history to only include frames that actually have images
      const framesWithImages = result.history.filter(h => h.image !== null);
      if (framesWithImages.length > 0) {
        setPlayback({
            active: true,
            frames: framesWithImages,
            currentIndex: framesWithImages.length - 1, // Start at the end
            isPlaying: false
        });
      }

    } catch (err) {
      setError('Error generating boundary: ' + err.message);
    } finally {
      setLoading(null);
    }
  };

  const resetData = () => {
    setTrainingPoints([]); setPredictPoints([]); setPredictions([]); setPointsMode('train'); setError(null);
    setResults(null); setHoveredPoint(null); setNearestNeighbors([]);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  useEffect(() => {
    const handleKeyDown = (e) => { if (e.key === 'Escape' && showPointDialog) setShowPointDialog(false); };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showPointDialog]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const drawCanvas = () => {
      const canvasWidth = canvas.width; const canvasHeight = canvas.height;
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

      trainingPoints.forEach(point => {
        const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
        const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
        const isNearest = nearestNeighbors.some(np => np.x1 === point.x1 && np.x2 === point.x2 && np.y === point.y);
        ctx.beginPath(); ctx.arc(x, y, isNearest ? 8 : 6, 0, Math.PI * 2);

        if (activeTab === 'classification') {
          const pointClass = point.y.toString();
          if (pointClass === '1') ctx.fillStyle = isNearest ? 'rgba(30, 64, 175, 0.9)' : 'rgba(59, 130, 246, 0.7)';
          else if (pointClass === '2') ctx.fillStyle = isNearest ? 'rgba(185, 28, 28, 0.9)' : 'rgba(239, 68, 68, 0.7)';
          else if (pointClass === '3') ctx.fillStyle = isNearest ? 'rgba(21, 128, 61, 0.9)' : 'rgba(34, 197, 94, 0.7)';
          else ctx.fillStyle = isNearest ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)';
        } else {
          const normalizedValue = Math.min(Math.max(parseFloat(point.y) / 10, 0), 1);
          const r = Math.round(59 + (239 - 59) * normalizedValue); const g = Math.round(130 + (68 - 130) * normalizedValue); const b = Math.round(246 + (68 - 246) * normalizedValue);
          ctx.fillStyle = isNearest ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` : `rgba(${r}, ${g}, ${b}, 0.7)`;
          ctx.fillStyle = '#000000'; ctx.font = '10px Arial'; ctx.fillText(point.y, x + 10, y - 10);
          ctx.fillStyle = isNearest ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` : `rgba(${r}, ${g}, ${b}, 0.7)`;
        }
        ctx.fill(); ctx.strokeStyle = isNearest ? '#000' : '#333'; ctx.lineWidth = isNearest ? 2 : 1; ctx.stroke();

        if (isNearest && hoveredPoint) {
          const hpX = ((hoveredPoint.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
          const hpY = ((scale.y.max - hoveredPoint.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
          ctx.beginPath(); ctx.setLineDash([3, 3]); ctx.moveTo(x, y); ctx.lineTo(hpX, hpY); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
        }
      });

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

      predictions.forEach(pred => {
        const x = ((pred.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
        const y = ((scale.y.max - pred.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
        const isHovered = hoveredPoint && hoveredPoint.x1 === pred.x1 && hoveredPoint.x2 === pred.x2;
        ctx.beginPath(); ctx.arc(x, y, isHovered ? 8 : 6, 0, Math.PI * 2);

        if (activeTab === 'classification') {
          const predClass = pred.predictedClass.toString();
          if (predClass.startsWith('1') || Math.round(parseFloat(predClass)) === 1) ctx.fillStyle = isHovered ? 'rgba(30, 64, 175, 0.9)' : 'rgba(59, 130, 246, 0.7)';
          else if (predClass.startsWith('2') || Math.round(parseFloat(predClass)) === 2) ctx.fillStyle = isHovered ? 'rgba(185, 28, 28, 0.9)' : 'rgba(239, 68, 68, 0.7)';
          else if (predClass.startsWith('3') || Math.round(parseFloat(predClass)) === 3) ctx.fillStyle = isHovered ? 'rgba(21, 128, 61, 0.9)' : 'rgba(34, 197, 94, 0.7)';
          else ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)';
        } else {
          const normalizedValue = Math.min(Math.max(parseFloat(pred.predictedClass) / 10, 0), 1);
          const r = Math.round(59 + (239 - 59) * normalizedValue); const g = Math.round(130 + (68 - 130) * normalizedValue); const b = Math.round(246 + (68 - 246) * normalizedValue);
          ctx.fillStyle = isHovered ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` : `rgba(${r}, ${g}, ${b}, 0.7)`;
          ctx.fillStyle = '#000000'; ctx.font = '10px Arial'; ctx.fillText(pred.predictedClass, x + 10, y - 10);
          ctx.fillStyle = isHovered ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` : `rgba(${r}, ${g}, ${b}, 0.7)`;
        }

        ctx.setLineDash([2, 2]); ctx.strokeStyle = isHovered ? '#000' : '#333'; ctx.lineWidth = isHovered ? 2 : 1.5; ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
      });
    };
    drawCanvas();
  },[trainingPoints, predictPoints, predictions, scale, canvasDimensions, activeTab, hoveredPoint, nearestNeighbors]);

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
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
        </button>
        <h1 className="model-title">K-Nearest Neighbors (KNN) </h1>
      </div>

      <p className="model-description">
        K-Nearest Neighbors makes predictions based on k most similar data points in training set.
        <InfoButton algoId="knn" />
      </p>

      {backendStatus === "disconnected" && (
        <div className="backend-status error">
          <span>Backend service is not responding. Please make sure the Flask server is running on port 5000.</span>
        </div>
      )}

      <div className="tabs">
        <div className={`tab ${activeTab === 'classification' ? 'active' : ''}`} onClick={() => handleTabChange('classification')}>Classification</div>
        <div className={`tab ${activeTab === 'regression' ? 'active' : ''}`} onClick={() => handleTabChange('regression')}>Regression</div>
      </div>

      <div className="content-container" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

        {/* TOP ROW: Input & Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

          <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">KNN {activeTab === 'classification' ? 'Classification' : 'Regression'}</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={resetData} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Reset Data</button>
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <p style={{ color: '#4b5563', marginBottom: '1rem', lineHeight: '1.5' }}>
                Click on graph below to add points (atleast 5). Current mode: <strong>{pointsMode === 'train' ? 'Training' : 'Prediction'}</strong>
            </p>

            <div style={{ marginBottom: '1rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', width: '100%', height: '0', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={canvasDimensions.width} height={canvasDimensions.height} onClick={handleCanvasClick} onMouseMove={handleCanvasMouseMove} onMouseLeave={handleCanvasMouseLeave} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              </div>
              <div style={{ position: 'absolute', bottom: '10px', right: '10px', padding: '4px 8px', backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '4px', fontSize: '0.8rem', color: '#4b5563', pointerEvents: 'none' }}>
                {pointsMode === 'train' ? 'Click to add training point' : 'Click to add prediction point'}
              </div>
            </div>

            <div style={{ marginTop: '0.5rem', marginBottom: '1rem', backgroundColor: '#f9fafb', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #e5e7eb', width: '100%' }}>
              <p style={{ fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>Statistics:</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                <div><strong>Training Points:</strong> {trainingPoints.length}</div>
                <div><strong>Points to Predict:</strong> {predictPoints.length}</div>
                <div><strong>Predictions Made:</strong> {predictions.length}</div>
              </div>
            </div>
          </div>

          {/* Right column: Controls */}
          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Controls & Results</h2>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Parameters</h3>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="neighbors-slider" style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Number of Neighbors (k): {neighbors}</label>
                <input id="neighbors-slider" type="range" min="1" max="15" step="1" value={neighbors} onChange={(e) => setNeighbors(parseInt(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  <span>1 (Overfitted)</span><span>15 (Generalized)</span>
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
                    <button onClick={() => setSelectedClass('1')} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: selectedClass === '1' ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.1)', color: selectedClass === '1' ? 'white' : '#1e40af', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Class 1</button>
                    <button onClick={() => setSelectedClass('2')} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: selectedClass === '2' ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.1)', color: selectedClass === '2' ? 'white' : '#b91c1c', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Class 2</button>
                    <button onClick={() => setSelectedClass('3')} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: selectedClass === '3' ? 'rgba(34, 197, 94, 1)' : 'rgba(34, 197, 94, 0.1)', color: selectedClass === '3' ? 'white' : '#15803d', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Class 3</button>
                  </div>
                </div>
              )}
            </div>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Actions</h3>
              <button onClick={generateDecisionBoundary} disabled={loading || backendStatus === "disconnected" || trainingPoints.length < 5} style={{ width: '100%', backgroundColor: loading ? '#c4b5fd' : '#8b5cf6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected" || trainingPoints.length < 5) ? 0.7 : 1 }}>
                {loading === 'train' ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Generating...</>
                ) : activeTab === 'classification' ? 'Show Decision Regions' : 'Show Regression Surface'}
              </button>

              <button onClick={predictAllPoints} disabled={loading || backendStatus === "disconnected" || trainingPoints.length < 1 || predictPoints.length < 1} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected" || trainingPoints.length < 1 || predictPoints.length < 1) ? 0.7 : 1, marginTop: '1.25rem' }}>
                {loading === 'predict' ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Predicting Points...</>
                ) : 'Predict Points'}
              </button>
            </div>

            <div style={{ width: '100%', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', fontSize: '0.85rem', marginBottom: '1rem' }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>Legend</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.7)', border: '1px solid #333' }}></div><span>Class 1</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(239, 68, 68, 0.7)', border: '1px solid #333' }}></div><span>Class 2</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(34, 197, 94, 0.7)', border: '1px solid #333' }}></div><span>Class 3</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', backgroundColor: 'rgba(156, 163, 175, 0.7)', border: '1px solid #333' }}></div><span>Unpredicted</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'white', border: '2px dashed #333' }}></div><span>Predicted</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(30, 64, 175, 0.9)', border: '2px solid #000' }}></div><span>Neighbors (Hover)</span></div>
              </div>
            </div>
          </div>
        </div>

        {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>Model Evaluation</h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Boundary Evolution Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Boundary Evolution</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Neighbors (K): {playback.frames.length > 0 ? playback.frames[playback.currentIndex].k : results.metrics.chosen_k}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1.5rem', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                                src={`data:image/png;base64,${playback.frames.length > 0 ? playback.frames[playback.currentIndex].image : results.decision_boundary}`}
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

                    {/* (0,1) Final Stats and Mathematical Insights */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                {activeTab === 'classification' ? 'Accuracy' : 'MSE'}: {(activeTab === 'classification' ? results.metrics.val_score * 100 : results.metrics.val_score).toFixed(2)}{activeTab === 'classification' ? '%' : ''}
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>
                                Chosen Neighbors (K): {results.metrics.chosen_k} | Validation Method: 70/30 Split
                            </p>
                        </div>

                        <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center', flex: 1 }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Mathematical Insights</h4>
                            <ul style={{ textAlign: 'left', color: '#4b5563', fontSize: '0.9rem', lineHeight: '1.5', paddingLeft: '1.2rem' }}>
                                <li style={{marginBottom: '10px'}}><strong>K=1 (Overfitting):</strong> Boundary tightly wraps around every single training point. The Training Error is 0%, but Validation Error is high because it memorizes noise.</li>
                                <li style={{marginBottom: '10px'}}><strong>Optimal K:</strong> Creates a smooth boundary that ignores local outliers but captures the true global pattern of the data.</li>
                                <li><strong>High K (Underfitting):</strong> The boundary becomes completely rigid and smooth, ignoring the structure of the data and increasing both Train and Validation errors.</li>
                            </ul>
                        </div>
                    </div>

                    {/* (1,0) Validation Curve Graph */}
                    {results.history && results.history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                                {activeTab === 'classification' ? 'Accuracy vs. K-Value' : 'MSE vs. K-Value'}
                            </h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="k" tick={{fontSize: 12}} />
                                        <YAxis domain={activeTab === 'classification' ? [0, 1] : ['auto', 'auto']} tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line type="monotone" dataKey="train_score" stroke="#3b82f6" name="Training Score" strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                        <Line type="monotone" dataKey="val_score" stroke="#10b981" name="Validation Score" strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Training score always worsens as K increases. The optimal K is where the Validation Score peaks (or bottoms out for MSE).
                            </p>
                        </div>
                    )}

                    {/* (1,1) Final Decision Boundary Image */}
                    {results.decision_boundary && (
                        <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem', width: '100%' }}>
                                Final {activeTab === 'classification' ? 'Decision Boundary' : 'Regression Surface'}
                            </h4>
                            <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
                                <img
                                    src={`data:image/png;base64,${results.decision_boundary}`}
                                    alt="Final KNN Boundary"
                                    style={{ width: '100%', maxWidth: '400px', borderRadius: '4px' }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </motion.div>
        )}
      </div>

      {showPointDialog && (
        <div style={{ position: 'fixed', top: dialogPosition.y, left: dialogPosition.x, backgroundColor: 'white', padding: '1rem', borderRadius: '0.5rem', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', zIndex: 100, width: '200px' }}>
          <p style={{ marginBottom: '0.5rem', fontSize: '0.875rem', fontWeight: '500' }}>Enter value for this point:</p>
          <input type="text" value={tempValue} onChange={(e) => setTempValue(e.target.value)} style={{ width: '100%', padding: '0.5rem', border: '1px solid #e5e7eb', borderRadius: '0.375rem', marginBottom: '0.75rem' }} autoFocus onKeyDown={(e) => { if (e.key === 'Enter') handleDialogConfirm(); else if (e.key === 'Escape') setShowPointDialog(false); }} />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setShowPointDialog(false)} style={{ padding: '0.5rem 1rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.375rem', fontSize: '0.875rem', cursor: 'pointer' }}>Cancel</button>
            <button onClick={handleDialogConfirm} style={{ padding: '0.5rem 1rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.375rem', fontSize: '0.875rem', cursor: 'pointer' }}>Add</button>
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default KNN;
