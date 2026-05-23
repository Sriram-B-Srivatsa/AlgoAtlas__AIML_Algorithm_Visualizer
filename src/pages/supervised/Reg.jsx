/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import axios from 'axios';
import InfoButton from '../../components/InfoButton';

function Reg() {
  const navigate = useNavigate();
  const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");

  // Algorithm parameters
  const [alpha, setAlpha] = useState(0.01);
  const [iterations, setIterations] = useState(200);
  const [degree, setDegree] = useState(1);

  // Interaction & Prediction States
  const [pointsMode, setPointsMode] = useState('train');
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);

  // Canvas ref and state for interactive plotting
  const canvasRef = useRef(null);
  const playbackCanvasRef = useRef(null); // Dedicated canvas for the 2x2 grid animation
  const canvasWidth = 600;
  const canvasHeight = 600;

  // Scale for visualization
  const scale = useMemo(() => ({
    x: { min: -8, max: 8 },
    y: { min: -8, max: 8 }
  }), []);

  // Sample data modal
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleCount, setSampleCount] = useState(30);
  const [sampleNoise, setSampleNoise] = useState(2.0);
  const [sampleType, setSampleType] = useState('linear');

  // Playback State
  const [playback, setPlayback] = useState({
      active: false,
      frames: [],
      currentIndex: 0,
      isPlaying: false
  });

  const [playbackSpeed, setPlaybackSpeed] = useState(100);

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

  // Playback Loop Effect for 2x2 Grid Animation
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
          }, playbackSpeed);
      }
      return () => clearInterval(interval);
  }, [playback.active, playback.isPlaying, playbackSpeed]);

  const togglePlayback = () => {
      setPlayback(p => {
          if (!p.isPlaying && p.currentIndex >= p.frames.length - 1) {
              return { ...p, currentIndex: 0, isPlaying: true };
          }
          return { ...p, isPlaying: !p.isPlaying };
      });
  };

  const getValidPoints = () => {
    return dataPairs
      .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
      .map(pair => ({ x: parseFloat(pair.x), y: parseFloat(pair.y) }));
  };

  // Draw grid function
  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;

    const stepX = width / 16;
    const stepY = height / 16;

    for (let i = 0; i <= 16; i++) {
      const y = i * stepY;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    for (let i = 0; i <= 16; i++) {
      const x = i * stepX;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    const yAxisPos = height / 2;
    ctx.beginPath();
    ctx.moveTo(0, yAxisPos);
    ctx.lineTo(width, yAxisPos);
    ctx.stroke();

    const xAxisPos = width / 2;
    ctx.beginPath();
    ctx.moveTo(xAxisPos, 0);
    ctx.lineTo(xAxisPos, height);
    ctx.stroke();

    ctx.setLineDash([]);

    ctx.fillStyle = '#4b5563';
    ctx.font = '12px Inter, sans-serif';

    for (let i = 0; i <= 16; i += 2) {
      const x = i * stepX;
      const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min);
      ctx.fillText(value.toFixed(0), x - 8, height - 5);
    }

    for (let i = 0; i <= 16; i += 2) {
      const y = i * stepY;
      const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min);
      ctx.fillText(value.toFixed(0), 5, y + 4);
    }
  };

  // Helper function to draw points
  const drawDataPoints = (ctx, points, width, height) => {
    if (!points || points.length === 0) return;

    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * height;

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  };

  // Draw regression line function
  const drawRegressionLine = (ctx, coefficients, intercept, width, height) => {
    if (!coefficients || intercept === undefined) return;

    const numPoints = 100;
    const xMin = scale.x.min;
    const xMax = scale.x.max;
    const xStep = (xMax - xMin) / (numPoints - 1);

    const points = [];
    for (let i = 0; i < numPoints; i++) {
      const x = xMin + i * xStep;

      let y = intercept;
      for (let j = 0; j < coefficients.length; j++) {
        y += coefficients[j] * Math.pow(x, j + 1);
      }

      const screenX = ((x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
      const screenY = height - ((y - scale.y.min) / (scale.y.max - scale.y.min)) * height;

      points.push({ x: screenX, y: screenY });
    }

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }

    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.lineWidth = 3;
    ctx.stroke();
  };

  // Main input canvas effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f9f9f9";
    ctx.fillRect(0, 0, width, height);

    drawGrid(ctx, width, height);

    const points = getValidPoints();
    drawDataPoints(ctx, points, width, height);

    // Draw Unpredicted Points
    if (predictions.length === 0 && pointsMode === 'predict') {
      predictPoints.forEach(point => {
        const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
        const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * height;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray
        ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
      });
    }

    // Draw Predicted Points
    predictions.forEach(pred => {
      const x = ((pred.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
      const y = ((scale.y.max - pred.predictedValue) / (scale.y.max - scale.y.min)) * height;
      const originalY = ((scale.y.max - pred.y) / (scale.y.max - scale.y.min)) * height;

      // Dotted line connecting original to prediction
      ctx.beginPath(); ctx.moveTo(x, originalY); ctx.lineTo(x, y);
      ctx.strokeStyle = 'rgba(107, 114, 128, 0.5)'; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);

      // Predicted dot
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(139, 92, 246, 0.8)'; // Purple prediction
      ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5; ctx.setLineDash([2, 2]); ctx.stroke(); ctx.setLineDash([]);

      ctx.fillStyle = '#000'; ctx.font = 'bold 11px Arial';
      ctx.fillText(pred.predictedValue.toFixed(2), x + 10, y - 10);
    });

    if (points.length === 0 && predictPoints.length === 0) {
      ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
      ctx.font = '16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click to add data points', width / 2, height / 2);
    }
  }, [dataPairs, predictPoints, predictions, pointsMode]);

  // Playback canvas effect for 2x2 Grid Animation
  useEffect(() => {
    const canvas = playbackCanvasRef.current;
    if (!canvas || !playback.active || playback.frames.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    drawGrid(ctx, width, height);
    drawDataPoints(ctx, getValidPoints(), width, height);

    const frame = playback.frames[playback.currentIndex];
    if (frame && frame.coefficients) {
        drawRegressionLine(ctx, frame.coefficients, frame.intercept, width, height);
    }
  }, [playback.currentIndex, playback.active, dataPairs]);

  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const adjustedX = x * scaleX;
    const adjustedY = y * scaleY;

    const dataX = scale.x.min + (adjustedX / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - (adjustedY / canvas.height) * (scale.y.max - scale.y.min);

    return { x: dataX, y: dataY };
  };

  const handleCanvasClick = (e) => {
    if (loading) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const dataPoint = screenToData(x, y);

    if (pointsMode === 'train') {
        const emptyPairIndex = dataPairs.findIndex(pair => pair.x === '' && pair.y === '');
        const newPoint = {
          x: dataPoint.x.toFixed(2),
          y: dataPoint.y.toFixed(2)
        };

        if (emptyPairIndex >= 0) {
          const newPairs = [...dataPairs];
          newPairs[emptyPairIndex] = newPoint;
          setDataPairs(newPairs);
        } else {
          setDataPairs([...dataPairs, newPoint]);
        }
        setResults(null);
        setPlayback({ active: false, frames: [], currentIndex: 0, isPlaying: false });
    } else {
        setPredictPoints([...predictPoints, { x: dataPoint.x, y: dataPoint.y }]);
    }
  };

  const handlePredict = () => {
    if (predictPoints.length === 0) { setError("Add points to predict first!"); return; }
    if (!results || results.intercept === undefined) { setError("Train the model first!"); return; }

    setLoading(true); setError(null);

    try {
      const newPredictions = predictPoints.map(p => {
        let y_pred = results.intercept;
        for (let j = 0; j < results.coefficients.length; j++) {
          y_pred += results.coefficients[j] * Math.pow(p.x, j + 1);
        }
        return { ...p, predictedValue: y_pred, isPredicted: true };
      });
      setPredictions(newPredictions);
    } catch(err) {
      setError("Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const generateSampleData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.post(`${apiUrl}/regression/sample_data`, {
        dataset_type: sampleType,
        n_samples: sampleCount,
        noise_level: sampleNoise / 10
      });

      if (response.data.X && response.data.y) {
        const pairs = response.data.X.map((x, index) => ({
          x: x.toString(),
          y: response.data.y[index].toString()
        }));

        setDataPairs(pairs);
        setPredictPoints([]);
        setPredictions([]);
        setResults(null);
        setPlayback({ active: false, frames: [], currentIndex: 0, isPlaying: false });
      } else {
        setError("Failed to generate sample data: No data points received");
      }

      setShowSampleDataModal(false);
    } catch (err) {
      console.error("Error generating sample data:", err);
      setError(`Failed to generate sample data: ${err.message || "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  const resetData = () => {
    setDataPairs([{ x: '', y: '' }]);
    setPredictPoints([]);
    setPredictions([]);
    setResults(null);
    setError(null);
    setPlayback({ active: false, frames: [], currentIndex: 0, isPlaying: false });
  };

  const handleRunModel = async () => {
    const validPairs = getValidPoints();
    if (validPairs.length < 2) { setError('Please add at least 2 data points for meaningful regression'); return; }

    setError(null); setLoading(true); setPlayback({ active: false, frames: [], currentIndex: 0, isPlaying: false });

    try {
      const apiData = {
        X: validPairs.map(pair => pair.x),
        y: validPairs.map(pair => pair.y),
        alpha: alpha,
        iterations: iterations,
        degree: degree
      };

      const response = await axios.post(`${apiUrl}/regression/train`, apiData);

      if (response.data.error) {
        throw new Error(response.data.error);
      }

      setResults(response.data);

      if (response.data.iteration_history && response.data.iteration_history.length > 0) {
        setPlayback({
            active: true,
            frames: response.data.iteration_history,
            currentIndex: 0,
            isPlaying: true
        });
      }
    } catch (err) {
      console.error('Error details:', err);
      setResults(null);

      const errorMessage = err.message || 'An error occurred while running the model.';

      if (
        errorMessage.toLowerCase().includes('diverg') ||
        errorMessage.toLowerCase().includes('explod') ||
        errorMessage.toLowerCase().includes('inf') ||
        errorMessage.toLowerCase().includes('nan') ||
        errorMessage.toLowerCase().includes('overflow')
      ) {
        setError(
          `Error: Gradient descent failed to converge. This is likely due to a learning rate (${alpha.toFixed(4)}) that is too high. ` +
          `Try reducing the learning rate or using a lower polynomial degree.`
        );
      } else {
        setError(`Error: ${errorMessage} Please try again.`);
      }
    } finally {
      setLoading(false);
    }
  };

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', maxWidth: '500px', width: '90%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', margin: 0 }}>Generate Sample Data</h2>
          <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Dataset Type</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            <button onClick={() => setSampleType('linear')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleType === 'linear' ? '#3b82f6' : '#e5e7eb', color: sampleType === 'linear' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Linear</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Straight line</span>
            </button>
            <button onClick={() => setSampleType('quadratic')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleType === 'quadratic' ? '#3b82f6' : '#e5e7eb', color: sampleType === 'quadratic' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Quadratic</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Parabolic curve</span>
            </button>
            <button onClick={() => setSampleType('sinusoidal')} style={{ padding: '0.5rem 0.75rem', backgroundColor: sampleType === 'sinusoidal' ? '#3b82f6' : '#e5e7eb', color: sampleType === 'sinusoidal' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontWeight: '500' }}>Sinusoidal</span><span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Wave pattern</span>
            </button>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="samples-slider" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
          <input id="samples-slider" type="range" min="10" max="100" step="5" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}>
            <span>10 (Fewer points)</span><span>100 (More points)</span>
          </div>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="noise-slider" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Noise Level: {sampleNoise.toFixed(1)}</label>
          <input id="noise-slider" type="range" min="1.0" max="10.0" step="0.5" value={sampleNoise} onChange={(e) => setSampleNoise(Number(e.target.value))} style={{ width: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}>
            <span>1.0 (Clean data)</span><span>10.0 (Noisy data)</span>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleData} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Generate</button>
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
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
        </button>
        <h1 className="model-title">Polynomial Regression</h1>
      </div>

      <p className="model-description">
        Polynomial Regression extends linear regression to fit an nth degree polynomial equation to data, modeling non-linear relationships.
        <span style={{ display: 'inline-flex', verticalAlign: 'middle', marginLeft: '8px' }}>
            <InfoButton algoId="reg" />
        </span>
      </p>

      {backendStatus === "disconnected" && (
        <div className="backend-status error">
          <span>Backend service is not responding. Please make sure the Flask server is running on port 5000.</span>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      <div className="content-container" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

        {/* TOP ROW: Input & Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

          <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
            <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 className="section-title">Data Points</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Load Sample Data</button>
                <button className="reset-button" onClick={resetData} disabled={loading} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Reset Data</button>
              </div>
            </div>

            <p style={{ fontSize: '0.85rem', color: '#6b7280', fontWeight: '500', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              Click on the grid below to plot your data points.
            </p>

            <div style={{ marginBottom: '1rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', width: '100%', height: 0, paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas ref={canvasRef} width={canvasWidth} height={canvasHeight} onClick={handleCanvasClick} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              </div>
            </div>
          </div>

          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title" style={{ marginBottom: '1rem' }}>Algorithm Controls</h2>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Parameters</h3>

              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Learning Rate (Alpha): {alpha.toFixed(4)}</label>
                <input type="range" min="0.0001" max="0.1" step="0.0001" value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>0.0001 (Slow)</span><span>0.1 (Fast)</span></div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Max Iterations: {iterations}</label>
                <input type="range" min="10" max="2000" step="10" value={iterations} onChange={(e) => setIterations(parseInt(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>10</span><span>2000</span></div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Polynomial Degree: {degree}</label>
                <input type="range" min="1" max="10" step="1" value={degree} onChange={(e) => setDegree(parseInt(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>1 (Linear)</span><span>10 (Complex)</span></div>
              </div>

              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Interaction Mode</h3>
              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <button onClick={() => setPointsMode('train')} style={{ padding: '0.75rem 1rem', backgroundColor: pointsMode === 'train' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'train' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Training Points</button>
                <button onClick={() => setPointsMode('predict')} style={{ padding: '0.75rem 1rem', backgroundColor: pointsMode === 'predict' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'predict' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', flex: 1, fontSize: '0.95rem' }}>Prediction Points</button>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <button onClick={handleRunModel} disabled={loading || getValidPoints().length < 2} style={{ width: '100%', padding: '0.9rem', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', fontSize: '1.05rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}>
                  {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Running...</>
                  ) : 'Run Polynomial Regression'}
                </button>
                <button onClick={handlePredict} disabled={loading || predictPoints.length < 1 || !results} style={{ width: '100%', padding: '0.9rem', backgroundColor: loading ? '#c4b5fd' : '#8b5cf6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', fontSize: '1.05rem', opacity: (predictPoints.length < 1 || !results) ? 0.7 : 1 }}>
                  Predict Points
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>Training Results</h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Regression Curve Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Regression Evolution</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Iteration: {playback.frames.length > 0 ? playback.frames[playback.currentIndex].iteration : results.iterations}
                            </span>
                        </div>
                        <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#ffffff', width: '100%', height: '0', paddingBottom: '100%', marginBottom: '1.5rem' }}>
                            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                                <canvas ref={playbackCanvasRef} width={canvasWidth} height={canvasHeight} style={{ display: 'block', width: '100%', height: '100%' }} />
                            </div>
                        </div>
                        {playback.frames.length > 0 && (
                            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', justifyContent: 'center', maxWidth: '400px', margin: '0 auto' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '1rem' }}>
                                    <span style={{ fontSize: '0.8rem', color: '#6b7280' }}>Speed:</span>
                                    <input type="range" min="10" max="500" step="10" value={510 - playbackSpeed} onChange={(e) => setPlaybackSpeed(510 - parseInt(e.target.value))} style={{ width: '60px' }} />
                                </div>
                                <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '1.2rem' }}>
                                    {playback.isPlaying ? '⏸' : '▶'}
                                </button>
                                <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1, cursor: 'pointer' }} />
                            </div>
                        )}
                    </div>

                    {/* (0,1) Final Stats and Equation */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                R² Score: {(results.r2 * 100).toFixed(2)}%
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>
                                Final MSE: {results.mse?.toFixed(4)} | Iterations: {results.iterations}
                            </p>
                        </div>
                        <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center', flex: 2, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Polynomial Equation</h4>
                            <div style={{ fontWeight: '600', fontSize: '1.1rem', fontFamily: 'math, serif', color: '#3b82f6', padding: '1rem', backgroundColor: '#eff6ff', borderRadius: '8px', wordBreak: 'break-word' }}>
                                {`y = ${results.intercept.toFixed(2)} ${
                                  results.coefficients.map((coef, index) => `${coef >= 0 ? '+' : '-'} ${Math.abs(coef).toFixed(2)}x${index+1 > 1 ? `^${index+1}` : ''}`).join(' ')
                                }`}
                            </div>
                            <p style={{fontSize: '0.85rem', color: '#6b7280', marginTop: '1rem'}}>
                                Degree 1 is a line. Degree 2 is a parabola. Higher degrees map to complex non-linear curves!
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Cost History Curve */}
                    {results.cost_history_data && results.cost_history_data.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Gradient Descent Cost History</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.cost_history_data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="iteration" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line type="monotone" dataKey="cost" stroke="#ef4444" name="Mean Squared Error" strokeWidth={2} dot={false} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Plunging cost indicates the model successfully learned the curvature of the data.
                            </p>
                        </div>
                    )}

                    {/* (1,1) Final Regression Surface (Static Image) */}
                    {results.final_plot && (
                        <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem', width: '100%' }}>Final Regression Surface</h4>
                            <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
                                <img
                                    src={`data:image/png;base64,${results.final_plot}`}
                                    alt="Final Regression Plot"
                                    style={{ width: '100%', maxWidth: '400px', borderRadius: '4px' }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </motion.div>
        )}
      </div>

    </motion.div>
  );
}

export default Reg;
