/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function QLearning() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const[backendStatus, setBackendStatus] = useState("connected");

  // Grid State
  const gridSize = 8;
  const [grid, setGrid] = useState({
      width: gridSize,
      height: gridSize,
      start: { x: 0, y: 0 },
      goal: { x: 7, y: 7 },
      walls: []
  });

  // Brush Mode: 'wall', 'eraser', 'start', 'goal'
  const[brushMode, setBrushMode] = useState('wall');
  const [isMouseDown, setIsMouseDown] = useState(false);

  // Q-Learning Parameters
  const [episodes, setEpisodes] = useState(500);
  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.9);
  const [epsilon, setEpsilon] = useState(0.1);
  const [results, setResults] = useState(null);

  // Playback State
  const [playback, setPlayback] = useState({
      active: false,
      frames:[],
      currentIndex: 0,
      isPlaying: false
  });

  // Modals
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleMapType, setSampleMapType] = useState('simple');

  const canvasRef = useRef(null);
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
        }, 1000);
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

  const drawGridWorld = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f9fafb";
    ctx.fillRect(0, 0, width, height);

    const cellWidth = width / grid.width;
    const cellHeight = height / grid.height;

    // Draw lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let x = 0; x <= grid.width; x++) {
        ctx.beginPath(); ctx.moveTo(x * cellWidth, 0); ctx.lineTo(x * cellWidth, height); ctx.stroke();
    }
    for (let y = 0; y <= grid.height; y++) {
        ctx.beginPath(); ctx.moveTo(0, y * cellHeight); ctx.lineTo(width, y * cellHeight); ctx.stroke();
    }

    // Draw Walls
    ctx.fillStyle = '#4b5563';
    grid.walls.forEach(w => {
        ctx.fillRect(w.x * cellWidth, w.y * cellHeight, cellWidth, cellHeight);
    });

    // Draw Start
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(grid.start.x * cellWidth, grid.start.y * cellHeight, cellWidth, cellHeight);
    ctx.fillStyle = 'white'; ctx.font = 'bold 20px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('S', grid.start.x * cellWidth + cellWidth/2, grid.start.y * cellHeight + cellHeight/2);

    // Draw Goal
    ctx.fillStyle = '#10b981';
    ctx.fillRect(grid.goal.x * cellWidth, grid.goal.y * cellHeight, cellWidth, cellHeight);
    ctx.fillStyle = 'white'; ctx.font = 'bold 20px Arial';
    ctx.fillText('G', grid.goal.x * cellWidth + cellWidth/2, grid.goal.y * cellHeight + cellHeight/2);
  };

  useEffect(() => { drawGridWorld(); }, [grid]);

  const handleCanvasInteraction = (e) => {
    if (loading) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellX = Math.floor((x / rect.width) * grid.width);
    const cellY = Math.floor((y / rect.height) * grid.height);

    if (cellX < 0 || cellX >= grid.width || cellY < 0 || cellY >= grid.height) return;

    setGrid(prev => {
        const newGrid = { ...prev };

        // Check if clicking start or goal
        const isStart = (cellX === newGrid.start.x && cellY === newGrid.start.y);
        const isGoal = (cellX === newGrid.goal.x && cellY === newGrid.goal.y);

        if (brushMode === 'start' && !isGoal) {
            newGrid.start = { x: cellX, y: cellY };
            newGrid.walls = newGrid.walls.filter(w => w.x !== cellX || w.y !== cellY);
        }
        else if (brushMode === 'goal' && !isStart) {
            newGrid.goal = { x: cellX, y: cellY };
            newGrid.walls = newGrid.walls.filter(w => w.x !== cellX || w.y !== cellY);
        }
        else if (brushMode === 'wall' && !isStart && !isGoal) {
            if (!newGrid.walls.some(w => w.x === cellX && w.y === cellY)) {
                newGrid.walls.push({ x: cellX, y: cellY });
            }
        }
        else if (brushMode === 'eraser') {
            newGrid.walls = newGrid.walls.filter(w => w.x !== cellX || w.y !== cellY);
        }

        return newGrid;
    });
    setResults(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const resetData = () => {
    setGrid({ width: gridSize, height: gridSize, start: { x: 0, y: 0 }, goal: { x: 7, y: 7 }, walls:[] });
    setResults(null); setError(null);
    setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
  };

  const generateSampleDataWithOptions = () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);

    axios.post(`${apiUrl}/rl/sample_data`, { map_type: sampleMapType })
    .then(response => {
      if (response.data && response.data.walls !== undefined) {
        setGrid({
            width: response.data.width, height: response.data.height,
            start: response.data.start, goal: response.data.goal,
            walls: response.data.walls
        });
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
      } else if (response.data.error) {
        setError(`Backend Error: ${response.data.error}`);
      }
    })
    .catch(err => setError(`Network Error: ${err.message}`))
    .finally(() => setLoading(false));
  };

  const trainModel = () => {
    setLoading(true); setError(null);

    const requestData = {
      grid: grid,
      parameters: { episodes: episodes, alpha: alpha, gamma: gamma, epsilon: epsilon }
    };

    axios.post(`${apiUrl}/rl/train`, requestData)
      .then(response => {
        if (response.data.error) { setError(response.data.error); return; }

        setResults(response.data);
        if (response.data.history && response.data.history.length > 0) {
            setPlayback({
                active: true,
                frames: response.data.history,
                currentIndex: 0,
                isPlaying: true
            });
        }
      })
      .catch(err => setError("Failed to train model."))
      .finally(() => setLoading(false));
  };

  const SampleDataModal = () => (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', width: '90%', maxWidth: '550px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', margin: 0 }}>Generate Maze Layout</h2>
          <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
        </div>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Map Layout</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            <button onClick={() => setSampleMapType('simple')} style={{ padding: '0.5rem', backgroundColor: sampleMapType === 'simple' ? '#3b82f6' : '#e5e7eb', color: sampleMapType === 'simple' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Simple Block</button>
            <button onClick={() => setSampleMapType('corridor')} style={{ padding: '0.5rem', backgroundColor: sampleMapType === 'corridor' ? '#3b82f6' : '#e5e7eb', color: sampleMapType === 'corridor' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Corridor</button>
            <button onClick={() => setSampleMapType('maze')} style={{ padding: '0.5rem', backgroundColor: sampleMapType === 'maze' ? '#3b82f6' : '#e5e7eb', color: sampleMapType === 'maze' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' }}>Hard Maze</button>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem' }}>
          <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', color: '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Cancel</button>
          <button onClick={generateSampleDataWithOptions} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500' }}>Load Map</button>
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
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Hub</span>
        </button>
        <h1 className="model-title">Q-Learning (QL)</h1>
      </div>

      <p className="model-description">
        Q-Learning is a model-free reinforcement learning algorithm. Instead of analyzing a dataset, an AI Agent learns the best policy by wandering around an environment, hitting walls (penalties), and eventually finding the goal (rewards).
        <InfoButton algoId="ql" />
      </p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

          <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Grid-World Maze Builder</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} disabled={loading} style={{ backgroundColor: '#3b82f6', color: 'white'}}>Load Maps</button>
                <button className="sample-data-button" onClick={resetData} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }} disabled={loading}>Clear Board</button>
              </div>
            </div>

            <p style={{ color: '#4b5563', marginBottom: '1rem', lineHeight: '1.5' }}>
              Select a brush and click/drag on the grid to build a custom environment for the AI Agent.
            </p>

            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <button onClick={() => setBrushMode('wall')} style={{ padding: '0.5rem 1rem', backgroundColor: brushMode === 'wall' ? '#4b5563' : '#f3f4f6', color: brushMode === 'wall' ? 'white' : 'black', border: '1px solid #d1d5db', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Wall</button>
                <button onClick={() => setBrushMode('start')} style={{ padding: '0.5rem 1rem', backgroundColor: brushMode === 'start' ? '#3b82f6' : '#f3f4f6', color: brushMode === 'start' ? 'white' : 'black', border: '1px solid #d1d5db', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Start (S)</button>
                <button onClick={() => setBrushMode('goal')} style={{ padding: '0.5rem 1rem', backgroundColor: brushMode === 'goal' ? '#10b981' : '#f3f4f6', color: brushMode === 'goal' ? 'white' : 'black', border: '1px solid #d1d5db', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Goal (G)</button>
                <button onClick={() => setBrushMode('eraser')} style={{ padding: '0.5rem 1rem', backgroundColor: brushMode === 'eraser' ? '#fca5a5' : '#f3f4f6', color: brushMode === 'eraser' ? 'white' : 'black', border: '1px solid #d1d5db', borderRadius: '4px', cursor: 'pointer', flex: 1 }}>Eraser</button>
            </div>

            {/* NEW: Reward System Rules Box */}
            <div style={{ backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', padding: '1rem', borderRadius: '0.5rem', marginBottom: '1rem' }}>
                <h4 style={{ margin: '0 0 0.75rem 0', color: '#166534', fontSize: '0.95rem' }}>Environment Rules & Reward System</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', fontSize: '0.85rem', color: '#15803d' }}>
                    <div style={{ backgroundColor: 'white', padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #bbf7d0', textAlign: 'center' }}><strong>🚶 Valid Step:</strong> -1 pt</div>
                    <div style={{ backgroundColor: 'white', padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #bbf7d0', textAlign: 'center' }}><strong>🧱 Hit Wall:</strong> -5 pts</div>
                    <div style={{ backgroundColor: 'white', padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #bbf7d0', textAlign: 'center' }}><strong>🏆 Find Goal:</strong> +100 pts</div>
                </div>
                <p style={{ margin: '0.75rem 0 0 0', fontSize: '0.8rem', color: '#166534', fontStyle: 'italic', lineHeight: '1.4' }}>
                    The AI's objective is to maximize its total score. Because every single step drains points (-1), the AI is mathematically forced to find the absolute shortest, fastest path to the goal without hitting walls!
                </p>
            </div>

            <div style={{ marginBottom: '1rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', width: '100%', height: '0', paddingBottom: '100%' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas
                    ref={canvasRef} width={600} height={600}
                    onMouseDown={(e) => { setIsMouseDown(true); handleCanvasInteraction(e); }}
                    onMouseUp={() => setIsMouseDown(false)}
                    onMouseLeave={() => setIsMouseDown(false)}
                    onMouseMove={(e) => { if (isMouseDown) handleCanvasInteraction(e); }}
                    style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }}
                />
              </div>
            </div>
          </div>

          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Controls & Results</h2>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Q-Learning Hyperparameters</h3>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Total Episodes: {episodes}</label>
                <input type="range" min="10" max="1000" step="10" value={episodes} onChange={(e) => setEpisodes(parseInt(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How many times the agent attempts the maze.</p>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Learning Rate (Alpha): {alpha}</label>
                <input type="range" min="0.01" max="1.0" step="0.01" value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How quickly the agent overrides old information with new info.</p>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Discount Factor (Gamma): {gamma}</label>
                <input type="range" min="0.1" max="0.99" step="0.01" value={gamma} onChange={(e) => setGamma(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How much the agent cares about long-term rewards vs short-term.</p>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Exploration Rate (Epsilon): {epsilon}</label>
                <input type="range" min="0.01" max="1.0" step="0.01" value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How often the agent chooses a random move vs following the policy.</p>
              </div>
            </div>

            <div style={{ marginBottom: '1rem', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <button onClick={trainModel} disabled={loading || backendStatus === "disconnected"} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1 }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Agent is Learning...</>
                ) : 'Train RL Agent'}
              </button>
            </div>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
                <div style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>
                    <p style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>
                        <h2 style={{display:'inline'}}>Note: </h2>Agent can only move in 4 directions: Up, Down, Left, Right. In RL, we try to keep the "Action Space" as small as possible so the AI learns faster. If we added 4 diagonal moves, the AI would have 8 choices per square. Its Q-Table (brain) would double in size, and it would take twice as many episodes to train
                    </p>
                </div>
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Agent Animation */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Agent Behavior Evolution</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Episode: {playback.frames.length > 0 ? playback.frames[playback.currentIndex].episode : results.episodes}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                                src={`data:image/png;base64,${playback.frames.length > 0 ? playback.frames[playback.currentIndex].image : results.history[results.history.length-1].image}`}
                                alt="Agent Path"
                                style={{ maxHeight: '350px', maxWidth: '100%', borderRadius: '8px', border: '1px solid #e5e7eb' }}
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
                                Win Rate: {(results.success_rate * 100).toFixed(2)}%
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>
                                Final Episode Reward: {results.final_reward}
                            </p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>How the AI Learns</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6', textAlign: 'left' }}>
                                Watch the animation on the left. In Episode 1, the red line is chaotic. The agent has no brain, wanders randomly, hits walls (penalty), and takes hundreds of steps. <br/><br/>
                                By the final Episode, the agent has filled out its Q-Table. The red line snaps instantly from Start to Goal using the absolute most mathematically optimal path!
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Reward Convergence Curve */}
                    {results.reward_history && results.reward_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                                Reward Convergence
                            </h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.reward_history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="episode" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line type="monotone" dataKey="reward" stroke="#10b981" name="Total Reward" strokeWidth={2} dot={false} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Total Reward starts highly negative (hitting walls). As the agent learns, rewards spike upwards and plateau at the maximum possible score.
                            </p>
                        </div>
                    )}

                    {/* (1,1) Final Q-Table Heatmap */}
                    {results.q_table_image && (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem', width: '100%' }}>Learned Policy (Q-Table)</h4>
                            <div style={{ height: 350, width: '100%', display: 'flex', justifyContent: 'center' }}>
                                <img
                                    src={`data:image/png;base64,${results.q_table_image}`}
                                    alt="Q-Table Heatmap"
                                    style={{ maxHeight: '100%', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}
                                />
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                The white arrows show the final "Brain" of the AI. Drop the agent anywhere, and it will just follow the arrows to the goal.
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

export default QLearning;
