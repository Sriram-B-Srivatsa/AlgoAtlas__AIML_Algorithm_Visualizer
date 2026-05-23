/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function DQN() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const[backendStatus, setBackendStatus] = useState("connected");

  // DQN Parameters
  const [episodes, setEpisodes] = useState(150);
  const [learningRate, setLearningRate] = useState(0.01);
  const [gamma, setGamma] = useState(0.95);
  const[epsilonDecay, setEpsilonDecay] = useState(0.95);
  const[results, setResults] = useState(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(30); // Default to 30ms per frame

  // Playback State (for Canvas)
  const[playback, setPlayback] = useState({
      episodes:[],
      currentEpIndex: 0,
      currentFrame: 0,
      isPlaying: false
  });

  const physicsCanvasRef = useRef(null);
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

  // 60FPS Physics Render Loop
  useEffect(() => {
    let interval;
    if (playback.isPlaying && playback.episodes.length > 0) {
        interval = setInterval(() => {
            setPlayback(prev => {
                const epData = prev.episodes[prev.currentEpIndex];
                if (!epData || prev.currentFrame >= epData.states.length - 1) {
                    clearInterval(interval);
                    return { ...prev, isPlaying: false };
                }
                return { ...prev, currentFrame: prev.currentFrame + 1 };
            });
        }, playbackSpeed); // <--- NOW USES THE SLIDER SPEED
    }
    return () => clearInterval(interval);
  },[playback.isPlaying, playback.currentEpIndex, playback.episodes, playbackSpeed]); // <--- ADDED DEPENDENCY

  const togglePlayback = () => {
      setPlayback(p => {
          const epData = p.episodes[p.currentEpIndex];
          if (!p.isPlaying && p.currentFrame >= epData.states.length - 1) {
              // Restart from 0 if at the end
              return { ...p, currentFrame: 0, isPlaying: true };
          }
          return { ...p, isPlaying: !p.isPlaying };
      });
  };

  const selectEpisode = (index) => {
      setPlayback(p => ({
          ...p,
          currentEpIndex: index,
          currentFrame: 0,
          isPlaying: true
      }));
  };

  // Draw the CartPole on the Canvas using raw physics data
  useEffect(() => {
    if (!physicsCanvasRef.current || playback.episodes.length === 0) return;
    const ctx = physicsCanvasRef.current.getContext('2d');
    const { width, height } = physicsCanvasRef.current;

    // Clear background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, width, height);

    const epData = playback.episodes[playback.currentEpIndex];
    if (!epData) return;

    const state = epData.states[playback.currentFrame];
    if (!state) return;

    // The 4 numbers from the backend: x, x_velocity, theta, theta_velocity
    const[x, , theta] = state;

    // Mapping Physics Coordinates to Canvas Pixels
    // Cart limits are -2.4 to 2.4. We map this to 0 - width.
    const scaleX = width / 4.8;
    const cartX = (x + 2.4) * scaleX;
    const cartY = height / 1.5;

    // 1. Draw Track
    ctx.beginPath();
    ctx.moveTo(0, cartY);
    ctx.lineTo(width, cartY);
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 2. Draw Cart
    const cartW = 60;
    const cartH = 30;
    ctx.fillStyle = epData.success ? '#10b981' : '#3b82f6'; // Green if it wins, Blue if it loses
    ctx.fillRect(cartX - cartW/2, cartY - cartH/2, cartW, cartH);

    // 3. Draw Pole
    const poleLen = 120;
    // Math: pole is mounted on top of cart. Theta is angle from vertical.
    const poleX = cartX + poleLen * Math.sin(theta);
    const poleY = cartY - poleLen * Math.cos(theta); // Subtract because Canvas Y goes down

    ctx.beginPath();
    ctx.moveTo(cartX, cartY);
    ctx.lineTo(poleX, poleY);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.stroke();

    // 4. Draw Axle
    ctx.beginPath();
    ctx.arc(cartX, cartY, 6, 0, Math.PI*2);
    ctx.fillStyle = 'black';
    ctx.fill();

    // 5. Draw Text Overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Episode: ${epData.episode}`, 10, 25);
    ctx.fillText(`Step: ${playback.currentFrame} / ${epData.states.length - 1}`, 10, 45);

    if (epData.success && playback.currentFrame >= epData.states.length - 1) {
        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 24px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('BALANCED!', width/2, 40);
    } else if (!epData.success && playback.currentFrame >= epData.states.length - 1) {
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 24px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('CRASHED!', width/2, 40);
    }

  },[playback.currentEpIndex, playback.currentFrame, playback.episodes]);

  const trainModel = () => {
    setLoading(true); setError(null);
    setPlayback({ active: false, episodes:[], currentEpIndex: 0, currentFrame: 0, isPlaying: false });

    const requestData = {
      parameters: {
        episodes: episodes,
        learningRate: learningRate,
        gamma: gamma,
        epsilonDecay: epsilonDecay
      }
    };

    axios.post(`${apiUrl}/dqn/train`, requestData)
      .then(response => {
        if (response.data.error) { setError(response.data.error); return; }

        setResults(response.data);
        if (response.data.animation_episodes && response.data.animation_episodes.length > 0) {
            setPlayback({
                active: true,
                episodes: response.data.animation_episodes,
                currentEpIndex: 0,
                currentFrame: 0,
                isPlaying: true
            });
        }
      })
      .catch(err => setError("Failed to train model. Make sure backend is running."))
      .finally(() => setLoading(false));
  };

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
          <span style={{ marginLeft: '0.5rem' }}>Back to Hub</span>
        </button>
        <h1 className="model-title">Deep Q-Networks (DQN)</h1>
      </div>

      <p className="model-description">
        Deep Q-Networks combine Deep Learning with Reinforcement Learning. Instead of a spreadsheet (Q-Table), the AI uses a Neural Network to predict the best physical action to take in real-time. We will use the famous <strong>CartPole</strong> physics environment to test it!
        <InfoButton algoId="dqn" />
      </p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

          <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">The CartPole Environment</h2>
            </div>

            <div style={{ backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', padding: '1rem', borderRadius: '0.5rem', marginBottom: '1.5rem' }}>
                <h4 style={{ margin: '0 0 0.75rem 0', color: '#166534', fontSize: '0.95rem' }}>Physics Rules & Reward System</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem', fontSize: '0.85rem', color: '#15803d' }}>
                    <div style={{ backgroundColor: 'white', padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #bbf7d0', textAlign: 'center' }}><strong>⬅️➡️ Action Space:</strong> Push Left or Right</div>
                    <div style={{ backgroundColor: 'white', padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #bbf7d0', textAlign: 'center' }}><strong>🎯 The Goal:</strong> Keep pole balanced perfectly.</div>
                    <div style={{ backgroundColor: 'white', padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #bbf7d0', textAlign: 'center' }}><strong>🏆 Reward:</strong> +1 pt for every step balanced.</div>
                    <div style={{ backgroundColor: 'white', padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #bbf7d0', textAlign: 'center' }}><strong>💀 Failure:</strong> Pole falls over 12 degrees and gets 0 points.</div>
                </div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>How DQN Solves This</h3>
              <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                <p style={{ marginBottom: '0.5rem' }}>The Neural Network looks at 4 raw physics variables in real time:</p>
                <ol style={{ paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                  <li>Cart Position</li>
                  <li>Cart Velocity</li>
                  <li>Pole Angle</li>
                  <li>Pole Angular Velocity</li>
                </ol>
                <p style={{ fontSize: '0.9rem', fontStyle: 'italic', marginTop: '1rem' }}>It passes these 4 numbers into a Hidden Layer, and the Output Layer spits out 2 Q-Values: (Score if I push Left) and (Score if I push Right).</p>
              </div>
            </div>
          </div>

          <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Controls & Results</h2>

            <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Deep RL Hyperparameters</h3>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Total Episodes: {episodes}</label>
                <input type="range" min="10" max="250" step="10" value={episodes} onChange={(e) => setEpisodes(parseInt(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How many times the AI attempts to balance the pole.</p>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Neural Net Learning Rate: {learningRate}</label>
                <input type="range" min="0.001" max="0.1" step="0.001" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How drastically the Neural Network updates its weights on error.</p>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Discount Factor (Gamma): {gamma}</label>
                <input type="range" min="0.5" max="0.99" step="0.01" value={gamma} onChange={(e) => setGamma(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How much the agent cares about long-term rewards vs short-term.</p>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.75rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Epsilon Decay Rate: {epsilonDecay}</label>
                <input type="range" min="0.8" max="0.99" step="0.01" value={epsilonDecay} onChange={(e) => setEpsilonDecay(parseFloat(e.target.value))} style={{ width: '100%' }} />
                <p style={{fontSize: '0.8rem', color: '#6b7280', marginTop: '4px'}}>How fast the AI stops experimenting and starts using its Neural Net.</p>
              </div>
            </div>

            <div style={{ marginBottom: '1rem', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
              <button onClick={trainModel} disabled={loading || backendStatus === "disconnected"} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1 }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Training Neural Network...</>
                ) : 'Train DQN Agent'}
              </button>
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) REAL-TIME PHYSICS CANVAS */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>High-FPS Physics Engine</h4>
                        </div>

                        {/* THE CANVAS */}
                        <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', backgroundColor: '#ffffff', width: '100%', marginBottom: '1.5rem' }}>
                            <canvas ref={physicsCanvasRef} width={600} height={350} style={{ display: 'block', width: '100%', height: 'auto' }} />
                        </div>

                        {playback.episodes.length > 0 && (
                            <div>
                                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', justifyContent: 'center' }}>
                                    {playback.episodes.map((ep, idx) => (
                                        <button key={idx} onClick={() => selectEpisode(idx)} style={{ padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', border: '1px solid #3b82f6', backgroundColor: playback.currentEpIndex === idx ? '#3b82f6' : 'white', color: playback.currentEpIndex === idx ? 'white' : '#3b82f6', fontWeight: '500' }}>
                                            Episode {ep.episode}
                                        </button>
                                    ))}
                                </div>
                                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', justifyContent: 'center', maxWidth: '400px', margin: '0 auto' }}>
                                    {/* NEW SPEED SLIDER */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '0.5rem' }}>
                                        <span style={{ fontSize: '0.8rem', color: '#6b7280' }}>Speed:</span>
                                        <input type="range" min="10" max="150" step="10" value={160 - playbackSpeed} onChange={(e) => setPlaybackSpeed(160 - parseInt(e.target.value))} style={{ width: '60px' }} />
                                    </div>
                                    <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '1.2rem' }}>
                                        {playback.isPlaying ? '⏸' : '▶'}
                                    </button>
                                    <input type="range" min="0" max={playback.episodes[playback.currentEpIndex].states.length - 1} value={playback.currentFrame} onChange={(e) => setPlayback(p => ({ ...p, currentFrame: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1, cursor: 'pointer' }} />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* (0,1) Final Stats */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                Max Balancing Score: {results.max_score} Steps
                            </p>
                            <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>
                                Final Epsilon (Randomness): {(results.final_epsilon * 100).toFixed(1)}%
                            </p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', textAlign: 'center' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>How to read the Animation</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6', textAlign: 'left' }}>
                                Click the buttons under the video player. <br/><br/>
                                In <strong>Episode 1</strong>, the AI drops the pole immediately because its Neural Network has random weights. <br/><br/>
                                By the <strong>Final Episode</strong>, the Neural Network has successfully mapped the 4 physics variables to the exact right pushes needed to keep it balanced!
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Reward & Loss Curve */}
                    {results.reward_loss_history && results.reward_loss_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                                NN Loss vs Total Reward
                            </h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.reward_loss_history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="episode" tick={{fontSize: 12}} />
                                        <YAxis yAxisId="left" tick={{fontSize: 12}} />
                                        <YAxis yAxisId="right" orientation="right" tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Line yAxisId="left" type="monotone" dataKey="reward" stroke="#10b981" name="Balance Score" strokeWidth={2} dot={false} activeDot={{r: 6}} />
                                        <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#ef4444" name="Neural Net Error" strokeWidth={2} dot={false} activeDot={{r: 6}} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                As the Neural Net Error (Loss) drops, the agent's ability to balance the pole (Reward) skyrockets.
                            </p>
                        </div>
                    )}

                    {/* (1,1) Epsilon Decay Curve */}
                    {results.reward_loss_history && results.reward_loss_history.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Exploration vs Exploitation (Epsilon)</h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                <ResponsiveContainer>
                                    <AreaChart data={results.reward_loss_history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="episode" tick={{fontSize: 12}} />
                                        <YAxis tick={{fontSize: 12}} domain={[0, 1]} />
                                        <Tooltip />
                                        <Legend verticalAlign="top" height={36}/>
                                        <Area type="monotone" dataKey="epsilon" name="Randomness %" stroke="#8b5cf6" fill="#c4b5fd" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                The AI starts by making 100% random moves to explore physics. As it learns, it relies exclusively on its Neural Network (Exploitation).
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

export default DQN;
