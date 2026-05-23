import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

// FIXED API URL
const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function DBScan() {
    const navigate = useNavigate();
    const [dataPoints, setDataPoints] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const[backendStatus, setBackendStatus] = useState("checking");
    const [algorithmStatus, setAlgorithmStatus] = useState("idle");
    const [results, setResults] = useState(null);

    // Parameters
    const[eps, setEps] = useState(0.7);
    const [minPoints, setMinPoints] = useState(5);

    // Visualization options
    const [showCorePoints, setShowCorePoints] = useState(true);
    const [showBorderPoints, setShowBorderPoints] = useState(true);
    const[showNoisePoints, setShowNoisePoints] = useState(true);
    const [showEpsilonRadius, setShowEpsilonRadius] = useState(true);

    // Canvas refs and state
    const canvasRef = useRef(null);
    const playbackCanvasRef = useRef(null); // Dedicated playback canvas for the 2x2 Grid
    const canvasWidth = 600;
    const canvasHeight = 600;
    const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -8, max: 8 } }),[]);

    // Standard 2x2 Grid Playback State
    const [playback, setPlayback] = useState({
        active: false,
        frames:[],
        currentIndex: 0,
        isPlaying: false
    });

    const [playbackSpeed, setPlaybackSpeed] = useState(200);

    const[stats, setStats] = useState({
        processedPoints: 0,
        totalPoints: 0,
        numClusters: 0,
        numCorePoints: 0,
        numBorderPoints: 0,
        numNoisePoints: 0
    });

    // Sample data modal
    const [showSampleDataModal, setShowSampleDataModal] = useState(false);
    const [sampleCount, setSampleCount] = useState(100);
    const [sampleClusters, setSampleClusters] = useState(3);
    const[datasetType, setDatasetType] = useState('blobs');
    const [noiseLevel, setNoiseLevel] = useState(0.05);

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
    },[]);

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

    // Update stats as animation plays
    useEffect(() => {
        if (playback.frames.length > 0 && playback.currentIndex < playback.frames.length) {
            const iteration = playback.frames[playback.currentIndex];
            setStats({
                processedPoints: iteration.num_processed,
                totalPoints: dataPoints.length,
                numClusters: iteration.num_clusters,
                numCorePoints: iteration.core_points.length,
                numBorderPoints: iteration.border_points.length,
                numNoisePoints: iteration.noise_points.length
            });
        }
    }, [playback.currentIndex, playback.frames, dataPoints.length]);

    const handleCanvasClick = (e) => {
        if (algorithmStatus !== "idle") return;

        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        const dataX = scale.x.min + (x / canvas.width) * (scale.x.max - scale.x.min);
        const dataY = scale.y.max - (y / canvas.height) * (scale.y.max - scale.y.min);

        setDataPoints([...dataPoints, { x: dataX, y: dataY }]);
    };

    const loadSampleData = () => setShowSampleDataModal(true);

    const generateSampleData = async () => {
        try {
            setLoading(true); setError(null);
            const response = await axios.post(`${apiUrl}/dbscan/sample_data`, {
                dataset_type: datasetType, n_samples: sampleCount, n_clusters: sampleClusters, noise_level: noiseLevel
            });

            if (response.data.points) setDataPoints(response.data.points);
            else setError("Failed to generate sample data: No data points received");

            setShowSampleDataModal(false);
            setResults(null);
            setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
            setAlgorithmStatus("idle");
        } catch (err) {
            setError(`Failed to generate sample data: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const resetData = () => {
        setDataPoints([]);
        setAlgorithmStatus("idle");
        setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
        setStats({ processedPoints: 0, totalPoints: 0, numClusters: 0, numCorePoints: 0, numBorderPoints: 0, numNoisePoints: 0 });
        setError(null);
    };

    const runDBSCAN = async () => {
        try {
            setLoading(true); setError(null); setAlgorithmStatus("running");
            setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });

            const response = await axios.post(`${apiUrl}/dbscan/run_complete`, {
                points: dataPoints, eps: eps, min_samples: minPoints,
                show_core_points: showCorePoints, show_border_points: showBorderPoints,
                show_noise_points: showNoisePoints, show_epsilon_radius: showEpsilonRadius
            });

            if (response.data.status === "success") {
                setResults(response.data);
                if (response.data.iterations && response.data.iterations.length > 0) {
                    setPlayback({
                        active: true,
                        frames: response.data.iterations,
                        currentIndex: 0,
                        isPlaying: true
                    });
                }
                setAlgorithmStatus("completed");
            } else {
                setError(`Error running DBSCAN: ${response.data.message}`);
                setAlgorithmStatus("idle");
            }
        } catch (err) {
            setError(`Error running DBSCAN: ${err.message}`);
            setAlgorithmStatus("idle");
        } finally {
            setLoading(false);
        }
    };

    // Shared Drawing Methods
    const drawGrid = (ctx, width, height) => {
        ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
        const stepX = width / 16; const stepY = height / 16;
        for (let i = 0; i <= 16; i++) { const y = i * stepY; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
        for (let i = 0; i <= 16; i++) { const x = i * stepX; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
        ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        const yAxisPos = height / 2; ctx.beginPath(); ctx.moveTo(0, yAxisPos); ctx.lineTo(width, yAxisPos); ctx.stroke();
        const xAxisPos = width / 2; ctx.beginPath(); ctx.moveTo(xAxisPos, 0); ctx.lineTo(xAxisPos, height); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
        for (let i = 0; i <= 16; i += 2) { const x = i * stepX; const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min); ctx.fillText(value.toFixed(0), x - 8, height - 5); }
        for (let i = 0; i <= 16; i += 2) { const y = i * stepY; const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min); ctx.fillText(value.toFixed(0), 5, y + 4); }
    };

    // Draw Input Canvas (Static Points)
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, width, height);
        drawGrid(ctx, width, height);

        dataPoints.forEach(point => {
            const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
            const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * height;
            ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray input points
            ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
        });

        if (dataPoints.length === 0) {
            ctx.fillStyle = 'rgba(100, 100, 100, 0.5)'; ctx.font = '16px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('Click to add data points', width / 2, height / 2);
        }
    }, [dataPoints, scale]);

    // Draw Playback Canvas for 2x2 Grid (Animation)
    useEffect(() => {
        const canvas = playbackCanvasRef.current;
        if (!canvas || !playback.active || playback.frames.length === 0) return;

        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, width, height);
        drawGrid(ctx, width, height);

        const iteration = playback.frames[playback.currentIndex];
        const labels = iteration.labels;

        if (showEpsilonRadius && iteration.current_point !== null) {
            const currentPoint = dataPoints[iteration.current_point];
            const x = ((currentPoint.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
            const y = ((scale.y.max - currentPoint.y) / (scale.y.max - scale.y.min)) * height;
            const epsPx = (eps / (scale.x.max - scale.x.min)) * width;
            ctx.beginPath(); ctx.setLineDash([4, 4]); ctx.arc(x, y, epsPx, 0, Math.PI * 2);
            ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
        }

        const clusterColors =[
            'rgba(59, 130, 246, 0.7)', 'rgba(239, 68, 68, 0.7)', 'rgba(34, 197, 94, 0.7)',
            'rgba(168, 85, 247, 0.7)', 'rgba(251, 146, 60, 0.7)'
        ];

        dataPoints.forEach((point, idx) => {
            const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
            const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * height;

            const label = labels[idx];
            const isCurrentPoint = idx === iteration.current_point;
            const isCore = iteration.core_points.includes(idx);
            const isBorder = iteration.border_points.includes(idx);
            const isNoise = iteration.noise_points.includes(idx);

            ctx.beginPath(); ctx.arc(x, y, isCurrentPoint ? 8 : 6, 0, Math.PI * 2);

            const isInitialIteration = playback.currentIndex === 0 && iteration.phase === "initialization";

            if (isCurrentPoint) ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
            else if (isInitialIteration) ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
            else if (isCore) ctx.fillStyle = clusterColors[label % clusterColors.length];
            else if (isBorder) ctx.fillStyle = label >= 0 ? clusterColors[label % clusterColors.length] : 'rgba(251, 146, 60, 0.7)';
            else if (isNoise && showNoisePoints) ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
            else if (label >= 0) ctx.fillStyle = clusterColors[label % clusterColors.length];
            else ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';

            ctx.fill();

            if (isCurrentPoint) { ctx.strokeStyle = '#000'; ctx.lineWidth = 2; }
            else if (isCore && showCorePoints) { ctx.strokeStyle = '#000'; ctx.lineWidth = 2; }
            else if (isBorder && showBorderPoints) { ctx.setLineDash([2, 2]); ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5; }
            else if (isNoise && showNoisePoints) { ctx.strokeStyle = '#6b7280'; ctx.lineWidth = 1; }
            else { ctx.strokeStyle = '#333'; ctx.lineWidth = 1; }

            ctx.stroke(); ctx.setLineDash([]);

            if (isCurrentPoint && showEpsilonRadius) {
                iteration.neighbors.forEach(neighborIdx => {
                    if (neighborIdx !== idx) {
                        const neighbor = dataPoints[neighborIdx];
                        const nx = ((neighbor.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
                        const ny = ((scale.y.max - neighbor.y) / (scale.y.max - scale.y.min)) * height;
                        ctx.beginPath(); ctx.setLineDash([2, 2]); ctx.moveTo(x, y); ctx.lineTo(nx, ny);
                        ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
                    }
                });
            }
        });
    },[playback.currentIndex, playback.active, dataPoints, eps, showCorePoints, showBorderPoints, showNoisePoints, showEpsilonRadius]);

    const SampleDataModal = () => (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
            <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', maxWidth: '500px', width: '90%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '600', margin: 0 }}>Generate Sample Data</h2>
                    <button onClick={() => setShowSampleDataModal(false)} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: '#6b7280' }}>×</button>
                </div>

                <div style={{ marginBottom: '1.25rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Dataset Type</label>
                    <select value={datasetType} onChange={(e) => setDatasetType(e.target.value)} style={{ width: '100%', padding: '0.6rem', border: '1px solid #d1d5db', borderRadius: '0.375rem', backgroundColor: 'white' }}>
                        <option value="blobs">Isotropic Blobs</option>
                        <option value="moons">Two Moons</option>
                        <option value="circles">Concentric Circles</option>
                    </select>
                </div>

                <div style={{ marginBottom: '1.25rem' }}>
                    <label htmlFor="samples-slider" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
                    <input id="samples-slider" type="range" min="50" max="500" step="50" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                        <span>50 (Fewer Points)</span><span>500 (More points)</span>
                    </div>
                </div>

                {datasetType === 'blobs' && (
                    <div style={{ marginBottom: '1.25rem' }}>
                        <label htmlFor="clusters-gen-slider" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Clusters: {sampleClusters}</label>
                        <input id="clusters-gen-slider" type="range" min="2" max="8" step="1" value={sampleClusters} onChange={(e) => setSampleClusters(Number(e.target.value))} style={{ width: '100%' }} />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                            <span>2 Clusters</span><span>8 Clusters</span>
                        </div>
                    </div>
                )}

                <div style={{ marginBottom: '1.25rem' }}>
                    <label htmlFor="noise-slider" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Noise Level: {noiseLevel.toFixed(2)}</label>
                    <input id="noise-slider" type="range" min="0.01" max="0.2" step="0.01" value={noiseLevel} onChange={(e) => setNoiseLevel(Number(e.target.value))} style={{ width: '100%' }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                        <span>0.01 (Fewer Spread Samples)</span><span>0.2 (More Spread Samples)</span>
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
                        <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                    <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
                </button>
                <h1 className="model-title">DBSCAN Clustering </h1>
            </div>

            <p className="model-description">
                Density-Based Spatial Clustering of Applications with Noise (DBSCAN) groups points that are close to each other. It finds clusters of arbitrary shape and identifies noise.
                <InfoButton algoId="dbscan" />
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
                                <button className="sample-data-button" onClick={loadSampleData} disabled={loading || algorithmStatus !== "idle"} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Load Sample Data</button>
                                <button className="reset-button" onClick={resetData} disabled={loading || dataPoints.length === 0} style={{ padding: '0.5rem 0.75rem', backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer', fontSize: '0.875rem' }}>Reset</button>
                            </div>
                        </div>

                        <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>
                            Click on the canvas below to add data points.
                        </p>

                        <div style={{ marginBottom: '1rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', width: '100%', height: 0, paddingBottom: '100%' }}>
                            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                                <canvas ref={canvasRef} width={canvasWidth} height={canvasHeight} onClick={handleCanvasClick} style={{ display: 'block', cursor: algorithmStatus === "idle" ? 'crosshair' : 'default', width: '100%', height: '100%' }} />
                            </div>
                        </div>

                        {/* Moved Legend & How it works to left column bottom */}
                        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%', marginBottom: '1rem' }}>
                            <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>How DBSCAN Works</h3>
                            <div style={{ fontSize: '0.9rem', color: '#4b5563', lineHeight: '1.5' }}>
                                <p style={{ marginBottom: '0.75rem' }}>DBSCAN groups points based on two parameters:</p>
                                <ol style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
                                    <li><strong>Epsilon (ε):</strong> Radius around each point.</li>
                                    <li><strong>MinPoints:</strong> Minimum points to form a dense region.</li>
                                </ol>
                                <ul style={{ paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>
                                    <li><strong>Core:</strong> At least MinPoints within ε.</li>
                                    <li><strong>Border:</strong> In a Core's ε but fewer than MinPoints.</li>
                                    <li><strong>Noise:</strong> Neither core nor border.</li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
                        <h2 className="section-title" style={{ marginBottom: '1rem' }}>Algorithm Controls</h2>

                        <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
                            <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Parameters</h3>

                            <div style={{ marginBottom: '1.5rem' }}>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Epsilon (radius): {eps.toFixed(1)}</label>
                                <input type="range" min="0.1" max="2.0" step="0.1" value={eps} onChange={(e) => setEps(parseFloat(e.target.value))} disabled={algorithmStatus !== "idle"} style={{ width: '100%' }} />
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>0.1 (Small)</span><span>2.0 (Large)</span></div>
                            </div>

                            <div style={{ marginBottom: '1.5rem' }}>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Minimum Points: {minPoints}</label>
                                <input type="range" min="2" max="15" step="1" value={minPoints} onChange={(e) => setMinPoints(parseInt(e.target.value))} disabled={algorithmStatus !== "idle"} style={{ width: '100%' }} />
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>2 (Few needed)</span><span>15 (Many needed)</span></div>
                            </div>

                            <div style={{ marginBottom: '1.5rem' }}>
                                <h4 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500', color: '#4b5563' }}>Visualization Options</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}><input type="checkbox" checked={showCorePoints} onChange={(e) => setShowCorePoints(e.target.checked)} /><span style={{ fontSize: '0.9rem' }}>Show Core Points</span></label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}><input type="checkbox" checked={showBorderPoints} onChange={(e) => setShowBorderPoints(e.target.checked)} /><span style={{ fontSize: '0.9rem' }}>Show Border Points</span></label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}><input type="checkbox" checked={showNoisePoints} onChange={(e) => setShowNoisePoints(e.target.checked)} /><span style={{ fontSize: '0.9rem' }}>Show Noise Points</span></label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}><input type="checkbox" checked={showEpsilonRadius} onChange={(e) => setShowEpsilonRadius(e.target.checked)} /><span style={{ fontSize: '0.9rem' }}>Show Epsilon Radius</span></label>
                                </div>
                            </div>

                            <div style={{ marginTop: '2rem' }}>
                                <button onClick={runDBSCAN} disabled={loading || dataPoints.length < 2 || algorithmStatus !== "idle"} style={{ width: '100%', padding: '0.75rem 1rem', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', fontSize: '1rem', opacity: (loading || dataPoints.length < 2 || algorithmStatus !== "idle") ? 0.7 : 1 }}>
                                    {loading ? (
                                        <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Running...</>
                                    ) : 'Run DBSCAN'}
                                </button>
                            </div>
                        </div>

                        {/* Legend */}
                        <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                            <h4 style={{ marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500' }}>Legend</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.7)', border: '1px solid #333' }}></div><span style={{ fontSize: '0.85rem' }}>Initial</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(239, 68, 68, 0.7)', border: '1px solid #000' }}></div><span style={{ fontSize: '0.85rem' }}>Current</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.7)', border: '2px solid #000' }}></div><span style={{ fontSize: '0.85rem' }}>Core Point</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(239, 68, 68, 0.7)', border: '2px dashed #333' }}></div><span style={{ fontSize: '0.85rem' }}>Border Point</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(156, 163, 175, 0.7)', border: '1px solid #6b7280' }}></div><span style={{ fontSize: '0.85rem' }}>Noise</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', border: '2px dashed #3b82f6', backgroundColor: 'transparent' }}></div><span style={{ fontSize: '0.85rem' }}>Epsilon Radius</span></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
                {results && (
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                        <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>DBSCAN Analytics Dashboard</h3>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                            {/* (0,0) Density Evolution Animation */}
                            <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                    <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Density Scanning Progress</h4>
                                    <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                        Step: {playback.currentIndex + 1} / {playback.frames.length}
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

                            {/* (0,1) Final Stats and Info */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                                    <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                        Final Clusters Found: {results.num_clusters}
                                    </p>
                                    <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0', fontSize: '0.95rem' }}>
                                        Core: {results.core_points.length} | Border: {results.border_points.length} | Noise: {results.noise_points.length}
                                    </p>
                                </div>
                                <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', flex: 1, justifyContent:'center' }}>
                                    <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Algorithm State</h4>
                                    <p style={{ fontWeight: '500', color: '#4b5563', marginBottom: '0.5rem' }}>
                                        Current Phase: {playback.frames[playback.currentIndex]?.phase.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                    </p>
                                    <p style={{ fontSize: '0.9rem', color: '#6b7280', padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '4px', fontStyle: 'italic' }}>
                                        "{playback.frames[playback.currentIndex]?.message}"
                                    </p>
                                    <p style={{marginTop:'6rem',color: '#1f2937'}}>
                                        A fun fact about DBSCAN is that it doesn't need to be told how many clusters to find. Unlike K-Means, which requires you to pre-define number of clusters ('k'), DBSCAN automatically determines the number of clusters based on density of data.
                                        <br /><br />
                                        <ol style={{marginLeft:'5rem'}}>
                                            <li>It Loves Weird Shapes</li>
                                            <li>It’s Great at Ignoring the "Noise"</li>
                                            <li>It Treats Data Like a "Chameleon"</li>
                                            <li>It Was Invented in 1996</li>
                                        </ol>
                                    </p>
                                </div>
                            </div>

                            {/* (1,0) Progression Area Chart */}
                            {results.progress_history && results.progress_history.length > 0 && (
                                <div style={{ display: 'flex', flexDirection: 'column' }}>
                                    <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Point Classification Progress</h4>
                                    <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                        <ResponsiveContainer>
                                            <AreaChart data={results.progress_history.slice(0, playback.currentIndex + 1)} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                                <XAxis dataKey="iteration" tick={{fontSize: 12}} />
                                                <YAxis tick={{fontSize: 12}} />
                                                <Tooltip />
                                                <Legend verticalAlign="top" height={36}/>
                                                <Area type="monotone" dataKey="Core" stackId="1" stroke="#3b82f6" fill="#3b82f6" />
                                                <Area type="monotone" dataKey="Border" stackId="1" stroke="#f59e0b" fill="#f59e0b" />
                                                <Area type="monotone" dataKey="Noise" stackId="1" stroke="#9ca3af" fill="#9ca3af" />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                        Shows the cumulative area of data points being classified as the algorithm explores.
                                    </p>
                                </div>
                            )}

                            {/* (1,1) Cluster Distribution Bar Chart */}
                            {results.cluster_sizes && results.cluster_sizes.length > 0 && (
                                <div style={{ display: 'flex', flexDirection: 'column' }}>
                                    <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Final Cluster Distribution</h4>
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
                                        Displays how points are partitioned. Noise points are excluded from clusters.
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

export default DBScan;
