/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { runSVM, getSVMSampleData, checkHealth } from '../../api';
import InfoButton from '../../components/InfoButton';

function SVM() {
    const navigate = useNavigate();
    const[dataPairs, setDataPairs] = useState([{ x: '', y: '', class: 0 }]);
    const[loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [backendStatus, setBackendStatus] = useState("checking");
    const [sampleLoading, setSampleLoading] = useState(false);
    const[currentClass, setCurrentClass] = useState(0);
    const [kernel, setKernel] = useState('linear');
    const [decisionBoundary, setDecisionBoundary] = useState(null);
    const[supportVectors, setSupportVectors] = useState([]);
    const [interactionMode, setInteractionMode] = useState('train');
    const [predictedPoints, setPredictedPoints] = useState([]);
    const [showSupportVectors, setShowSupportVectors] = useState(true);

    const [showSampleDataModal, setShowSampleDataModal] = useState(false);
    const[sampleDataType, setSampleDataType] = useState('blobs');
    const [sampleCount, setSampleCount] = useState(40);
    const [sampleClusters, setSampleClusters] = useState(2);
    const[sampleVariance, setSampleVariance] = useState(0.5);

    const[hyperparams, setHyperparams] = useState({
        gamma: 0.1,
        degree: 3,
        coef0: 0.0,
        marginWidth: 1.0
    });

    // --- Playback State for Validation Animation ---
    const [playback, setPlayback] = useState({
        active: false,
        frames:[],
        currentIndex: 0,
        isPlaying: false
    });

    const canvasRef = useRef(null);
    const tempCanvasRef = useRef(null);
    const [canvasDimensions] = useState({ width: 600, height: 600 });

    const scale = { x: { min: -8, max: 8 }, y: { min: -8, max: 8 } };

    // API URL FIX
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

    useEffect(() => {
        const checkBackendHealth = async () => {
            try {
                const health = await axios.get(`${apiUrl}/health`);
                setBackendStatus(health.data.status === "healthy" ? "connected" : "disconnected");
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
            }, 800); // 0.8 seconds per margin frame
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

    useEffect(() => {
        if (kernel === 'linear') {
            setHyperparams(prev => ({ ...prev, gamma: 0.1, degree: 3, coef0: 0.0 }));
        } else if (kernel === 'poly') {
            setHyperparams(prev => ({ ...prev, gamma: 0.1, degree: 3, coef0: 1.0 }));
        } else if (kernel === 'rbf') {
            setHyperparams(prev => ({ ...prev, gamma: 0.1, degree: 3, coef0: 0.0 }));
        } else if (kernel === 'sigmoid') {
            setHyperparams(prev => ({ ...prev, gamma: 0.1, degree: 3, coef0: 1.0 }));
        }
        if (predictedPoints.length > 0) setPredictedPoints([]);
    }, [kernel]);

    useEffect(() => {
        if (predictedPoints.length > 0 && results) setPredictedPoints([]);
    },[hyperparams.gamma, hyperparams.degree, hyperparams.coef0, hyperparams.marginWidth]);

    const drawGrid = (ctx, canvas) => {
        const stepX = canvas.width / 16;
        const stepY = canvas.height / 16;

        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 0.5;

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

    const isSupportVector = (point) => {
        if (!supportVectors) return false;
        for (const sv of supportVectors) {
            if (Math.abs(parseFloat(point.x) - sv[0]) < 0.01 && Math.abs(parseFloat(point.y) - sv[1]) < 0.01) return true;
        }
        return false;
    };

    const drawPoints = (ctx, canvas, points) => {
        if (!points || points.length === 0) return;

        points.forEach(point => {
            if (showSupportVectors && supportVectors && isSupportVector(point)) return;

            const x = ((parseFloat(point.x) - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
            const y = canvas.height - ((parseFloat(point.y) - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;

            if (point.isPredicted) {
                const squareSize = 10;
                ctx.beginPath();
                ctx.rect(x - squareSize/2, y - squareSize/2, squareSize, squareSize);

                if (point.predictedClass === undefined) {
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
                } else {
                    ctx.fillStyle = point.predictedClass === 0 ? 'rgba(59, 130, 246, 0.7)' : 'rgba(239, 68, 68, 0.7)';
                }
                ctx.setLineDash([2, 2]);
            } else {
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fillStyle = point.class === 0 ? 'rgba(59, 130, 246, 0.7)' : 'rgba(239, 68, 68, 0.7)';
                ctx.setLineDash([]);
            }

            ctx.fill(); ctx.strokeStyle = '#000000'; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
        });
    };

    const drawSupportVectorMarkers = (ctx, canvas) => {
        if (!supportVectors || supportVectors.length === 0) return;
        supportVectors.forEach((sv, index) => {
            const screenCoords = dataToScreen(sv[0], sv[1]);
            const x = screenCoords.x;
            const y = screenCoords.y;

            ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.strokeStyle = '#000000'; ctx.lineWidth = 3; ctx.stroke();

            const svClass = results?.supportVectorClasses?.[index] ?? 0;
            ctx.fillStyle = svClass === 0 ? 'rgba(30, 80, 200, 0.85)' : 'rgba(200, 30, 30, 0.85)';
            ctx.fill();
        });
    };

    const screenToData = (x, y) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        const dataX = scale.x.min + (x / canvas.width) * (scale.x.max - scale.x.min);
        const dataY = scale.y.min + ((canvas.height - y) / canvas.height) * (scale.y.max - scale.y.min);
        return { x: dataX, y: dataY };
    };

    const dataToScreen = (dataX, dataY) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        const x = ((dataX - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = canvas.height - ((dataY - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
        return { x, y };
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawGrid(ctx, canvas);

        if (decisionBoundary) {
            const img = new Image();
            img.onerror = () => { setError("Failed to display decision boundary."); };
            img.onload = () => {
                ctx.globalAlpha = 0.7;
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                ctx.globalAlpha = 1.0;

                drawPoints(ctx, canvas, [...getValidPoints(), ...predictedPoints]);
                if (showSupportVectors && supportVectors && supportVectors.length) {
                    drawSupportVectorMarkers(ctx, canvas);
                }
            };
            img.src = `data:image/png;base64,${decisionBoundary}`;
        } else {
            drawPoints(ctx, canvas,[...getValidPoints(), ...predictedPoints]);
        }
    },[dataPairs, predictedPoints, decisionBoundary, supportVectors, showSupportVectors]);

    const handleCanvasClick = (e) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        const dataPoint = screenToData(x, y);

        if (interactionMode === 'train') {
            const newPoint = { x: dataPoint.x.toFixed(2), y: dataPoint.y.toFixed(2), class: currentClass };
            setDataPairs([...dataPairs, newPoint]);
            setDecisionBoundary(null);
            setResults(null);
            setSupportVectors([]);
            setPredictedPoints([]);
            setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
        } else {
            const pointToPredict = { x: parseFloat(dataPoint.x.toFixed(2)), y: parseFloat(dataPoint.y.toFixed(2)), isPredicted: true };
            setPredictedPoints([...predictedPoints, pointToPredict]);
        }
    };

    const loadSampleData = () => setShowSampleDataModal(true);

    const generateSampleDataWithOptions = () => {
        setLoading(true); setShowSampleDataModal(false);
        axios.post(`${apiUrl}/svm/sample_data`, {
            dataset_type: sampleDataType, count: sampleCount, n_clusters: sampleClusters, variance: sampleVariance
        })
        .then(response => {
            if (response.data && response.data.X && response.data.y) {
                const pairs =[];
                for (let i = 0; i < response.data.X.length; i++) {
                    pairs.push({
                        x: parseFloat(response.data.X[i][0]).toFixed(2),
                        y: parseFloat(response.data.X[i][1]).toFixed(2),
                        class: parseInt(response.data.y[i])
                    });
                }
                setDataPairs(pairs);
                setPredictedPoints([]); setResults(null); setDecisionBoundary(null); setSupportVectors([]);
                setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
            }
        })
        .catch(err => { setError("Failed to generate sample data."); })
        .finally(() => setLoading(false));
    };

    const resetData = () => {
        setDataPairs([{ x: '', y: '', class: 0 }]);
        setResults(null); setError(null); setPredictedPoints([]);
        setDecisionBoundary(null); setSupportVectors([]);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });
    };

    const getValidPoints = () => {
        return dataPairs
            .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
            .map(pair => ({ x: parseFloat(pair.x), y: parseFloat(pair.y), class: pair.class }));
    };

    const handleRunModel = async () => {
        const validPairs = getValidPoints();
        if (validPairs.length < 2) { setError('Please add at least 2 data points.'); return; }
        const uniqueClasses =[...new Set(validPairs.map(p => p.class))];
        if (uniqueClasses.length < 2) { setError('Please add points from at least two different classes.'); return; }

        setError(null); setLoading(true); setPredictedPoints([]); setResults(null);
        setPlayback({ active: false, frames:[], currentIndex: 0, isPlaying: false });

        try {
            const apiData = {
                X: validPairs.map(pair => ({ x: parseFloat(pair.x), y: parseFloat(pair.y) })),
                y: validPairs.map(pair => parseInt(pair.class)),
                kernel: kernel,
                gamma: kernel === 'rbf' || kernel === 'poly' || kernel === 'sigmoid' ?
                    (hyperparams.gamma === 'scale' || hyperparams.gamma === 'auto' ? hyperparams.gamma : parseFloat(hyperparams.gamma)) : 'auto',
                degree: parseInt(hyperparams.degree),
                coef0: parseFloat(hyperparams.coef0),
                marginWidth: parseFloat(hyperparams.marginWidth)
            };

            const response = await axios.post(`${apiUrl}/svm`, apiData);
            if (response.data.error) throw new Error(response.data.error);

            setResults(response.data);

            if (response.data.history && response.data.history.length > 0) {
                setPlayback({
                    active: true,
                    frames: response.data.history,
                    currentIndex: response.data.history.length - 1,
                    isPlaying: false
                });
            }

            if (response.data.decisionBoundary) setDecisionBoundary(response.data.decisionBoundary);
            if (response.data.supportVectors) setSupportVectors(response.data.supportVectors);

        } catch (err) {
            setError(`Error: ${err.message || 'An error occurred.'}`);
        } finally {
            setLoading(false);
        }
    };

    const handlePredict = () => {
        if (!results || !results.model_info) { setError('Please run SVM first.'); return; }
        if (predictedPoints.length === 0) { setError('Please add test points.'); return; }

        setError(null); setLoading(true);

        try {
            const newPredictedPoints = [...predictedPoints].map(point => {
                const predictedClass = classifyPointWithSVMModel(point);
                return { ...point, predictedClass, isPredicted: true };
            });
            setPredictedPoints(newPredictedPoints);
        } catch (err) {
            setError(`Error during prediction: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const classifyPointWithSVMModel = (point) => {
        if (!results || !results.model_info) return 0;
        if (decisionBoundary && canvasRef.current) {
            const { x, y } = dataToScreen(point.x, point.y);
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (tempCanvasRef.current) {
                const tempCanvas = tempCanvasRef.current;
                const tempCtx = tempCanvas.getContext('2d');
                const img = new Image();
                img.onload = () => {
                    tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
                    tempCtx.drawImage(img, 0, 0, tempCanvas.width, tempCanvas.height);
                };
                img.src = `data:image/png;base64,${decisionBoundary}`;
                const colorData = ctx.getImageData(x, y, 1, 1).data;
                return colorData[0] > colorData[2] ? 1 : 0;
            }
        }

        const info = results.model_info;
        if (info.kernel === 'linear' && info.weights) {
            const weights = info.weights[0];
            const intercept = info.intercept[0];
            return weights[0] * point.x + weights[1] * point.y + intercept > 0 ? 1 : 0;
        }

        if (supportVectors && supportVectors.length > 0 && results.supportVectorClasses) {
            let minDist = Infinity;
            let nearestSVIndex = -1;
            for (let i = 0; i < supportVectors.length; i++) {
                const sv = supportVectors[i];
                const dist = Math.sqrt(Math.pow(point.x - sv[0], 2) + Math.pow(point.y - sv[1], 2));
                if (dist < minDist) { minDist = dist; nearestSVIndex = i; }
            }
            if (nearestSVIndex >= 0 && results.supportVectorClasses) return results.supportVectorClasses[nearestSVIndex];
        }
        const isPositiveSide = (point.x > 0 && point.y > 0) || (point.x < 0 && point.y < 0);
        return isPositiveSide ? 1 : 0;
    };

    const handleHyperparamChange = (param, value) => {
        setHyperparams({ ...hyperparams, [param]: value });
    };

    const SampleDataModal = () => (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
            <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', width: '90%', maxWidth: '550px' }}>
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                        <span>20 (Fewer Points)</span><span>200 (More points)</span>
                    </div>
                </div>

                <div style={{ marginBottom: '1.25rem' }}>
                    <label htmlFor="variance" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Variance: {sampleVariance.toFixed(1)}</label>
                    <input id="variance" type="range" min="0.1" max="1.0" step="0.1" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                        <span>0.1 (Less Variance)</span><span>1.0 (More Variance)</span>
                    </div>
                </div>

                {sampleDataType === 'blobs' && (
                    <div style={{ marginBottom: '1.25rem' }}>
                        <label htmlFor="clusters" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Clusters: {sampleClusters}</label>
                        <input id="clusters" type="range" min="2" max="4" step="1" value={sampleClusters} onChange={(e) => setSampleClusters(Number(e.target.value))} style={{ width: '100%' }} />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                            <span>2 Clusters</span><span>4 Clusters</span>
                        </div>
                    </div>
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
                    <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
                </button>
                <h1 className="model-title">Support Vector Machine (SVM) </h1>
            </div>

            <p className="model-description">
                Support Vector Machine (SVM) is a supervised learning algorithm used for classification & regression tasks.
                It finds optimal hyperplane to separate data points into different classes.
                <InfoButton algoId="svm" />
            </p>

            {backendStatus === "disconnected" && (
                <div className="backend-status error">
                    <span>Backend service is not responding. Please make sure the Flask server is running on port 5000.</span>
                </div>
            )}

            <div className="content-container" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>

                {/* TOP ROW: Input & Controls */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '1.5rem' }}>

                    <div style={{ width: '100%', gridColumn: '1 / 2', display: 'flex', flexDirection: 'column' }}>
                        <div className="section-header">
                            <h2 className="section-title">Interactive SVM</h2>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <button className="sample-data-button" onClick={loadSampleData} disabled={sampleLoading} style={{ backgroundColor: '#3b82f6', color: 'white'}}>{sampleLoading ? 'Load Sample Data' : 'Load Sample Data'}</button>
                                <button className="sample-data-button" onClick={resetData} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Reset Data</button>
                            </div>
                        </div>

                        {error && <div className="error-message">{error}</div>}

                        <div style={{ marginBottom: '1rem' }}>
                            <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>Click on the graph below to add data points. Current mode: <strong>{interactionMode === 'train' ? 'Training' : 'Testing'}</strong></p>
                        </div>

                        <div style={{ marginBottom: '1rem', border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)', width: '100%', height: '0', paddingBottom: '100%' }}>
                            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                                <canvas ref={canvasRef} width={canvasDimensions.width} height={canvasDimensions.height} onClick={handleCanvasClick} style={{ display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
                            </div>
                            <canvas ref={tempCanvasRef} width={canvasDimensions.width} height={canvasDimensions.height} style={{display: 'none'}} />
                            <div style={{ position: 'absolute', bottom: '10px', right: '10px', padding: '4px 8px', backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '4px', fontSize: '0.8rem', color: '#4b5563', pointerEvents: 'none' }}>
                                Click to add {interactionMode === 'train' ? 'training' : 'testing'} point
                            </div>
                        </div>

                        <div style={{ marginTop: '0.5rem', marginBottom: '1rem', backgroundColor: '#f9fafb', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #e5e7eb', width: '100%' }}>
                            <p style={{ fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>Statistics:</p>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                                <div><strong>Training Points:</strong> {getValidPoints().length}</div>
                                <div><strong>Points to Predict:</strong> {predictedPoints.length}</div>
                                <div><strong>Support Vectors:</strong> {supportVectors.length}</div>
                            </div>
                        </div>

                        <div style={{ width: '100%', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', fontSize: '0.85rem' }}>
                            <h3 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>Legend</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.7)', border: '1px solid #333' }}></div><span>Class 0</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(239, 68, 68, 0.7)', border: '1px solid #333' }}></div><span>Class 1</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', backgroundColor: 'rgba(156, 163, 175, 0.7)', border: '1px solid #333', borderStyle: 'dashed' }}></div><span>Unpredicted</span></div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(30, 80, 200, 0.85)', border: '3px solid #000000' }}></div><span>Support Vector</span></div>
                            </div>
                        </div>
                    </div>

                    <div style={{ width: '100%', gridColumn: '2 / 3', display: 'flex', flexDirection: 'column' }}>
                        <h2 className="section-title">Controls & Results</h2>
                        <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
                            <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Parameters</h3>

                            <div style={{ marginBottom: '1.5rem' }}>
                                <label htmlFor="kernel-selector" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Kernel Type</label>
                                <select id="kernel-selector" value={kernel} onChange={(e) => setKernel(e.target.value)} style={{ width: '100%', padding: '10px', border: '1px solid #d1d5db', borderRadius: '6px', backgroundColor: 'white', fontSize: '1rem', color: '#4b5563', cursor: 'pointer' }}>
                                    <option value="linear">Linear</option>
                                    <option value="poly">Polynomial</option>
                                    <option value="rbf">RBF (Radial Basis Function)</option>
                                    <option value="sigmoid">Sigmoid</option>
                                </select>
                            </div>

                            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                                <label htmlFor="margin-width" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Margin Width: {hyperparams.marginWidth}</label>
                                <input id="margin-width" type="range" min="0.1" max="5.0" step="0.1" value={hyperparams.marginWidth} onChange={(e) => handleHyperparamChange('marginWidth', e.target.value)} style={{ width: '100%' }} />
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem' }}><span>0.1 (Hard Margin)</span><span>5.0 (Soft Margin)</span></div>
                            </div>

                            {(kernel === 'rbf' || kernel === 'poly' || kernel === 'sigmoid') && (
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <label htmlFor="gamma-param" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563', fontSize: '1rem' }}>Gamma: {hyperparams.gamma}</label>
                                    <input id="gamma-param" type="range" min="0.01" max="2.0" step="0.01" value={hyperparams.gamma} onChange={(e) => handleHyperparamChange('gamma', e.target.value)} style={{ width: '100%' }} />
                                    <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>
                                        Gamma controls the influence a single point reaches. High Gamma = tight, jagged islands around points.
                                    </p>
                                </div>
                            )}

                            {kernel === 'poly' && (
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <label htmlFor="degree-param" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Polynomial Degree: {hyperparams.degree}</label>
                                    <input id="degree-param" type="range" min="1" max="10" step="1" value={hyperparams.degree} onChange={(e) => handleHyperparamChange('degree', e.target.value)} style={{ width: '100%' }} />
                                    <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>
                                        Degree maps the data into higher mathematical dimensions to slice it with a hyperplane.
                                    </p>
                                </div>
                            )}

                            {(kernel === 'poly' || kernel === 'sigmoid') && (
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <label htmlFor="coef0-param" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Coef0: {hyperparams.coef0}</label>
                                    <input id="coef0-param" type="range" min="0" max="5" step="0.1" value={hyperparams.coef0} onChange={(e) => handleHyperparamChange('coef0', e.target.value)} style={{ width: '100%' }} />
                                    <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>
                                        Coef0 controls how much the model is influenced by high-degree polynomials versus low-degree polynomials.
                                    </p>
                                </div>
                            )}

                            <div style={{ marginBottom: '1.5rem' }}>
                                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontWeight: '500', color: '#4b5563' }}>
                                    <input type="checkbox" checked={showSupportVectors} onChange={() => setShowSupportVectors(!showSupportVectors)} style={{ marginRight: '0.5rem' }} />
                                    Show Support Vectors on Canvas
                                </label>
                            </div>

                            <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Interaction Mode</h3>
                            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
                                <button onClick={() => setInteractionMode('train')} style={{ padding: '0.75rem 1rem', backgroundColor: interactionMode === 'train' ? '#3b82f6' : '#e5e7eb', color: interactionMode === 'train' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', flex: 1 }}>Training Points</button>
                                <button onClick={() => setInteractionMode('test')} style={{ padding: '0.75rem 1rem', backgroundColor: interactionMode === 'test' ? '#3b82f6' : '#e5e7eb', color: interactionMode === 'test' ? 'white' : '#4b5563', border: 'none', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: '500', flex: 1 }}>Prediction Points</button>
                            </div>

                            {interactionMode === 'train' && (
                                <div style={{ marginBottom: '0.75rem' }}>
                                    <p style={{ marginBottom: '0.75rem', color: '#4b5563', fontSize: '1rem' }}>Select class for training points:</p>
                                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                                        <button onClick={() => setCurrentClass(0)} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: currentClass === 0 ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.1)', color: currentClass === 0 ? 'white' : '#1e40af', cursor: 'pointer', fontWeight: '500', flex: 1 }}>Class 0</button>
                                        <button onClick={() => setCurrentClass(1)} style={{ padding: '0.75rem 0.5rem', border: 'none', borderRadius: '0.5rem', backgroundColor: currentClass === 1 ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.1)', color: currentClass === 1 ? 'white' : '#b91c1c', cursor: 'pointer', fontWeight: '500', flex: 1 }}>Class 1</button>
                                    </div>
                                </div>
                            )}
                        </div>

                        <div style={{ marginBottom: '1.5rem', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%' }}>
                            <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Actions</h3>
                            <button onClick={handleRunModel} disabled={loading || backendStatus === "disconnected"} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1, marginBottom: '1.25rem' }}>
                                {loading ? (
                                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Running...</>
                                ) : 'Run SVM'}
                            </button>
                            <button onClick={handlePredict} disabled={loading || backendStatus === "disconnected" || !results} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '0.9rem', fontSize: '1.05rem', fontWeight: '500', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', opacity: (loading || backendStatus === "disconnected" || !results) ? 0.7 : 1 }}>
                                {loading ? 'Predict Points' : 'Predict Points'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
                {results && (
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ width: '100%', backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                            {/* (0,0) Margin Evolution Animation */}
                            <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                    <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Margin Convergence</h4>
                                    <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                        Margin Width: {playback.frames.length > 0 ? playback.frames[playback.currentIndex].margin.toFixed(2) : results.model_info.marginWidth}
                                    </span>
                                </div>
                                <div style={{ textAlign: 'center', marginBottom: '1.5rem', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <img
                                        src={`data:image/png;base64,${playback.frames.length > 0 ? playback.frames[playback.currentIndex].decision_boundary : results.decisionBoundary}`}
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

                            {/* (0,1) Final Stats and Validation Curve Graph */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                <div style={{ backgroundColor: '#f3f4f6', padding: '1rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                                    <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#111827', margin: 0 }}>
                                        Final Accuracy: {(results.accuracy * 100).toFixed(2)}%
                                    </p>
                                    <p style={{ color: '#4b5563', margin: '0.5rem 0 0 0' }}>
                                        Total Support Vectors: {results.model_info.total_support_vectors} | Kernel: {kernel.toUpperCase()}
                                    </p>
                                </div>

                                {results.history && results.history.length > 0 && (
                                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                                        <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                                            Accuracy vs. Margin Width
                                        </h4>
                                        <div style={{ height: 250, width: '100%' }}>
                                            <ResponsiveContainer>
                                                <LineChart data={results.history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                                    {/* We reverse the X Axis visually so it shows Soft -> Hard margin */}
                                                    <XAxis dataKey="margin" reversed={true} tick={{fontSize: 12}} />
                                                    <YAxis domain={[0, 1]} tick={{fontSize: 12}} />
                                                    <Tooltip />
                                                    <Legend verticalAlign="top" height={36}/>
                                                    <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" name="Accuracy" strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                                </LineChart>
                                            </ResponsiveContainer>
                                        </div>
                                        <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                            A harder margin (lower width) fits data tighter, but too hard causes overfitting.
                                        </p>
                                    </div>
                                )}
                            </div>

                            {/* (1,0) SV Plot Analysis */}
                            {results.history && results.history.length > 0 && (
                                <div style={{ display: 'flex', flexDirection: 'column' }}>
                                    <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>Support Vectors vs. Margin Width</h4>
                                    <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                        <ResponsiveContainer>
                                            <LineChart data={results.history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                                <XAxis dataKey="margin" reversed={true} tick={{fontSize: 12}} />
                                                <YAxis tick={{fontSize: 12}} />
                                                <Tooltip />
                                                <Legend verticalAlign="top" height={36}/>
                                                <Line type="monotone" dataKey="n_support" stroke="#8b5cf6" name="Number of Support Vectors" strokeWidth={2} dot={{r: 4}} activeDot={{r: 6}} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                        A wider/softer margin forces the algorithm to rely on more Support Vectors to define the boundary.
                                    </p>
                                </div>
                            )}

                            {/* (1,1) Final High-Res SV Plot */}
                            {results.final_plot && (
                                <div style={{ padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                    <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem', width: '100%' }}>Final Support Vector Boundary</h4>
                                    <div style={{ width: '100%', height: '300px', position: 'relative', display: 'flex', justifyContent: 'center' }}>
                                        <img
                                            src={`data:image/png;base64,${results.final_plot}`}
                                            alt="Support Vector Highlight"
                                            style={{ maxHeight: '100%', maxWidth: '100%', borderRadius: '4px' }}
                                        />
                                    </div>
                                    <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '1rem'}}>
                                        The gold circles indicate the exact data points mathematically chosen as the "Support Vectors" holding up the boundary.
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

export default SVM;
