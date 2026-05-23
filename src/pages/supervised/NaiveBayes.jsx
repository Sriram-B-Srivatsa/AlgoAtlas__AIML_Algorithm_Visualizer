/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function NaiveBayes() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [brushClass, setBrushClass] = useState(0);
  const [pointsMode, setPointsMode] = useState('train');

  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleCount, setSampleCount] = useState(60);
  const [sampleVariance, setSampleVariance] = useState(1.0);
  const [results, setResults] = useState(null);

  const canvasDimensions = useMemo(() => ({ width: 600, height: 600 }),[]);
  const scale = useMemo(() => ({ x: { min: -8, max: 8 }, y: { min: -8, max: 8 } }),[]);
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const dataX = scale.x.min + ((x * (canvas.width / rect.width)) / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - ((y * (canvas.height / rect.height)) / canvas.height) * (scale.y.max - scale.y.min);
    return { x: dataX, y: dataY };
  };

  const handleCanvasClick = (e) => {
    if (loading) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const dataPoint = screenToData(e.clientX - rect.left, e.clientY - rect.top);

    if (pointsMode === 'train') {
        setPoints([...points, { x: dataPoint.x, y: dataPoint.y, class: brushClass }]);
        setResults(null);
    } else {
        setPredictPoints([...predictPoints, { x: dataPoint.x, y: dataPoint.y }]);
    }
  };

  const generateSampleData = () => {
    setLoading(true); setShowSampleDataModal(false); setError(null);
    axios.post(`${apiUrl}/nb/sample_data`, { count: sampleCount, variance: sampleVariance })
    .then(response => {
        if(response.data.error) throw new Error(response.data.error);
        setPoints(response.data.points);
        setResults(null);
    })
    .catch(err => setError("Failed to load sample data"))
    .finally(() => setLoading(false));
  };

  const trainModel = () => {
    const uniqueClasses = new Set(points.map(p => p.class));
    if (uniqueClasses.size < 2) { setError("Place points for at least two different classes."); return; }

    setLoading(true); setError(null);
    axios.post(`${apiUrl}/nb/train`, { points: points })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
    })
    .catch(err => setError("Failed to run Naive Bayes."))
    .finally(() => setLoading(false));
  };

  const handlePrediction = () => {
    if (predictPoints.length === 0) { setError("Add points to predict first!"); return; }
    if (!results) { setError("Train the model first!"); return; }

    setLoading(true); setError(null);
    axios.post(`${apiUrl}/nb/predict`, {
        trained_points: points,
        predict_points: predictPoints
    })
    .then(response => {
      if (response.data.error) throw new Error(response.data.error);
      const formattedResults = response.data.predictions.map((r, idx) => ({
        x: predictPoints[idx].x, y: predictPoints[idx].y, predictedClass: r
      }));
      setPredictions(formattedResults);
    })
    .catch(err => setError("Prediction failed. Ensure backend has /api/nb/predict mapped!"))
    .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 0.5;
    const stepX = canvas.width / 16; const stepY = canvas.height / 16;
    for (let i = 0; i <= 16; i++) {
        ctx.beginPath(); ctx.moveTo(0, i * stepY); ctx.lineTo(canvas.width, i * stepY); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(i * stepX, 0); ctx.lineTo(i * stepX, canvas.height); ctx.stroke();
    }
    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, canvas.height/2); ctx.lineTo(canvas.width, canvas.height/2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#4b5563'; ctx.font = '12px Inter, sans-serif';
    for (let i = 0; i <= 16; i += 2) {
        ctx.fillText((scale.x.min + (i/16)*(scale.x.max - scale.x.min)).toFixed(0), i*stepX - 8, canvas.height - 5);
        ctx.fillText((scale.y.max - (i/16)*(scale.y.max - scale.y.min)).toFixed(0), 5, i*stepY + 4);
    }

    points.forEach(point => {
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = point.class === 0 ? '#ef4444' : '#3b82f6';
      ctx.fill(); ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.stroke();
    });

    if (predictions.length === 0) {
        predictPoints.forEach(point => {
          const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
          const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * canvas.height;
          ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
          ctx.fill(); ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
        });
    }

    predictions.forEach(pred => {
        const x = ((pred.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = ((scale.y.max - pred.y) / (scale.y.max - scale.y.min)) * canvas.height;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = pred.predictedClass === 0 ? '#ef4444' : '#3b82f6';
        ctx.fill();
        ctx.fillStyle = '#000000'; ctx.font = '10px Arial'; ctx.fillText(pred.predictedClass, x + 10, y - 10);
        ctx.setLineDash([2, 2]); ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    });
  }, [points, predictPoints, predictions]);

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {showSampleDataModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
          <div style={{ backgroundColor: 'white', padding: '2rem', borderRadius: '0.5rem', width: '90%', maxWidth: '500px' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '1.5rem' }}>Generate Gaussian Data</h2>
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Number of Samples: {sampleCount}</label>
              <input type="range" min="20" max="200" step="10" value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>20 (Fewer points)</span><span>200 (More points)</span></div>
            </div>
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Variance: {sampleVariance.toFixed(1)}</label>
              <input type="range" min="0.1" max="1.0" step="0.1" value={sampleVariance} onChange={(e) => setSampleVariance(Number(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#6b7280', marginTop: '0.25rem' }}><span>0.1 (Compact)</span><span>1.0 (Spread out)</span></div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button onClick={() => setShowSampleDataModal(false)} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#f3f4f6', cursor: 'pointer', borderRadius: '0.5rem', border: 'none' }}>Cancel</button>
              <button onClick={generateSampleData} style={{ padding: '0.6rem 1.2rem', backgroundColor: '#3b82f6', color: 'white', cursor: 'pointer', borderRadius: '0.5rem', border: 'none' }}>Generate</button>
            </div>
          </div>
        </div>
      )}

      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Gaussian Naive Bayes </h1>
      </div>

      <p className="model-description">Naive Bayes uses Bayes' Probability Theorem. Instead of drawing lines, it builds invisible 3D Bell Curves (Gaussians) over your data points and calculates the exact probability of an item belonging to a class!<InfoButton algoId="nb" /></p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Data Sandbox</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={() => setShowSampleDataModal(true)} style={{backgroundColor:'#3b82f6',color:'white'}}>Load Samples</button>
                <button className="sample-data-button" onClick={() => {setPoints([]); setPredictPoints([]); setPredictions([]); setResults(null);}} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
              </div>
            </div>

            <p style={{ fontSize: '0.85rem', color: '#6b7280', fontWeight: '500', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              Click on the grid below to plot your data points.
            </p>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Interaction Mode</h3>
                <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
                    <button onClick={() => setPointsMode('train')} style={{ padding: '0.75rem', backgroundColor: pointsMode === 'train' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'train' ? 'white' : '#4b5563', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Training Points</button>
                    <button onClick={() => setPointsMode('predict')} style={{ padding: '0.75rem', backgroundColor: pointsMode === 'predict' ? '#3b82f6' : '#e5e7eb', color: pointsMode === 'predict' ? 'white' : '#4b5563', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Prediction Points</button>
                </div>
                {pointsMode === 'train' && (
                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <button onClick={() => setBrushClass(0)} style={{ padding: '0.5rem', backgroundColor: brushClass === 0 ? '#ef4444' : '#fef2f2', color: brushClass === 0 ? 'white' : '#b91c1c', border: '1px solid #fecaca', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Class 0 (Red)</button>
                        <button onClick={() => setBrushClass(1)} style={{ padding: '0.5rem', backgroundColor: brushClass === 1 ? '#3b82f6' : '#eff6ff', color: brushClass === 1 ? 'white' : '#1e40af', border: '1px solid #bfdbfe', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Class 1 (Blue)</button>
                    </div>
                )}
            </div>

            <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', width: '100%', paddingBottom: '100%' }}>
              <canvas ref={canvasRef} width={600} height={600} onClick={handleCanvasClick} style={{ position: 'absolute', top: 0, left: 0, display: 'block', cursor: 'crosshair', width: '100%', height: '100%' }} />
              <div style={{ position: 'absolute', bottom: '10px', right: '10px', padding: '4px 8px', backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '4px', fontSize: '0.8rem', color: '#4b5563', pointerEvents: 'none' }}>
                {pointsMode === 'train' ? 'Click to add training point' : 'Click to add prediction point'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Algorithm Overview</h2>
            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#1e40af' }}>Why "Naive"?</h4>
                <p style={{ margin: 0, fontSize: '0.9rem', color: '#3b82f6', lineHeight: '1.5', paddingBottom: '1rem', borderBottom: '1px solid #e5e7eb' }}>
                    It assumes that the X-axis and Y-axis have absolutely nothing to do with each other (they are mathematically independent). This makes the math lightning fast!
                </p>
                <h4 style={{ margin: '1rem 0 0.5rem 0', color: '#1e40af' }}>Bayes' Theorem Formula:</h4>
                <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '4px', textAlign: 'center', fontFamily: 'serif', fontSize: '1.2rem', fontStyle: 'italic', fontWeight: 'bold' }}>
                    P(A|B) = [P(B|A) * P(A)] / P(B)
                </div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                <button onClick={trainModel} disabled={loading || points.length === 0} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', marginBottom: '1rem' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Calculating Probabilities...</>
                ) : 'Train Naive Bayes'}
                </button>
                <button onClick={handlePrediction} disabled={loading || !results || predictPoints.length === 0} style={{ width: '100%', backgroundColor: '#8b5cf6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: (!results || predictPoints.length === 0) ? 'not-allowed' : 'pointer' }}>
                Predict Points
                </button>
            </div>
          </div>
        </div>

        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', width: '100%' }}>Final Decision Boundary</h4>
                        <img src={`data:image/png;base64,${results.boundary_image}`} alt="Boundary" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}/>
                        <p style={{fontSize: '0.85rem', color: '#4b5563', textAlign: 'center', margin: 0}}>Because it uses circular Bell Curves, the boundary it draws between the classes is usually a smooth parabola or curve!</p>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Accuracy: {(results.accuracy * 100).toFixed(2)}%</p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Prior Probabilities P(A)</h4>
                            <div style={{ height: 180, width: '100%' }}>
                                <ResponsiveContainer>
                                    <BarChart data={results.priors}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="class" />
                                        <YAxis domain={[0, 100]} />
                                        <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
                                        <Bar dataKey="prob" name="Prior Probability" fill="#8b5cf6" radius={[4,4,0,0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', margin: '10px 0 0 0'}}>
                                Before looking at the exact coordinates, what are the odds a random point is Red vs Blue?
                            </p>
                        </div>
                    </div>

                    {/* NEW GRAPH: Gaussian Bell Curves! */}
                    <div style={{ display: 'flex', flexDirection: 'column', gridColumn: '1 / 3' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Gaussian Distributions P(B|A)</h4>
                        <div style={{ height: 300, width: '100%' }}>
                            <ResponsiveContainer>
                                <AreaChart data={results.gaussian_data}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="x_val" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Area type="monotone" dataKey="Class_0_Dist" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} name="Class 0 Bell Curve" />
                                    <Area type="monotone" dataKey="Class_1_Dist" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} name="Class 1 Bell Curve" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        <p style={{fontSize: '0.9rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                            This is the physical manifestation of the math! The AI builds a 3D Bell Curve over the Red dots, and a Bell Curve over the Blue dots. Where the two shaded areas overlap is where the AI is confused!
                        </p>
                    </div>

                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default NaiveBayes;
