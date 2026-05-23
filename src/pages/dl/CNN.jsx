/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function CNN() {
  const navigate = useNavigate();
  const[loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");
  const [results, setResults] = useState(null);

  // Canvas Drawing State
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Parameters
  const [epochs, setEpochs] = useState(50);

  // Playback state for showing different Convolution Filters
  const [filterIndex, setFilterIndex] = useState(0);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    axios.get(`${apiUrl}/health`).then(() => setBackendStatus("connected")).catch(() => setBackendStatus("disconnected"));
    clearCanvas();
  },[apiUrl]);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    setResults(null);
  };

  // Drawing Functions
  const startDrawing = (e) => {
    setIsDrawing(true);
    draw(e);
  };

  const endDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.beginPath(); // Reset path
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  // Extract 28x28 array from canvas
  const getCanvasDataArray = () => {
    const canvas = canvasRef.current;
    // Create a tiny hidden canvas to downscale the image to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');

    // Draw the large canvas onto the tiny 28x28 canvas
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    // Extract pixel data
    const imgData = tempCtx.getImageData(0, 0, 28, 28).data;
    const grayscaleArray =[];

    // Convert RGBA to Grayscale (0 to 1)
    for (let i = 0; i < imgData.length; i += 4) {
      grayscaleArray.push(imgData[i] / 255.0);
    }
    return grayscaleArray;
  };

  const trainModel = () => {
    setLoading(true); setError(null);

    const imageData = getCanvasDataArray();
    // Check if canvas is completely empty (all 0s)
    if (imageData.every(val => val === 0)) {
        setError("Please draw a digit (0-9) on the canvas first!");
        setLoading(false);
        return;
    }

    axios.post(`${apiUrl}/cnn/train`, {
      image: imageData,
      parameters: { epochs: epochs }
    })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        setFilterIndex(0);
    })
    .catch(err => setError("Failed to run CNN. Make sure backend is running."))
    .finally(() => setLoading(false));
  };

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Convolutional Neural Networks (CNN) </h1>
      </div>

      <p className="model-description">
        CNNs process image data. Instead of looking at every pixel individually, they use mathematical "Filters" (Convolutions) to slide across the image, detecting Edges, Textures, and Shapes layer by layer.
        <InfoButton algoId="cnn" />
      </p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Draw a Digit (0-9)</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={clearCanvas} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear Pad</button>
              </div>
            </div>

            <div style={{ border: '4px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: 'black', width: '100%', paddingBottom: '100%', cursor: 'crosshair' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <canvas
                    ref={canvasRef}
                    width={400}
                    height={400}
                    onMouseDown={startDrawing}
                    onMouseUp={endDrawing}
                    onMouseOut={endDrawing}
                    onMouseMove={draw}
                    style={{ display: 'block', width: '100%', height: '100%', touchAction: 'none' }}
                />
              </div>
            </div>
            <p style={{ color: '#4b5563', marginTop: '0.5rem', fontSize: '0.9rem', textAlign: 'center' }}>
                Draw a really large, clear number in the exact center of the black box.
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Network Architecture</h2>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>

              {/* UPDATED EPOCHS SLIDER WITH DETAILS */}
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Training Epochs: {epochs}</h3>
              <input type="range" min="10" max="200" step="10" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                  <span>10 (Underfits, dumb AI)</span>
                  <span>200 (Learns patterns well)</span>
              </div>
              <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>
                  An epoch is one complete pass through the training dataset. Too few, and the AI hasn't learned the shapes. Too many, and it might overfit and memorize the data instead of actually learning.
              </p>

              {/* UPDATED LAYERS WITH EXPLANATIONS */}
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Network Layers</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ padding: '0.75rem', backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#1e40af' }}>1. Input Layer (28x28 Image)</div>
                  <div style={{ fontSize: '0.8rem', color: '#3b82f6', marginTop: '4px' }}>Receives the raw pixels from your drawing pad.</div>
                </div>

                <div style={{ padding: '0.75rem', backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#166534' }}>2. Conv2D Layer (Filters)</div>
                  <div style={{ fontSize: '0.8rem', color: '#22c55e', marginTop: '4px' }}>Slides mathematical filters over the image to highlight physical edges and curves.</div>
                </div>

                <div style={{ padding: '0.75rem', backgroundColor: '#fffbeb', border: '1px solid #fde68a', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#b45309' }}>3. MaxPooling (Downsizing)</div>
                  <div style={{ fontSize: '0.8rem', color: '#f59e0b', marginTop: '4px' }}>Shrinks the image to compress the data and ignore empty black space.</div>
                </div>

                <div style={{ padding: '0.75rem', backgroundColor: '#f3f4f6', border: '1px solid #d1d5db', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#1f2937' }}>4. Flatten & Dense (Neural Net)</div>
                  <div style={{ fontSize: '0.8rem', color: '#4b5563', marginTop: '4px' }}>Turns the 2D image into a 1D list of numbers and feeds it into the deep hidden brain layers.</div>
                </div>

                <div style={{ padding: '0.75rem', backgroundColor: '#fef2f2', border: '1px solid #fca5a5', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: '#b91c1c' }}>5. Softmax Output (0-9)</div>
                  <div style={{ fontSize: '0.8rem', color: '#ef4444', marginTop: '4px' }}>Outputs 10 percentages. The highest percentage is the AI's final guess!</div>
                </div>
              </div>

            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <button onClick={trainModel} disabled={loading} style={{ width: '100%', backgroundColor: loading ? '#93c5fd' : '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Processing Convolutions...</>
                ) : 'Extract Features & Predict'}
              </button>
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Feature Maps Viewer */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>Convolutional Feature Maps</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>
                                Layer 1 & 2
                            </span>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <div style={{ textAlign: 'center' }}>
                                <img src={`data:image/png;base64,${results.original_image}`} alt="Original" style={{ width: '100px', border: '1px solid #d1d5db', borderRadius: '4px' }}/>
                                <p style={{ fontSize: '0.8rem', fontWeight: 'bold', marginTop: '4px' }}>Original</p>
                            </div>
                            <div style={{ fontSize: '24px', color: '#9ca3af' }}>&rarr;</div>
                            <div style={{ textAlign: 'center' }}>
                                <img src={`data:image/png;base64,${results.feature_maps[filterIndex].conv_image}`} alt="Conv" style={{ width: '100px', border: '1px solid #d1d5db', borderRadius: '4px' }}/>
                                <p style={{ fontSize: '0.8rem', fontWeight: 'bold', marginTop: '4px' }}>Conv2D Filter</p>
                            </div>
                            <div style={{ fontSize: '24px', color: '#9ca3af' }}>&rarr;</div>
                            <div style={{ textAlign: 'center' }}>
                                <img src={`data:image/png;base64,${results.feature_maps[filterIndex].pool_image}`} alt="Pool" style={{ width: '100px', border: '1px solid #d1d5db', borderRadius: '4px' }}/>
                                <p style={{ fontSize: '0.8rem', fontWeight: 'bold', marginTop: '4px' }}>Max Pooling</p>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
                            {results.feature_maps.map((fMap, idx) => (
                                <button key={idx} onClick={() => setFilterIndex(idx)} style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #3b82f6', backgroundColor: filterIndex === idx ? '#3b82f6' : 'white', color: filterIndex === idx ? 'white' : '#3b82f6', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 'bold' }}>
                                    {fMap.name}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* (0,1) Final Prediction Stats */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f0fdf4', padding: '1.5rem', borderRadius: '8px', border: '2px solid #22c55e', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#14532d', margin: 0 }}>AI Prediction: It's a "{results.prediction}"</p>
                            <p style={{ color: '#166534', margin: '0.5rem 0 0 0', fontWeight: 'bold' }}>Confidence: {results.confidence.toFixed(2)}%</p>
                        </div>

                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>How Convolutions Work</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                Click the different filters on the left. You will see that the CNN doesn't view the whole image at once. <br/><br/>
                                The <strong>Conv2D Layer</strong> mathematically isolates vertical edges, horizontal lines, and outlines. The <strong>Max Pooling</strong> layer shrinks the image to delete useless blank space. Finally, it feeds those concentrated edges into the Neural Network!
                            </p>
                        </div>
                    </div>

                    {/* (1,0) Training Loss Curve */}
                    {results.loss_history && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Neural Network Convergence (Loss)</h4>
                            <div style={{ height: 250, width: '100%' }}>
                                <ResponsiveContainer>
                                    <LineChart data={results.loss_history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="epoch" />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line type="monotone" dataKey="loss" stroke="#ef4444" name="Crossentropy Loss" strokeWidth={2} dot={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}

                    {/* (1,1) Softmax Probabilities Bar Chart */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Softmax Class Probabilities</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <BarChart data={results.probabilities} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="digit" />
                                    <YAxis domain={[0, 100]} />
                                    <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                                    <Bar dataKey="probability" name="Probability %" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                            The final layer of the network outputs 10 percentages. The tallest bar is the AI's final prediction!
                        </p>
                    </div>
                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default CNN;
