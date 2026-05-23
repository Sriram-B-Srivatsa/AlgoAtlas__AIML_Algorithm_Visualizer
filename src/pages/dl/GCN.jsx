/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function GCN() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");

  const canvasRef = useRef(null);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);

  const [interactionMode, setInteractionMode] = useState('node'); // node or edge
  const [brushClass, setBrushClass] = useState(-1); // -1 = Unlabeled
  const [dragStartNode, setDragStartNode] = useState(null);

  const [epochs, setEpochs] = useState(100);
  const [learningRate, setLearningRate] = useState(0.1);
  const [results, setResults] = useState(null);

  const [playback, setPlayback] = useState({ active: false, frames:[], currentIndex: 0, isPlaying: false });

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    axios.get(`${apiUrl}/health`).then(() => setBackendStatus("connected")).catch(() => setBackendStatus("disconnected"));
  }, [apiUrl]);

  useEffect(() => {
    let interval;
    if (playback.active && playback.isPlaying) {
        interval = setInterval(() => {
            setPlayback(prev => {
                if (prev.currentIndex >= prev.frames.length - 1) {
                    clearInterval(interval); return { ...prev, isPlaying: false };
                }
                return { ...prev, currentIndex: prev.currentIndex + 1 };
            });
        }, 500);
    }
    return () => clearInterval(interval);
  }, [playback.active, playback.isPlaying]);

  const togglePlayback = () => {
      setPlayback(p => {
          if (!p.isPlaying && p.currentIndex >= p.frames.length - 1) return { ...p, currentIndex: 0, isPlaying: true };
          return { ...p, isPlaying: !p.isPlaying };
      });
  };

  const handleCanvasDown = (e) => {
    if (loading) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (interactionMode === 'node') {
        setNodes([...nodes, { id: nodes.length, x, y, class: brushClass }]);
        setResults(null);
    } else if (interactionMode === 'edge') {
        // Find if we clicked on a node (radius of 15 for easy clicking)
        const clickedNode = nodes.find(n => Math.sqrt(Math.pow(n.x - x, 2) + Math.pow(n.y - y, 2)) < 15);
        if (clickedNode) setDragStartNode(clickedNode);
    }
  };

  const handleCanvasUp = (e) => {
    if (interactionMode === 'edge' && dragStartNode) {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        const endNode = nodes.find(n => Math.sqrt(Math.pow(n.x - x, 2) + Math.pow(n.y - y, 2)) < 15);

        if (endNode && endNode.id !== dragStartNode.id) {
            // Check if edge already exists
            const exists = edges.some(e =>
                (e.source === dragStartNode.id && e.target === endNode.id) ||
                (e.source === endNode.id && e.target === dragStartNode.id)
            );
            if (!exists) setEdges([...edges, { source: dragStartNode.id, target: endNode.id }]);
        }
        setDragStartNode(null);
        setResults(null);
    }
  };

  const generateGraph = () => {
      const n =[]; const e =[];
      const numNodes = Math.floor(Math.random() * 5) + 8;
      for(let i=0; i<numNodes; i++) {
        const cls = Math.random() > 0.6 ? (Math.random() > 0.5 ? 1 : 0) : -1;
        n.push({id: i, x: Math.random()*400 + 100, y: Math.random()*400 + 100, class: cls});
      }
      // Force at least one of each class
      n[0].class = 0; n[1].class = 1;
      // Connect nodes randomly based on distance
      for(let i=0; i<numNodes; i++) {
        for(let j=i+1; j<numNodes; j++) {
           const dist = Math.sqrt(Math.pow(n[i].x - n[j].x, 2) + Math.pow(n[i].y - n[j].y, 2));
           if (Math.random() < (80 / (dist + 1))) e.push({source: n[i].id, target: n[j].id});
        }
      }
      setNodes(n); setEdges(e); setResults(null);
  };

  const trainModel = () => {
    if (nodes.length < 3) { setError("Place at least 3 nodes."); return; }
    if (edges.length === 0) { setError("Connect nodes using the Edge tool."); return; }

    setLoading(true); setError(null);
    axios.post(`${apiUrl}/gcn/train`, { nodes, edges, parameters: { epochs, learningRate } })
    .then(response => {
        if (response.data.error) { setError(response.data.error); return; }
        setResults(response.data);
        if (response.data.history) setPlayback({ active: true, frames: response.data.history, currentIndex: 0, isPlaying: true });
    })
    .catch(err => setError("Failed to run GCN."))
    .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, 600, 600);
    ctx.fillStyle = "#f9f9f9"; ctx.fillRect(0, 0, 600, 600);

    // Draw Edges
    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 2;
    edges.forEach(edge => {
        const source = nodes.find(n => n.id === edge.source);
        const target = nodes.find(n => n.id === edge.target);
        if (source && target) {
            ctx.beginPath(); ctx.moveTo(source.x, source.y); ctx.lineTo(target.x, target.y); ctx.stroke();
        }
    });

    // Draw Nodes
    nodes.forEach((node, i) => {
      ctx.beginPath(); ctx.arc(node.x, node.y, 12, 0, Math.PI * 2);

      // If we have predictions, color the unlabeled nodes!
      if (results && node.class === -1) {
          const pred = results.predictions[i];
          ctx.fillStyle = pred === 0 ? '#93c5fd' : '#fca5a5'; // Light blue/red for predictions
      } else {
          if (node.class === 0) ctx.fillStyle = '#3b82f6';
          else if (node.class === 1) ctx.fillStyle = '#ef4444';
          else ctx.fillStyle = '#9ca3af';
      }

      ctx.fill(); ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();

      // Draw ID
      ctx.fillStyle = 'white'; ctx.font = '10px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(node.id, node.x, node.y);
    });
  }, [nodes, edges, results]);

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Graph Convolutional Networks (GCN) </h1>
      </div>

      <p className="model-description">Deep Learning for Networks! Instead of looking at an image, a GCN looks at Nodes and Edges. It mathematically passes "Messages" along the edges so that a node can figure out what it is based on who its friends are!<InfoButton algoId="gcn" /></p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          {/* LEFT COLUMN: Data Input Sandbox */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">Network Canvas</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="sample-data-button" onClick={generateGraph} style={{backgroundColor:'#3b82f6',color:'white'}}>Load Random Example Graph</button>
                <button className="sample-data-button" onClick={() => {setNodes([]); setEdges([]); setResults(null);}} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
              </div>
            </div>

            <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>Click on the canvas below to add nodes and drag to add edges.</p>

            {/* NODE LEGEND (Moved directly above canvas) */}
            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: 'fit-content', marginBottom: '0.5rem', textAlign: 'left' }}>
                <h4 style={{ margin: '0 0 0.5rem 0' }}>Node Legend</h4>
                <div style={{ display: 'flex', gap: '1rem', fontSize: '0.9rem', flexWrap: 'wrap' }}>
                    <span><strong style={{color: '#3b82f6'}}>Blue:</strong> Labeled Class 0</span>
                    <span><strong style={{color: '#ef4444'}}>Red:</strong> Labeled Class 1</span>
                    <span><strong style={{color: '#9ca3af'}}>Gray:</strong> Unlabeled (AI guesses)</span>
                    <span><strong style={{color: '#93c5fd'}}>Light Blue/Red:</strong> Final Prediction</span>
                </div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
                    <button onClick={() => setInteractionMode('node')} style={{ padding: '0.75rem', backgroundColor: interactionMode === 'node' ? '#3b82f6' : '#e5e7eb', color: interactionMode === 'node' ? 'white' : '#4b5563', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>1. Add Nodes</button>
                    <button onClick={() => setInteractionMode('edge')} style={{ padding: '0.75rem', backgroundColor: interactionMode === 'edge' ? '#3b82f6' : '#e5e7eb', color: interactionMode === 'edge' ? 'white' : '#4b5563', border: 'none', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>2. Draw Edges</button>
                </div>
                {interactionMode === 'node' && (
                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <button onClick={() => setBrushClass(0)} style={{ padding: '0.5rem', backgroundColor: brushClass === 0 ? '#3b82f6' : '#eff6ff', color: brushClass === 0 ? 'white' : '#1e40af', border: '1px solid #bfdbfe', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Blue (Class 0)</button>
                        <button onClick={() => setBrushClass(1)} style={{ padding: '0.5rem', backgroundColor: brushClass === 1 ? '#ef4444' : '#fef2f2', color: brushClass === 1 ? 'white' : '#b91c1c', border: '1px solid #fecaca', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Red (Class 1)</button>
                        <button onClick={() => setBrushClass(-1)} style={{ padding: '0.5rem', backgroundColor: brushClass === -1 ? '#4b5563' : '#f3f4f6', color: brushClass === -1 ? 'white' : '#1f2937', border: '1px solid #d1d5db', borderRadius: '4px', cursor: 'pointer', flex: 1, fontWeight: 'bold' }}>Gray (Unlabeled)</button>
                    </div>
                )}
                {interactionMode === 'edge' && <p style={{fontSize: '0.85rem', color: '#6b7280', margin: 0, textAlign: 'center'}}>Click and drag from one node to another to connect them!</p>}
            </div>

            <div style={{ border: '1px solid #e5e7eb', borderRadius: '0.75rem', overflow: 'hidden', position: 'relative', backgroundColor: '#f9fafb', width: '100%', paddingBottom: '100%' }}>
              <canvas ref={canvasRef} width={600} height={600} onMouseDown={handleCanvasDown} onMouseUp={handleCanvasUp} style={{ position: 'absolute', top: 0, left: 0, display: 'block', cursor: interactionMode === 'edge' ? 'crosshair' : 'pointer', width: '100%', height: '100%' }} />
            </div>
          </div>

          {/* RIGHT COLUMN: Controls */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Message Passing Controls</h2>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Training Epochs: {epochs}</h3>
              <input type="range" min="10" max="300" step="10" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}>
                  <span>10 (Weak message passing)</span><span>300 (Strong message passing)</span>
              </div>

              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Learning Rate: {learningRate.toFixed(2)}</h3>
              <input type="range" min="0.01" max="0.5" step="0.01" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem' }}>
                  <span>0.01 (Slow & Stable)</span><span>0.50 (Fast & Erratic)</span>
              </div>

              <div style={{ padding: '1rem', backgroundColor: '#faf5ff', borderRadius: '6px', border: '1px dashed #d8b4fe' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#6b21a8', fontSize: '1rem' }}>The GCN Formula:</h4>
                  <div style={{ fontFamily: 'monospace', fontSize: '1.1rem', backgroundColor: '#f3e8ff', padding: '10px', borderRadius: '4px', textAlign: 'center', marginBottom: '10px' }}>
                    H' = σ(D⁻¹² * A * D⁻¹² * X * W)
                  </div>
                  <p style={{ margin: 0, fontSize: '0.85rem', color: '#7e22ce', lineHeight: '1.5' }}>
                      <strong>A:</strong> Adjacency Matrix (Who is connected to whom).<br/>
                      <strong>X:</strong> The Node Feature Matrix (The Node's own data).<br/>
                      <strong>W:</strong> The Weights Matrix (The brain the AI is learning).<br/>
                      This formula mathematically blends a node's features with the features of its direct neighbors!
                  </p>
              </div>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <button onClick={trainModel} disabled={loading || nodes.length < 3 || edges.length === 0} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}>
                  {loading ? (
                      <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Passing Messages...</>
                  ) : 'Run Graph Convolution'}
              </button>
            </div>

            {/* Why no prediction box */}
            <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '6px', border: '1px solid #bbf7d0', marginBottom: '1rem' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#166534' }}>Why are there no Predictions?</h4>
                <p style={{ fontSize: '0.85rem', color: '#15803d', margin: 0, lineHeight: '1.5' }}>
                    GCN is a "transductive" method. It doesn't learn a general rule for new, unseen nodes. Instead, it uses the few labeled nodes to figure out the labels of the <strong>unlabeled nodes already in the graph!</strong> Notice the gray dots on the left magically get colored in after you hit train!
                </p>
            </div>
          </div>
        </div>

        {/* BOTTOM ROWS: THE 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0 }}>GCN Latent Embeddings</h4>
                            <span style={{ backgroundColor: '#eff6ff', color: '#3b82f6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>Epoch: {playback.frames[playback.currentIndex]?.epoch}</span>
                        </div>
                        <img src={`data:image/png;base64,${playback.frames[playback.currentIndex]?.image}`} alt="Embeddings" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}/>
                        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                            <button onClick={togglePlayback} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', width: '40px', height: '40px', cursor: 'pointer' }}>{playback.isPlaying ? '⏸' : '▶'}</button>
                            <input type="range" min="0" max={playback.frames.length - 1} value={playback.currentIndex} onChange={(e) => setPlayback(p => ({ ...p, currentIndex: parseInt(e.target.value), isPlaying: false }))} style={{ flex: 1 }} />
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', margin: 0 }}>Final Loss: {results.final_loss.toFixed(4)}</p>
                        </div>
                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                Look at the Left animation! Because nodes passed messages to their connected neighbors, the GCN mathematically pulled connected nodes together into the same physical space.<br/><br/>
                                Look at the Main Canvas! The AI successfully guessed the colors of the gray nodes based on who they were connected to.
                            </p>
                            <div style={{ display: 'flex', flexDirection: 'column' }}>
                                <h4 style={{ marginBottom: '1rem', fontWeight: '600', marginTop: '1.5rem' }}>Graph Topology Stats</h4>
                                <div style={{ backgroundColor: '#f8fafc', padding: '1.5rem', borderRadius: '8px', border: '1px dashed #cbd5e1', height: '100%' }}>
                                    <p><strong>Total Nodes:</strong> {nodes.length}</p>
                                    <p><strong>Total Edges:</strong> {edges.length}</p>
                                    <p><strong>Graph Density:</strong> {((2 * edges.length) / (nodes.length * (nodes.length - 1))).toFixed(2)}</p>
                                    <p><strong>Labeled Ratio:</strong> {((nodes.filter(n => n.class !== -1).length / nodes.length) * 100).toFixed(0)}%</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gridColumn: '1 / 3' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Cross-Entropy Loss Curve</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <LineChart data={results.loss_history}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="epoch" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line type="monotone" dataKey="loss" stroke="#ef4444" name="Classification Loss" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default GCN;
