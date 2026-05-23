/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';

function Transformer() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");

  // Sandbox State
  const[prompt, setPrompt] = useState("the quick brown fox");

  // Parameters
  const [temperature, setTemperature] = useState(1.0);
  const [causalMask, setCausalMask] = useState(true);
  const [results, setResults] = useState(null);

  // Streaming State
  const[isStreaming, setIsStreaming] = useState(false);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  useEffect(() => {
    axios.get(`${apiUrl}/health`).then(() => setBackendStatus("connected")).catch(() => setBackendStatus("disconnected"));
  }, [apiUrl]);

  // AUTO-REGRESSIVE GENERATION LOOP
  useEffect(() => {
    if (isStreaming && results && results.sampled_word) {
      // Add a slight delay so the user can watch the matrix expand!
      const timer = setTimeout(() => {
        const words = prompt.trim().split(/\s+/);
        if (words.length >= 20) {
          setIsStreaming(false); // Stop if text gets too long
          return;
        }

        const newPrompt = prompt.trim() + " " + results.sampled_word;
        setPrompt(newPrompt);
        runGeneration(newPrompt); // Immediately calculate the next word
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, [isStreaming, results]);

  const handleGenerateSingle = () => {
    setIsStreaming(false);
    runGeneration(prompt);
  };

  const handleStartStream = () => {
    setIsStreaming(true);
    runGeneration(prompt);
  };

  const handleStopStream = () => {
    setIsStreaming(false);
  };

  const runGeneration = (currentPrompt) => {
    if (!currentPrompt.trim()) {
      setError("Prompt cannot be empty.");
      setIsStreaming(false);
      return;
    }

    setLoading(true); setError(null);

    axios.post(`${apiUrl}/transformer/generate`, {
      prompt: currentPrompt,
      parameters: { temperature: temperature, causal_mask: causalMask }
    })
    .then(response => {
        if (response.data.error) {
            setError(response.data.error);
            setIsStreaming(false);
            return;
        }
        setResults(response.data);
    })
    .catch(err => {
        setError("Failed to run Transformer math.");
        setIsStreaming(false);
    })
    .finally(() => setLoading(false));
  };

  const handleAppendWord = (word) => {
    setPrompt(prev => prev + " " + word);
    setResults(null);
  };

  // Custom Tooltip for the Scatter Plot
  const CustomScatterTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{ backgroundColor: 'white', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}>
          <p style={{ fontWeight: 'bold', margin: 0, color: '#3b82f6' }}>"{data.word}"</p>
          <p style={{ margin: 0, fontSize: '0.8rem', color: '#6b7280' }}>Category: {data.category}</p>
          <p style={{ margin: 0, fontSize: '0.8rem', color: '#6b7280' }}>[X: {data.x.toFixed(2)}, Y: {data.y.toFixed(2)}]</p>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Hub</span>
        </button>
        <h1 className="model-title">Transformers (LLMs) </h1>
      </div>

      <p className="model-description">
        Transformers are the mathematical engine behind Large Language Models (LLMs) like ChatGPT. Instead of reading words one-by-one, they use a <strong>Self-Attention Mechanism</strong> to look at the entire sentence at once, calculating the relationships between every single word simultaneously!
        <InfoButton algoId="transformer" />
      </p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          {/* LEFT: Prompt Sandbox */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header">
              <h2 className="section-title">The Prompt Sandbox</h2>
              <button className="sample-data-button" onClick={() => {setPrompt(""); setIsStreaming(false); setResults(null);}} style={{ backgroundColor: '#fee2e2', color: '#b91c1c' }}>Clear</button>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', width: '100%', marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>Enter a sequence (max 20 words):</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Type a sentence here..."
                style={{ width: '100%', height: '100px', padding: '1rem', fontSize: '1.2rem', borderRadius: '6px', border: '2px solid #bfdbfe', resize: 'none', fontFamily: 'monospace' }}
              />
              <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.85rem', color: '#6b7280', display: 'flex', alignItems: 'center', marginRight: '0.5rem' }}>Try adding:</span>
                {['cat', 'jumps', 'over', 'lazy', 'dog'].map(w => (
                    <button key={w} onClick={() => handleAppendWord(w)} disabled={isStreaming} style={{ padding: '4px 10px', backgroundColor: '#f3f4f6', border: '1px solid #d1d5db', borderRadius: '4px', cursor: isStreaming ? 'not-allowed' : 'pointer', fontSize: '0.85rem' }}>{w}</button>
                ))}
              </div>
            </div>

            <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#166534' }}>How to play with this:</h4>
                <p style={{ fontSize: '0.9rem', color: '#15803d', margin: 0 }}>
                    Type a sentence and hit <strong>Auto-Generate Stream</strong>. The AI will constantly predict the next word, add it to your sentence, and mathematically expand the Self-Attention Matrix in real-time!
                </p>
            </div>
          </div>

          {/* RIGHT: Controls */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Generation Controls</h2>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>

              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Temperature: {temperature.toFixed(2)}</h3>
              <input type="range" min="0.1" max="2.0" step="0.1" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} disabled={isStreaming} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem', marginTop: '0.5rem' }}>
                  <span>0.1 (Robotic, Predictable)</span><span>2.0 (Creative, Chaotic)</span>
              </div>

              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontWeight: '500', color: '#4b5563' }}>
                    <input type="checkbox" checked={causalMask} onChange={(e) => setCausalMask(e.target.checked)} disabled={isStreaming} style={{ marginRight: '0.5rem', width: '18px', height: '18px' }} />
                    Apply Causal Mask (GPT Style)
                </label>
                <p style={{fontSize: '0.85rem', color: '#6b7280', marginTop: '0.5rem', marginLeft: '1.5rem'}}>
                    If checked, words can only pay attention to the words <i>before</i> them. If unchecked (BERT style), words can look at the future!
                </p>
              </div>

            </div>

            <div style={{ display: 'flex', gap: '1rem' }}>
                <button onClick={handleGenerateSingle} disabled={loading || isStreaming} style={{ flex: 1, backgroundColor: 'white', color: '#8b5cf6', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: '2px solid #8b5cf6', borderRadius: '6px', cursor: (loading || isStreaming) ? 'not-allowed' : 'pointer' }}>
                  Generate 1 Token
                </button>
                {isStreaming ? (
                    <button onClick={handleStopStream} style={{ flex: 2, backgroundColor: '#ef4444', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 4px 6px rgba(239, 68, 68, 0.2)' }}>
                      Stop Streaming 🛑
                    </button>
                ) : (
                    <button onClick={handleStartStream} disabled={loading} style={{ flex: 2, backgroundColor: '#8b5cf6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: loading ? 'wait' : 'pointer', boxShadow: '0 4px 6px rgba(139, 92, 246, 0.2)' }}>
                      Auto-Generate Stream
                    </button>
                )}
            </div>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>Transformer Analytics</h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Self-Attention Matrix */}
                    <div style={{ padding: '1.5rem', border: '2px solid #8b5cf6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h4 style={{ fontWeight: '600', margin: 0, color: '#1f2937' }}>Self-Attention Matrix</h4>
                            <span style={{ backgroundColor: '#ede9fe', color: '#8b5cf6', padding: '4px 10px', borderRadius: '12px', fontWeight: '600', fontSize: '0.9rem' }}>
                                Head 1 | {isStreaming ? "Streaming..." : "Idle"}
                            </span>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                            <img src={`data:image/png;base64,${results.attention_heatmap}`} alt="Attention Heatmap" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e5e7eb' }}/>
                        </div>
                        <p style={{fontSize: '0.85rem', color: '#4b5563', textAlign: 'center'}}>
                            The darker the blue, the stronger the mathematical connection between the two words. Notice the upper right is white because the <strong>Causal Mask</strong> prevents looking into the future!
                        </p>
                    </div>

                    {/* (0,1) Next Token Probabilities */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                            Next Token Prediction (Top 5)
                        </h4>

                        <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '8px', border: '1px solid #bbf7d0', textAlign: 'center', marginBottom: '1rem' }}>
                            <p style={{ fontSize: '1.1rem', fontWeight: '600', color: '#166534', margin: 0 }}>
                                Appending: "{results.sampled_word}"
                            </p>
                        </div>

                        <div style={{ height: 230, width: '100%' }}>
                            <ResponsiveContainer>
                                <BarChart data={results.predictions} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                    <XAxis type="number" domain={[0, 100]} />
                                    <YAxis dataKey="word" type="category" width={80} fontWeight="bold" />
                                    <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                                    <Bar dataKey="probability" name="Probability %" fill="#10b981" radius={[0, 4, 4, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* NEW DETAILED EXPLANATION BLOCK */}
                        <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e5e7eb', marginTop: '1rem' }}>
                            <p style={{fontSize: '0.85rem', color: '#4b5563', margin: '0 0 0.75rem 0', lineHeight: '1.4'}}>
                                <strong>Why did it pick a word not in the Top 5?</strong> Due to <em>Temperature Sampling</em>, the AI rolls a loaded dice. A word with a 2% probability will still be chosen 2 out of 100 times! Higher temperature = more randomness.
                            </p>
                            <p style={{fontSize: '0.85rem', color: '#4b5563', margin: 0, lineHeight: '1.4'}}>
                                <strong>Why is the sentence gibberish?</strong> This is an untrained "Toy Model" with random weights. Real LLMs like ChatGPT use this exact same math, but with billions of weights trained on supercomputers so the outputs make sense!
                            </p>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginTop: '1rem' }}>
                            <button onClick={() => handleAppendWord(results.predictions[0].word)} disabled={isStreaming} style={{ padding: '8px 16px', backgroundColor: '#10b981', color: 'white', border: 'none', borderRadius: '4px', cursor: isStreaming ? 'not-allowed' : 'pointer', fontWeight: 'bold' }}>
                                Force Top Prediction
                            </button>
                        </div>
                    </div>

                    {/* (1,0) The Embedding Space */}
                    {results.embedding_data && (
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <h4 style={{ marginBottom: '1rem', fontWeight: '600', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                                The Latent Embedding Space (AI Dictionary)
                            </h4>
                            <div style={{ height: 250, width: '100%', marginTop: '1rem', border: '1px solid #e5e7eb', borderRadius: '4px', backgroundColor: '#f8fafc' }}>
                                <ResponsiveContainer>
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis type="number" dataKey="x" name="Dimension 1" domain={[-1.2, 1.2]} tick={false} />
                                        <YAxis type="number" dataKey="y" name="Dimension 2" domain={[-1.2, 1.2]} tick={false} />
                                        <ZAxis type="number" range={[100, 100]} />
                                        <Tooltip content={<CustomScatterTooltip />} cursor={{strokeDasharray: '3 3'}} />

                                        <Scatter name="Animals" data={results.embedding_data.filter(d => d.category === 'Animal')} fill="#3b82f6" />
                                        <Scatter name="Actions" data={results.embedding_data.filter(d => d.category === 'Action')} fill="#ef4444" />
                                        <Scatter name="Descriptors" data={results.embedding_data.filter(d => d.category === 'Descriptor')} fill="#10b981" />
                                        <Scatter name="Stop Words" data={results.embedding_data.filter(d => d.category === 'Stop Word')} fill="#9ca3af" />

                                        <Legend />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </div>
                            <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                                Hover over the dots! The AI doesn't read English. It maps words into a physical space where similar concepts group together.
                            </p>
                        </div>
                    )}

                    {/* (1,1) The Math Pipeline */}
                    <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                        <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0', color: '#1f2937', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem' }}>The Transformer Pipeline</h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            <div style={{ padding: '0.5rem', backgroundColor: '#eff6ff', borderRadius: '4px', borderLeft: '4px solid #3b82f6' }}>
                                <strong>1. Tokenization:</strong> Split prompt into {results.tokens.length} words.
                            </div>
                            <div style={{ padding: '0.5rem', backgroundColor: '#f8fafc', borderRadius: '4px', borderLeft: '4px solid #94a3b8' }}>
                                <strong>2. Embedding:</strong> Look up the X/Y coordinates for each word in the Embedding Space (Left Graph).
                            </div>
                            <div style={{ padding: '0.5rem', backgroundColor: '#fef2f2', borderRadius: '4px', borderLeft: '4px solid #ef4444' }}>
                                <strong>3. Positional Encoding:</strong> Add Sine/Cosine waves so the AI knows word order.
                            </div>
                            <div style={{ padding: '0.5rem', backgroundColor: '#fdf4ff', borderRadius: '4px', borderLeft: '4px solid #d946ef' }}>
                                <strong>4. Q, K, V Matrices:</strong> Project embeddings into Queries, Keys, and Values.
                            </div>
                            <div style={{ padding: '0.5rem', backgroundColor: '#faf5ff', borderRadius: '4px', borderLeft: '4px solid #a855f7' }}>
                                <strong>5. Self-Attention:</strong> Multiply Queries by Keys to get the Heatmap (Top Left Graph).
                            </div>
                            <div style={{ padding: '0.5rem', backgroundColor: '#f0fdf4', borderRadius: '4px', borderLeft: '4px solid #10b981' }}>
                                <strong>6. Softmax Prediction:</strong> Output probabilities for the next word (Top Right Graph) modified by Temperature!
                            </div>
                        </div>
                    </div>
                </div>
            </motion.div>
        )}
      </div>

    </motion.div>
  );
}

export default Transformer;
