import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import '../ModelPage.css';
import InfoButton from '../../components/InfoButton';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function Apriori() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [transactions, setTransactions] = useState([]);
  const [currentCart, setCurrentCart] = useState([]);

  const [minSupport, setMinSupport] = useState(0.2);
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [results, setResults] = useState(null);

  const availableItems = ["🍞 Bread", "🥛 Milk", "🥚 Eggs", "🍎 Apples", "🍺 Beer", "🍼 Diapers"];
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

  const addToCart = (item) => {
      if (!currentCart.includes(item)) setCurrentCart([...currentCart, item]);
  };

  const checkout = () => {
      if (currentCart.length > 0) {
          setTransactions([...transactions, currentCart]);
          setCurrentCart([]);
          setResults(null);
      }
  };

  const loadExampleDatabase = () => {
      const items = ["🍞 Bread", "🥛 Milk", "🥚 Eggs", "🍎 Apples", "🍺 Beer", "🍼 Diapers", "🥩 Meat", "🧀 Cheese"];
      const numTx = Math.floor(Math.random() * 6) + 5; // 5 to 10 receipts
      const newTx =[];
      for(let i=0; i<numTx; i++) {
        const cartSize = Math.floor(Math.random() * 4) + 2;
        const cart =[];
        while(cart.length < cartSize) {
           const randItem = items[Math.floor(Math.random() * items.length)];
           if(!cart.includes(randItem)) cart.push(randItem);
        }
        // Artificial correlation: If Beer, likely Diapers
        if (cart.includes("🍺 Beer") && Math.random() > 0.3 && !cart.includes("🍼 Diapers")) cart.push("🍼 Diapers");
        newTx.push(cart);
      }
      setTransactions(newTx);
      setResults(null);
  };

  const runApriori = () => {
      if (transactions.length < 3) { setError("Need at least 3 transactions!"); return; }
      setLoading(true); setError(null);
      axios.post(`${apiUrl}/apriori/train`, { transactions, parameters: { minSupport, minConfidence } })
      .then(res => {
          if (res.data.error) setError(res.data.error);
          else setResults(res.data);
      })
      .catch(err => setError("Failed to run Apriori."))
      .finally(() => setLoading(false));
  };

  return (
    <motion.div className="model-page" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}><span>&larr; Back to Hub</span></button>
        <h1 className="model-title">Apriori (Association Rules) </h1>
      </div>

      <p className="model-description">Apriori analyzes massive databases of shopping carts to find hidden patterns. It calculates the exact probability that "If a customer buys X, they will also buy Y."<InfoButton algoId="apriori" /></p>

      {error && <div className="error-message"><span>{error}</span></div>}

      <div className="content-container" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>

          {/* LEFT: Grocery Store Simulator */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="section-header" style={{ display: 'flex', gap: '0.25rem' }}>
              <h2 className="section-title">Grocery Store Simulator</h2>
              <button className="sample-data-button" onClick={loadExampleDatabase} style={{backgroundColor:'#3b82f6', color:'white'}}>Load Random Store Data</button>
              <button className="sample-data-button" onClick={() => setTransactions([])} style={{ backgroundColor: '#fee2e2', color: '#b91c1c'}}>Clear DB</button>
            </div>

            <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>Make atleast 3 Receipts to be able to start the algorithm or generate random receipts</p>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1rem' }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1rem' }}>1. Build a Shopping Cart</h3>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
                    {availableItems.map(item => (
                        <button key={item} onClick={() => addToCart(item)} style={{ padding: '8px 12px', backgroundColor: currentCart.includes(item) ? '#3b82f6' : '#f3f4f6', color: currentCart.includes(item) ? 'white' : 'black', border: '1px solid #d1d5db', borderRadius: '20px', cursor: 'pointer', fontWeight: 'bold' }}>
                            {item}
                        </button>
                    ))}
                </div>
                <button onClick={checkout} style={{ width: '100%', padding: '10px', backgroundColor: '#10b981', color: 'white', fontWeight: 'bold', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Checkout (Save Transaction)</button>
            </div>

            <div style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb', height: '300px', overflowY: 'auto' }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1rem' }}>2. Store Database ({transactions.length} Receipts)</h3>
                {transactions.map((tx, idx) => (
                    <div key={idx} style={{ padding: '8px', backgroundColor: '#f8fafc', border: '1px dashed #cbd5e1', marginBottom: '8px', borderRadius: '4px', fontSize: '0.9rem' }}>
                        <strong>Receipt #{idx+1}:</strong> {tx.join(", ")}
                    </div>
                ))}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-title">Algorithm Controls</h2>

            <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Minimum Support: {(minSupport*100).toFixed(0)}%</h3>
              <input type="range" min="0.1" max="1.0" step="0.1" value={minSupport} onChange={(e) => setMinSupport(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>How frequently an item must be bought overall to be considered.</p>

              <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Minimum Confidence: {(minConfidence*100).toFixed(0)}%</h3>
              <input type="range" min="0.1" max="1.0" step="0.1" value={minConfidence} onChange={(e) => setMinConfidence(parseFloat(e.target.value))} style={{ width: '100%', marginBottom: '0.5rem' }} />
              <p style={{fontSize: '0.85rem', color: '#6b7280', marginBottom: '1.5rem'}}>If they buy X, how likely is it they ALSO buy Y?</p>
            </div>

            <button onClick={runApriori} disabled={loading || transactions.length < 3} style={{ width: '100%', backgroundColor: '#3b82f6', color: 'white', padding: '1rem', fontSize: '1.1rem', fontWeight: 'bold', border: 'none', borderRadius: '6px', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}>
                {loading ? (
                    <><svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeDasharray="32" strokeDashoffset="16" strokeLinecap="round" opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"></path></svg> Analyzing Receipts...</>
                ) : 'Run Apriori Rule Miner'}
            </button>
          </div>
        </div>

        {/* 2x2 RESULTS GRID */}
        {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '6px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: '600', marginTop: 0 }}>Apriori Results Dashboard</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '2rem' }}>

                    {/* (0,0) Discovered Rules Table */}
                    <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '8px', backgroundColor: '#f9fafb' }}>
                        <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Discovered Rules ({results.rules.length} found)</h4>
                        <div style={{ height: '250px', overflowY: 'auto' }}>
                            {results.rules.length === 0 ? <p>No rules found. Try lowering the Support/Confidence sliders!</p> :
                             results.rules.map((r, i) => (
                                <div key={i} style={{ padding: '10px', backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '4px', marginBottom: '8px' }}>
                                    <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#1e40af', marginBottom: '4px' }}>
                                        {r.antecedent} &rarr; {r.consequent}
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#4b5563' }}>
                                        <span>Support: {(r.support*100).toFixed(0)}%</span>
                                        <span>Conf: {(r.confidence*100).toFixed(0)}%</span>
                                        <span style={{color: r.lift > 1 ? '#10b981' : '#ef4444'}}>Lift: {r.lift.toFixed(2)}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* (0,1) Top 5 Rules Bar Chart */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Top 5 Rules (Support vs Conf)</h4>
                        <div style={{ height: 250, width: '100%' }}>
                            <ResponsiveContainer>
                                <BarChart data={results.rules.slice(0,5).map(r => ({name: `${r.antecedent} \u2192 ${r.consequent}`, Support: r.support*100, Confidence: r.confidence*100}))}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="name" tick={{fontSize: 10}} interval={0} angle={-15} textAnchor="end"/>
                                    <YAxis domain={[0, 100]} />
                                    <Tooltip formatter={(val) => `${val.toFixed(1)}%`} />
                                    <Legend />
                                    <Bar dataKey="Support" fill="#3b82f6" />
                                    <Bar dataKey="Confidence" fill="#10b981" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* (1,0) Scatter Plot */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ marginBottom: '1rem', fontWeight: '600' }}>Confidence vs Support (Colored by Lift)</h4>
                        <div style={{ backgroundColor: '#f3f4f6', padding: '1rem', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
                            <img src={`data:image/png;base64,${results.scatter_plot}`} alt="Scatter" style={{ width: '100%', borderRadius: '4px' }}/>
                        </div>
                        <p style={{fontSize: '0.8rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem'}}>
                            *Dots are slightly "jittered" (scattered) so you can see rules that have the exact same math scores!
                        </p>
                    </div>

                    {/* (1,1) Insights & Why No Predictions */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ flex: 1, padding: '1.5rem', border: '1px solid #e5e7eb', borderRadius: '8px', backgroundColor: '#ffffff' }}>
                            <h4 style={{ fontWeight: '600', margin: '0 0 1rem 0' }}>Mathematical Insights</h4>
                            <p style={{ color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                <strong>Lift &gt; 1</strong> means the items are magically connected (e.g. Diapers and Beer). Buying Diapers actively *increases* the chance of buying Beer!<br/><br/>
                                <strong>Lift = 1</strong> means they are completely independent. Buying bread has zero effect on buying milk.<br/><br/>
                                <strong>Lift &lt; 1</strong> means the items are unlikely to be bought together, and the presence of one item actually reduces the likelihood of the other item being purchased.
                            </p>
                        </div>
                        <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                            <h4 style={{ margin: '0 0 0.5rem 0', color: '#166534' }}>Why are there no Predictions?</h4>
                            <p style={{ fontSize: '0.85rem', color: '#15803d', margin: 0, lineHeight: '1.5' }}>
                                Apriori is Unsupervised. It doesn't "predict" new points. Instead, it builds <strong>Rules</strong> from past data. You use these rules to build Recommendation Engines (like Amazon's "Frequently bought together").
                            </p>
                        </div>
                    </div>
                </div>
            </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default Apriori;
