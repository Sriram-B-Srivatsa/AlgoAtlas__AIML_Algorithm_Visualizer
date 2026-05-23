import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import './HomePage.css';

function HomePage() {
  const navigate = useNavigate();
  const [activeCategory, setActiveCategory] = useState(null);

  // NEW: Auto-scroll a little bit when a category opens
  useEffect(() => {
    if (activeCategory) {
      // Wait 100ms for the Framer Motion animation to render the cards, then scroll down 200 pixels
      setTimeout(() => {
        window.scrollBy({ top: 150, behavior: 'smooth' });
      }, 100);
    } else {
      // Smoothly scroll back to the absolute top if they hit "Back to Categories"
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [activeCategory]);

  // --- 1. DEFINE THE MASTER CATEGORIES ---
  const categories =[
    {
      id: 'supervised',
      name: 'Supervised Learning',
      description: 'The model learns from labeled data to predict outcomes for new, unseen data (Regression & Classification).',
      iconClass: 'cat-supervised',
      icon: '🎯'
    },
    {
      id: 'unsupervised',
      name: 'Unsupervised Learning',
      description: 'The model identifies hidden patterns, structures, or groups in completely unlabeled data.',
      iconClass: 'cat-unsupervised',
      icon: '🔍'
    },
    {
      id: 'ensemble',
      name: 'Ensemble Learning',
      description: 'Combines multiple weak algorithms (like simple trees) to create one highly accurate, robust model.',
      iconClass: 'cat-ensemble',
      icon: '🌳'
    },
    {
      id: 'dl',
      name: 'Deep Learning',
      description: 'Advanced Neural Network architectures designed to handle complex data like images, text, and sequences.',
      iconClass: 'cat-dl',
      icon: '🧠'
    },
    {
      id: 'rl',
      name: 'Reinforcement Learning',
      description: 'An AI Agent learns by interacting with an environment, earning rewards or penalties to find the optimal strategy.',
      iconClass: 'cat-rl',
      icon: '🎮'
    },
    {
      id: 'semi',
      name: 'Semi/Self-Supervised',
      description: 'Hybrid approaches used when you have massive amounts of data, but only a tiny fraction of it is labeled.',
      iconClass: 'cat-semi',
      icon: '⚖️'
    }
  ];

  // --- 2. DEFINE THE ALGORITHMS INSIDE EACH CATEGORY ---
  const algorithms = {
    supervised:[
      { name: "Polynomial Regression", desc: "Fit polynomial curves to predict continuous values.", path: "/reg", status: "active" },
      { name: "Logistic Regression", desc: "Predict categorical probabilities using the Sigmoid curve.", path: "/logreg", status: "active" },
      { name: "Ridge & Lasso Regularization", desc: "Shrink coefficients to prevent complex models from overfitting.", path: "/regularization", status: "active" },
      { name: "Naive Bayes", desc: "Fast probabilistic classifier based on Bayes' Theorem.", path: "/nb", status: "active" },
      { name: "K-Nearest Neighbors", desc: "Classify points based on the 'K' closest training examples.", path: "/knn", status: "active" },
      { name: "Decision Trees", desc: "Make decisions based on hierarchical feature splits.", path: "/d-trees", status: "active" },
      { name: "Support Vector Machines", desc: "Find the optimal mathematical hyperplane to separate classes.", path: "/svm", status: "active" }
    ],
    unsupervised:[
      { name: "K-Means Clustering", desc: "Automatically group data into K distinct clusters.", path: "/kmeans", status: "active" },
      { name: "Hierarchical Clustering", desc: "Builds a tree of clusters by continuously merging the closest points.", path: "/hierarchical", status: "active" },
      { name: "t-SNE", desc: "Mathematically unrolls complex 3D data into a flat 2D map.", path: "/tsne", status: "active" },
      { name: "DBSCAN", desc: "Density-based clustering that identifies core groups and isolates noise.", path: "/DBScan", status: "active" },
      { name: "Principal Component Analysis", desc: "Reduce dimensionality by extracting maximum variance.", path: "/pca", status: "active" },
      { name: "Gaussian Mixture Models", desc: "Probabilistic model assuming data is generated from a mix of Gaussian distributions.", path: "/gmm", status: "active" },
      { name: "Apriori (Market Basket)", desc: "Finds hidden association rules in databases.", path: "/apriori", status: "active" }
    ],
    ensemble:[
      { name: "Random Forest", desc: "A robust ensemble of decision trees voting together to prevent overfitting.", path: "/rf", status: "active" },
      { name: "AdaBoost", desc: "Sequential weak learners that focus specifically on previous mistakes.", path: "/adaboost", status: "active" },
      { name: "Gradient Boosting", desc: "Optimizes loss functions sequentially (XGBoost, LightGBM style).", path: "/gb", status: "active" }
    ],
    dl:[
      { name: "Artificial Neural Networks (ANN)", desc: "Standard multi-layer perceptron for deep mathematical mapping.", path: "/ann", status: "active" },
      { name: "Convolutional NNs (CNN)", desc: "Feature extraction filters designed specifically for image recognition.", path: "/cnn", status: "active" },
      { name: "Recurrent NNs (RNN / LSTM)", desc: "Networks with internal memory to process sequences and time-series data.", path: "/rnn", status: "active" },
      { name: "Transformers", desc: "Attention-based mechanisms that revolutionized NLP.", path: "/transformer", status: "active" },
      { name: "Autoencoders", desc: "Compresses data into a bottleneck and reconstructs it.", path: "/ae", status: "active" },
      { name: "Generative Adversarial Networks", desc: "Two networks fighting: one makes fake data, one catches it.", path: "/gan", status: "active" },
      { name: "Graph Convolutional Networks", desc: "Deep learning on interconnected nodes and edges.", path: "/gcn", status: "active" },
      { name: "Diffusion Models", desc: "The generative math behind Midjourney and DALL-E.", path: "/diffusion", status: "active" }
    ],
    rl:[
      { name: "Q-Learning", desc: "Model-free reinforcement learning used to find the optimal action-selection policy.", path: "/ql", status: "active" },
      { name: "Deep Q-Networks (DQN)", desc: "Uses deep learning to approximate Q-value functions for complex environments.", path: "/dqn", status: "active" }
    ],
    semi:[
      { name: "Pseudo-Labeling", desc: "Train a model on labeled data, then use it to guess labels for the rest.", path: "/pl", status: "active" },
      { name: "Contrastive Learning", desc: "Self-supervised technique teaching models to group similar inputs together.", path: "/cl", status: "active" }
    ]
  };

  const handleCardClick = (path, status) => {
    if (status === 'active') {
      navigate(path);
    }
  };

  return (
    <div className="homepage">
      <motion.div
        className="hero-section"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="homepage-title">Machine Learning Hub</h1>
        <p className="homepage-subtitle">
          {activeCategory
            ? categories.find(c => c.id === activeCategory).description
            : "Explore interactive algorithms, visualize the math, and understand how AI thinks."}
        </p>
      </motion.div>

      <AnimatePresence mode="wait">
        {/* VIEW 1: THE CATEGORIES HUB */}
        {!activeCategory && (
          <motion.div
            key="categories"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="model-grid"
          >
            {categories.map((cat) => (
              <div
                key={cat.id}
                className="model-card"
                onClick={() => { setActiveCategory(cat.id); window.scrollTo({ top: 0, behavior: 'smooth' }); }}              >
                <div className={`card-icon-wrapper ${cat.iconClass}`}>
                  {cat.icon}
                </div>
                <h3 className="model-name">{cat.name}</h3>
                <p className="model-description">{cat.description}</p>
                <div className="card-cta">
                  <span>Explore Category &rarr;</span>
                  <span style={{ color: '#94a3b8', fontSize: '0.85rem', fontWeight: '500' }}>{algorithms[cat.id].length} Algorithms</span>
                </div>
              </div>
            ))}
          </motion.div>
        )}

        {/* VIEW 2: THE ALGORITHMS SPOKE */}
        {activeCategory && (
          <motion.div
            key="algorithms"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
          >
            <button
              className="back-to-categories"
                onClick={() => { setActiveCategory(null);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
            >
              &larr; Back to Categories
            </button>

            <div className="model-grid">
              {algorithms[activeCategory].map((algo, index) => (
                <div
                  key={index}
                  className={`model-card ${algo.status === 'coming_soon' ? 'disabled' : ''}`}
                  onClick={() => handleCardClick(algo.path, algo.status)}
                >
                  <div className={`card-icon-wrapper ${categories.find(c => c.id === activeCategory).iconClass}`}>
                    {categories.find(c => c.id === activeCategory).icon}
                  </div>
                  <h3 className="model-name">{algo.name}</h3>
                  <p className="model-description">{algo.desc}</p>
                  <div className="card-cta">
                    {algo.status === 'active' ? (
                      <span>Launch Visualizer &rarr;</span>
                    ) : (
                      <span className="coming-soon">Coming Soon</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default HomePage;
