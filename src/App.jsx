import React, {useEffect} from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import HomePage from './pages/HomePage';
import About from './pages/About';
import Docs from './pages/Docs';

// Supervised
import Reg from './pages/supervised/Reg';
import KNN from './pages/supervised/KNN';
import DTrees from './pages/supervised/DTrees';
import SVM from './pages/supervised/SVM';
import LogisticRegression from './pages/supervised/LogisticRegression';
import Regularization from './pages/supervised/Regularization';
import NaiveBayes from './pages/supervised/NaiveBayes';

// Unsupervised
import KMeans from './pages/unsupervised/KMeans';
import DBScan from './pages/unsupervised/DBScan';
import PCA from './pages/unsupervised/PCA';
import GMM from './pages/unsupervised/GMM';
import Hierarchical from './pages/unsupervised/Hierarchical';
import TSNE from './pages/unsupervised/TSNE';
import Apriori from './pages/unsupervised/Apriori';

// Ensemble
import RandomForest from './pages/ensemble/RandomForest';
import AdaBoost from './pages/ensemble/AdaBoost';
import GradientBoosting from './pages/ensemble/GradientBoosting';

// Deep Learning
import ANN from './pages/dl/ANN';
import CNN from './pages/dl/CNN';
import RNN from './pages/dl/RNN';
import Transformer from './pages/dl/Transformer';
import Autoencoder from './pages/dl/Autoencoder';
import GAN from './pages/dl/GAN';
import Diffusion from './pages/dl/Diffusion';
import GCN from './pages/dl/GCN';

// Reinforcement Learning
import QLearning from './pages/rl/QLearning';
import DQN from './pages/rl/DQN';

// Semi-Supervised
import PseudoLabeling from './pages/semi/PseudoLabeling';
import Contrastive from './pages/semi/Contrastive';

// Bulletproof Scroll Fix
function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    // A 50ms delay ensures Framer Motion has fully rendered the page height!
    const timer = setTimeout(() => {
      window.scrollTo({ top: 0, left: 0, behavior: 'instant' });

      // Safety catch: If your CSS makes the body scroll instead of the window
      document.documentElement.scrollTo({ top: 0, left: 0, behavior: 'instant' });
      document.body.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    }, 50);

    return () => clearTimeout(timer);
  }, [pathname]);

  return null;
}

function App() {
  return (
    <Router>
      <ScrollToTop />

      <MainLayout>
        <Routes>
          <Route path="/" element={<HomePage />} />

          {/* Supervised */}
          <Route path="/reg" element={<Reg />} />
          <Route path="/logreg" element={<LogisticRegression />} />
          <Route path="/regularization" element={<Regularization />} />
          <Route path="/nb" element={<NaiveBayes />} />
          <Route path="/knn" element={<KNN />} />
          <Route path="/d-trees" element={<DTrees />} />
          <Route path="/svm" element={<SVM />} />

          {/* Unsupervised */}
          <Route path="/kmeans" element={<KMeans />} />
          <Route path="/hierarchical" element={<Hierarchical />} />
          <Route path="/tsne" element={<TSNE />} />
          <Route path="/DBScan" element={<DBScan />} />
          <Route path="/pca" element={<PCA />} />
          <Route path="/gmm" element={<GMM />} />
          <Route path="/apriori" element={<Apriori />} />

          {/* Ensembles */}
          <Route path="/rf" element={<RandomForest />} />
          <Route path="/adaboost" element={<AdaBoost />} />
          <Route path="/gb" element={<GradientBoosting />} />

          {/* Deep Learning */}
          <Route path="/ann" element={<ANN />} />
          <Route path="/cnn" element={<CNN />} />
          <Route path="/rnn" element={<RNN />} />
          <Route path="/transformer" element={<Transformer />} />
          <Route path="/ae" element={<Autoencoder />} />
          <Route path="/gan" element={<GAN />} />
          <Route path="/gcn" element={<GCN />} />
          <Route path="/diffusion" element={<Diffusion />} />

          {/* Reinforcement */}
          <Route path="/ql" element={<QLearning />} />
          <Route path="/dqn" element={<DQN />} />

          {/* Semi-Supervised */}
          <Route path="/pl" element={<PseudoLabeling />} />
          <Route path="/cl" element={<Contrastive />} />

          {/* Meta */}
          <Route path="/about" element={<About />} />
          <Route path="/docs/:algoId?" element={<Docs />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
