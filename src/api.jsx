import axios from 'axios';
const API_URL = '/api';

export const getModels = async () => {
  const response = await axios.get(`${API_URL}`);
  return response.data;
};

export const getModelDetails = async (id) => {
  const response = await axios.get(`${API_URL}/${id}`);
  return response.data;
};

// Polynomial Regression
export async function runPolynomialRegression(data) {
  const response = await axios.post(`${API_URL}/regression`, data);
  return response.data;
}

export async function getPolynomialRegressionSampleData(dataset_type = 'linear', n_samples = 30, noise_level = 0.5) {
  const response = await axios.post(`${API_URL}/regression/sample_data`, {
    dataset_type, n_samples, noise_level
  });
  return response.data;
}

// SVM
export const getSVMSampleData = async (n_samples = 30, noise = 0.1) => {
  const response = await axios.get(`${API_URL}/svm/sample?n_samples=${n_samples}&noise=${noise}`);
  return response.data;
};

export const runSVM = async (data) => {
  const response = await axios.post(`${API_URL}/svm`, data);
  return response.data;
};

// ANN (FIXED)
export const runANN = async (data) => {
  const response = await axios.post(`${API_URL}/ann/train`, data);
  return response.data;
};

export const getANNSampleData = async (dataset_type = 'blobs', count = 100, n_clusters = 2, variance = 0.5) => {
  const response = await axios.post(`${API_URL}/ann/sample_data`, {
    dataset_type, count, n_clusters, variance
  });
  return response.data;
};

// KNN
export const runKnnClassification = async (data) => {
  const response = await axios.post(`${API_URL}/knn-classification`, data);
  return response.data;
};

export const runKnnRegression = async (data) => {
  const response = await axios.post(`${API_URL}/knn-regression`, data);
  return response.data;
};

export const predictKnnPoint = async (data) => {
  const response = await axios.post(`${API_URL}/knn-predict-point`, data);
  return response.data;
};

export const getKnnDecisionBoundary = async (data) => {
  const response = await axios.post(`${API_URL}/knn-decision-boundary`, data);
  return response.data;
};

// Decision Trees (FIXED)
export const runDTrees = async (data) => {
  const response = await axios.post(`${API_URL}/dtree/visualize`, data);
  return response.data;
};

export const getDTreesSampleData = async (dataset_type = 'blobs', count = 40, n_clusters = 3, variance = 0.5) => {
  const response = await axios.post(`${API_URL}/dtree/sample_data`, {
    type: 'classification', dataset_type, count, n_clusters, variance
  });
  return response.data;
};

// K-Means
export const runKMeans = async (data) => {
  const response = await axios.post(`${API_URL}/kmeans`, data);
  return response.data;
};

export const getKMeansSampleData = async (dataset_type = 'blobs', n_samples = 100, n_clusters = 3, variance = 0.5) => {
  const response = await axios.get(`${API_URL}/kmeans/sample?dataset_type=${dataset_type}&n_samples=${n_samples}&n_clusters=${n_clusters}&variance=${variance}`);
  return response.data;
};

// PCA
export const runPCA = async (data) => {
  const response = await axios.post(`${API_URL}/pca`, data);
  return response.data;
};

export const getPCASampleData = async (n_samples = 30, noise = 5.0) => {
  const response = await axios.get(`${API_URL}/pca/sample-data?n_samples=${n_samples}&noise=${noise}`);
  return response.data;
};

// DBSCAN (FIXED)
export const runDBSCAN = async (data) => {
  const response = await axios.post(`${API_URL}/dbscan/run_complete`, data);
  return response.data;
};

export const getDBSCANSampleData = async (options) => {
  const response = await axios.post(`${API_URL}/dbscan/sample_data`, options);
  return response.data;
};

// Health check
export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_URL}/health`);
    return response.data;
  } catch (error) {
    return { status: 'error', message: error.message };
  }
};

// GLOBAL AUTO-SCROLL LISTENER
axios.interceptors.response.use((response) => {
  const url = (response.config.url || '').toLowerCase();
  // Only auto-scroll when a Training/Computation POST request finishes.
  // We ignore 'sample' and 'predict' so the screen doesn't jump away from the canvas!
  if (response.config.method === 'post' && !url.includes('sample') && !url.includes('predict')) {
    setTimeout(() => window.scrollTo({ top: 600, behavior: 'smooth' }), 300);
  }
  return response;
}, (error) => Promise.reject(error));
