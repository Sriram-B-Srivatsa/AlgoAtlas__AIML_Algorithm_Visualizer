# 🧠 AlgoAtlas: The Interactive Machine Learning Explorer

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Recharts](https://img.shields.io/badge/Recharts-22B573?style=for-the-badge&logo=react&logoColor=white)

Welcome to **AlgoAtlas**! If you have ever wondered how Artificial Intelligence actually "thinks", you are in the right place.

Most AI tutorials give you walls of complex math equations or boring lines of code. AlgoAtlas takes those complex mathematical concepts and turns them into **interactive video games, drawing pads, and visual animations**.

Whether you are an absolute beginner who wants to know how ChatGPT works, or a senior engineer looking for a deep-dive into the math of Gradient Descent, AlgoAtlas is your personal sandbox.

---

## 🌟 What is the "Secret Sauce"? (The 2x2 Dashboard)

Every single algorithm in AlgoAtlas follows a strict, beautifully organized **2x2 Analytical Dashboard** layout. When you train an AI, you won't just get a boring text response. You will see:

*   📍 **Top-Left (The Animation):** A video player showing the AI learning step-by-step over time. You will watch boundary lines morph, clusters separate, and agents navigate mazes.
*   📊 **Top-Right (Mathematical Insights):** Plain-English explanations of what the AI just did, alongside its final accuracy, error rates, and parameters.
*   📉 **Bottom-Left (The Learning Curve):** A dynamic, interactive graph showing the AI's "Loss" (its mistakes) dropping toward zero as it gets smarter.
*   🔬 **Bottom-Right (Algorithm-Specific Charts):** Deep-dive charts like Feature Importance, Softmax Probabilities, or 3D Data Visualizations.

---

## 🚀 Live Deployment
**Experience the platform instantly without any installation:**  
🔗 **[AlgoAtlas Live Demo](https://algo-atlas-aiml-algorithm-visualizer-6ophucfzr.vercel.app)**  
*(Hosted on a Vercel Serverless Edge network. Please allow 5-10 seconds for the Python backend to "wake up" on your very first click!)*

---

## ✨ Core Engineering Features

*   **100% Pure NumPy Deep Learning:** To achieve blazing fast, serverless deployment without crashing cloud memory limits, complex engines (like the **Nano-GPT Transformer**, **CartPole DQN**, **Autoencoders**, and **CNN Feature Extractors**) were built entirely from scratch using pure Matrix Mathematics (NumPy), completely bypassing heavy libraries like TensorFlow or PyTorch.
*   **High-FPS Physics via HTML5 Canvas:** Reinforcement Learning and Drawing Pad environments bypass the React Virtual DOM to paint directly to the screen via HTML5 Canvas, ensuring 60-FPS buttery-smooth rendering.
*   **Live Recharts Analytics:** Dynamic visualization of Training vs. Validation Loss, Coefficient Shrinkage, Epsilon Decay, and Self-Attention Matrices.
*   **Headless Matplotlib Rendering:** Heavy decision boundary calculations (e.g., SVM, GMM) are offloaded to the Flask backend using non-interactive `Agg` rendering, delivering highly compressed Base64 strings to keep the network payload under 50KB.

---

## 🚀 The Algorithms (What you can play with!)

AlgoAtlas features **over 20 of the most powerful algorithms in the world**, divided into 6 easy-to-understand categories.

### 🎯 1. Supervised Learning
*Here, we give the AI the answers (labels) and teach it to predict future answers.*
*   **Linear & Polynomial Regression:** Draw a curved line through dots to predict continuous numbers.
*   **Logistic Regression:** Watch the AI create a "Probability Heatmap" to classify Red vs. Blue dots.
*   **Ridge & Lasso Regularization:** Add a "penalty" to the AI to stop it from over-complicating its math.
*   **K-Nearest Neighbors (KNN):** The AI memorizes its neighbors to make decisions.
*   **Decision Trees:** Watch the AI build a literal "Yes/No" flowchart to categorize data.
*   **Support Vector Machines (SVM):** The AI draws the "widest possible street" between two classes.
*   **Gaussian Naive Bayes:** The AI builds 3D Bell Curves based on probability!

### 🔍 2. Unsupervised Learning
*Here, we give the AI NO answers. It has to find hidden patterns all by itself.*
*   **K-Means Clustering:** Watch moving "Stars" hunt down the center of data clumps.
*   **DBSCAN:** An algorithm that acts like a contagion, spreading through dense data and ignoring "noise".
*   **Principal Component Analysis (PCA):** Watch 2D data get mathematically crushed into a 1D line.
*   **Gaussian Mixture Models (GMM):** Like K-Means, but it can stretch and rotate into ovals!
*   **Hierarchical Clustering:** Builds a massive "Family Tree" (Dendrogram) of your data points.
*   **t-SNE:** Unrolls a 3D "Swiss Roll" of data onto a flat 2D piece of paper.
*   **Apriori (Association Rules):** A Grocery Store simulator! Find out why buying Diapers makes people buy Beer.

### 🤝 3. Ensemble Learning
*Why use one AI when you can combine 100 AIs into a super-brain?*
*   **Random Forest:** A massive crowd of Decision Trees taking a democratic vote.
*   **AdaBoost:** Watch the AI build a sequence of simple lines, focusing entirely on fixing its past mistakes!
*   **Gradient Boosting:** The ultimate tabular data algorithm. It mathematically predicts its own errors.

### 🧠 4. Deep Learning
*Algorithms inspired by the human brain.*
*   **Artificial Neural Networks (ANN):** The foundation of Deep Learning.
*   **Convolutional Neural Networks (CNN):** A **Drawing Pad**! Draw a number with your mouse, and watch the AI apply visual filters to extract edges and predict the digit.
*   **Recurrent Neural Networks (RNN):** A **Timeline**. Draw a stock market trend, and watch the AI's "Memory Cell" try to predict the future.
*   **Transformers (LLMs):** The math behind ChatGPT. Type a sentence and watch the "Self-Attention Matrix" light up as the AI understands grammar.
*   **Autoencoders:** Watch the AI crush data into a tiny bottleneck and try to rebuild it from memory.
*   **Generative Adversarial Networks (GANs):** Watch two AIs fight. One creates fake data, the other tries to catch the fakes.
*   **Diffusion Models:** The math behind Midjourney and DALL-E. Watch the AI build structures out of pure static noise.
*   **Graph Convolutional Networks (GCN):** Deep learning for Social Networks. Connect dots with lines, and watch the AI pass messages between friends.

### 🎮 5. Reinforcement Learning
*The AI learns by playing a video game.*
*   **Q-Learning:** A **Maze Builder**! Draw walls, a start, and a goal. Watch the AI crash into walls until it learns the perfect path to the treasure.
*   **Deep Q-Networks (DQN):** Watch an AI use a Neural Network to figure out the physics of balancing a broomstick on a moving cart!

### ⚖️ 6. Semi & Self-Supervised Learning
*Bridging the gap when human labels are too expensive.*
*   **Pseudo-Labeling:** Give the AI 3 colored dots and 100 gray dots. Watch it confidently color in the rest of the map by itself!
*   **Contrastive Learning:** Watch a messy cloud of data violently rip itself apart into perfect clusters without ever using a label.

---

## 💻 Tech Stack (How it was built)

**The Frontend (User Interface):**
*   **React.js:** The core framework that runs the website.
*   **HTML5 Canvas:** Used to draw the interactive dots, the CNN drawing pad, and the RL mazes.
*   **Recharts:** A beautiful charting library used to draw the Line, Bar, and Scatter plots.
*   **Framer Motion:** Adds buttery-smooth page transitions and animations.

**The Backend (The Brains):**
*   **Python (Flask):** The server that receives your canvas clicks and crunches the heavy math.
*   **NumPy & Pandas:** The core mathematical engines. (Many algorithms, like the Transformer and DQN, were written purely from scratch using NumPy!).
*   **Scikit-Learn:** The industry-standard library used for the baseline Machine Learning models.
*   **Matplotlib:** A graphing library that draws the high-definition animation frames on the server, converts them to text (Base64), and sends them instantly to the website.

---

## 🛠️ Step-by-Step Installation (For Absolute Beginners)

Don't worry if you aren't a programmer. Follow these steps exactly, and you will have this running on your computer in 5 minutes!

### Prerequisites (Things you need installed first)
1.  **Python** (Version 3.8 or higher) - [Download Here](https://www.python.org/downloads/)
2.  **Node.js** (Version 14 or higher) - [Download Here](https://nodejs.org/)
3.  **Git** - [Download Here](https://git-scm.com/downloads)
4.  *Optional but recommended:* A code editor like **Visual Studio Code (VSCode)**.

---

## 💻 How to Run Locally

Because the platform is deployed live on Vercel, **you do not need to run this locally.** However, if you wish to run it for development or modification, follow these steps:

### Step 1: Download the Code
Open your computer's Terminal (Command Prompt on Windows, Terminal on Mac) and run:
```bash
git clone [https://github.com/Sriram-B-Srivatsa/AlgoAtlas__AIML_Algorithm_Visualizer.git]
cd AlgoAtlas__AIML_Algorithm_Visualizer
```

### Step 2: Start the Brain (The Python Backend)
We need to start the server that does all the math. In your terminal, type:
```bash
cd api
```

Now, create a "Virtual Environment". This is like a safe, isolated room on your computer just for this project's Python files:
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On Mac/Linux:
python3 -m venv venv
source venv/bin/activate
```
*(You will know it worked if you see `(venv)` appear at the start of your terminal line).*

Next, install the required math libraries:
```bash
pip install -r requirements.txt
```

Finally, turn the brain on!
```bash
flask --app index run --port 5000
```
*(Leave this terminal window open!).*

### Step 3: Start the Website (The React Frontend)
Open a **brand new** terminal window (leave the Python one running in the background!).

Navigate to the frontend folder:
```bash
cd .
```

Install the website building blocks:
```bash
npm install
```

Start the website:
```bash
npm start
```
Your browser will automatically open to `http://localhost:3000`. **You are now running AlgoAtlas!**

---

## 📁 Project Structure (For the Nerds)

If you want to look at the code, here is how everything is beautifully organized:

```text
AlgoAtlas__AIML_Algorithm_Visualizer/
├── api/
│   ├── index.py               # The central nervous system (Routes API calls)
│   ├── requirements.txt       # List of all Libs and Packages
|   ├── datasets/              # Where sample data is generated
│   └── models/                # Where the math happens
│       ├── dl/                # CNN.py, Transformer.py, RNN.py...
│       ├── ensemble/          # AdaBoost.py, RandomForest.py, GradientBoosting.py
│       ├── rl/                # QLearning.py, DQN.py
│       ├── semi/              # PseudoLabeling.py, Contrastive.py
│       ├── supervised/        # DTrees.py, Reg.py, SVM.py...
│       └── unsupervised/      # kmeans.py, PCA.py, Apriori.py...
└── public/
└── src/
    ├── api.jsx            # How the frontend talks to the backend
    ├── App.jsx            # Main Container of UI
    ├── index.css          # Styling Entry Point
    ├── index.jsx          # Frontend Entry Point
    ├── data/              # Contains the text for this Documentation!
    ├── components/        # Reusable UI (Navbar, Footer)
    ├── layouts/           # Layout Design
    └── pages/             # The visualizers! Matches backend folders 1-to-1.
```

---

## 🛟 Troubleshooting & FAQ

**"I clicked 'Train', but it says Network Error!"**
*   This means your React website can't find your Python brain. Ensure your Python terminal is open, running, and says `Running on http://127.0.0.1:5000`.

**"My Decision Tree image is just a blank white box with an error inside!"**
*   The Scikit-Learn tree drawer requires a tiny external program called **Graphviz** to draw the arrows between the boxes. You can download it [here](https://graphviz.org/download/). If you don't want to download it, don't worry! The rest of the app and all the other graphs will still work perfectly.

**"The animations are too fast/slow!"**
*   Look for the **"Speed"** slider right underneath the video player on the dashboards! You can slow it down to a crawl to really analyze the math.

---

## 🤝 Contributing
Want to add a new algorithm? Contributions are welcome!
If you build a new model, please ensure you strictly follow the **2x2 Analytical Dashboard** UI format established in the existing `.jsx` files to keep the educational experience consistent.

## 📄 License
This project is open-source and available under the MIT License. Feel free to use it to learn, teach, or build upon!
