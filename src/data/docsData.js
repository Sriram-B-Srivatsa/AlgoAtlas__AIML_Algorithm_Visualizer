export const docsData = [
  {
    category: "Supervised Learning",
    algorithms:[
      {
        id: "reg",
        title: "Polynomial & Linear Regression",
        path: "/reg",
        content: `
          <h3>1. What is it?</h3>
          <p>Regression is the most fundamental algorithm in Machine Learning. It looks at a scatter of data points and mathematically draws the "Line of Best Fit" through them to predict future, continuous numbers (like stock prices or temperature).</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine stretching a rubber band through a pegboard. The pegs are your data points. The rubber band naturally snaps into a position that minimizes the distance between itself and all the pegs. That rubber band is your regression line!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> The AI guesses a random line.</li>
            <li><strong>Step 2:</strong> It calculates how far off its line is from the actual data points (the Error/Cost).</li>
            <li><strong>Step 3:</strong> Using Calculus (Gradient Descent), it tweaks the angle and height of the line to reduce the error.</li>
            <li><strong>Step 4:</strong> It repeats this until the line stops moving (Convergence).</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The Equation of the Line:</strong> <code>y = β₀ + β₁x + β₂x² ...</code><br/>
          <strong>The Cost Function (Mean Squared Error):</strong> <code>MSE = 1/n Σ(Actual - Predicted)²</code><br/>
          The algorithm squares the errors to punish massive mistakes much harder than small mistakes.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Polynomial Degree:</strong> Degree 1 is a straight line. Degree 2 is a U-shaped parabola. Degree 10 is a crazy squiggly line.</li>
            <li><strong>Learning Rate (Alpha):</strong> How big of a step the AI takes when adjusting the line. Too small = too slow. Too big = the line bounces uncontrollably.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>R² Score (R-Squared):</strong> A percentage of how well your line fits the data. 100% means the line perfectly hits every dot. 0% means the line is completely useless.</p>

          <h3>7. Best Use Cases</h3>
          <p>Predicting house prices based on square footage, forecasting sales, or predicting fuel efficiency based on car weight.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Categorizing data (e.g., Is this a dog or a cat?). Regression outputs infinite continuous numbers, not discrete categories.</p>

          <h3>9. Major Advantages</h3>
          <p>Incredibly fast to compute, very easy to understand, and highly interpretable (you can look at the exact mathematical equation it generates).</p>

          <h3>10. Major Disadvantages</h3>
          <p>Prone to Outliers. Because it squares the errors (MSE), a single massive outlier can drag the entire line away from the actual trend.</p>

          <h3>11. Data Assumptions</h3>
          <p>It assumes the data has a smooth mathematical relationship. If the data is purely random static, regression will just draw a flat, useless line through the middle.</p>

          <h3>12. How it Overfits</h3>
          <p>If you set the Polynomial Degree to 15 on a dataset with only 10 points, the line will desperately zig-zag to hit every single point perfectly, completely missing the actual underlying trend.</p>

          <h3>13. How it Underfits</h3>
          <p>If your data is curved (like a smile), and you set the Degree to 1, it will draw a stiff, straight line that fails to capture the curve.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * d² + d³)</code> (Matrix inversion is heavy on features, where <em>d</em> = degree/features).</li>
            <li><strong>Training Space:</strong> <code>O(n * d + d²)</code> (To store the dataset and covariance matrix).</li>
            <li><strong>Prediction Time:</strong> <code>O(d)</code> (Lightning fast, just a simple dot product).</li>
            <li><strong>Prediction Space:</strong> <code>O(d)</code> (Only stores the mathematical weights).</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Always normalize your data (scale X to be between -1 and 1) before running polynomial regression, or your numbers will explode to infinity when doing x², x³, etc!</p>
        `
      },
      {
        id: "logreg",
        title: "Logistic Regression",
        path: "/logreg",
        content: `
          <h3>1. What is it?</h3>
          <p>Despite the word "Regression" in its name, this is a <strong>Classification</strong> algorithm. Instead of predicting an infinite number, it predicts a percentage (0% to 100%) that an item belongs to a specific category.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Think of a medical test for a disease. The doctor doesn't just say "Yes" or "No". They say, "Based on your blood pressure, there is an 85% probability you have the disease." If the probability is above 50%, they classify it as "Yes".</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> It calculates a standard straight line through the data (just like linear regression).</li>
            <li><strong>Step 2:</strong> It forces that straight line through a mathematical "crusher" called a Sigmoid Function.</li>
            <li><strong>Step 3:</strong> The Sigmoid bends the straight line into an S-Shape, forcing all outputs to be trapped exactly between 0.0 and 1.0.</li>
            <li><strong>Step 4:</strong> It draws a boundary exactly where the probability is 0.5 (50%).</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The Sigmoid Function:</strong> <code>P(y=1) = 1 / (1 + e^-z)</code> (where z is the standard line equation).<br/>
          <strong>Log-Loss (Cross-Entropy):</strong> <code>Loss = -[y*log(p) + (1-y)*log(1-p)]</code>. This heavily penalizes the AI if it is 99% confident but gets the answer wrong.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Learning Rate:</strong> How aggressively the AI adjusts its probability curve during gradient descent.</li>
            <li><strong>Epochs:</strong> How many times it loops over the dataset to refine the boundary.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Log-Loss:</strong> Lower is better. A Log-Loss of 0.0 means the AI is 100% perfectly confident and correct about every single dot.</p>

          <h3>7. Best Use Cases</h3>
          <p>Spam filters (Is this email Spam or Not?), credit card fraud detection, and medical diagnosis.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Highly complex, swirling data. Because it relies on a straight mathematical boundary bent into an S-curve, it cannot draw circles or spirals to separate classes.</p>

          <h3>9. Major Advantages</h3>
          <p>It provides actual, mathematical probabilities! You know exactly <em>how confident</em> the AI is about its prediction.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It assumes the data is "linearly separable" (meaning you can draw a single straight line to divide the reds from the blues).</p>

          <h3>11. Data Assumptions</h3>
          <p>It assumes no severe multicollinearity (features shouldn't be highly correlated with each other, like "Weight in Lbs" and "Weight in Kgs").</p>

          <h3>12. How it Overfits</h3>
          <p>With too many features and no regularization, the S-curve becomes infinitely steep, essentially acting like a rigid wall rather than a gentle probability gradient.</p>

          <h3>13. How it Underfits</h3>
          <p>If you stop the training (Epochs) too early, the boundary line will sit aimlessly in the middle of nowhere, misclassifying everything.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * d * i)</code> (where <em>i</em> = number of gradient descent iterations). Very fast.</li>
            <li><strong>Training Space:</strong> <code>O(d)</code> (Only needs to update the weights matrix in memory).</li>
            <li><strong>Prediction Time:</strong> <code>O(d)</code> (Calculates the line and passes through the Sigmoid math).</li>
            <li><strong>Prediction Space:</strong> <code>O(d)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>The "Decision Boundary" you see in the visualizer is literally just a contour map of the 50% probability zone!</p>
        `
      },
      {
        id: "regularization",
        title: "Ridge & Lasso Regularization",
        path: "/regularization",
        content: `
          <h3>1. What is it?</h3>
          <p>Regularization is a "handbrake" for Machine Learning. If a model is too smart for its own good (drawing crazy zig-zags to memorize data), Regularization artificially punishes it to force it to draw a smooth, generalized curve.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine hiring an overly ambitious architect who designs a house with 47 tiny, useless rooms just because there was empty space on the blueprint. Regularization is the Project Manager who charges a massive tax for every extra wall built, forcing the architect to build a simple, spacious, usable house instead.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> The model tries to minimize its normal error (MSE).</li>
            <li><strong>Step 2:</strong> We add a "Penalty Term" to the total error based on how big the mathematical weights (coefficients) are getting.</li>
            <li><strong>Step 3:</strong> To lower the total error, the AI is mathematically forced to shrink its weights toward zero, flattening the curve.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Ridge (L2):</strong> <code>Cost = MSE + α * Σ(weights²)</code>. Punishes large weights by squaring them.<br/>
          <strong>Lasso (L1):</strong> <code>Cost = MSE + α * Σ|weights|</code>. Punishes weights using absolute value, which magically forces useless weights to exactly 0.0!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Alpha (Penalty Strength):</strong> If Alpha is 0, it acts like normal regression. If Alpha is 100, the penalty is so severe the model just draws a completely flat, horizontal line.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Coefficient Shrinkage Path:</strong> Watch the graph in the visualizer. As Alpha increases, you will literally see the math weights (w1, w2) shrink toward zero. In Lasso, they will snap directly to the 0 axis!</p>

          <h3>7. Best Use Cases</h3>
          <p>Datasets with thousands of features but very few rows of data (like Genetics/DNA datasets).</p>

          <h3>8. Worst Use Cases</h3>
          <p>When you have infinite, perfectly clean data. Regularization intentionally biases the model; if you have perfect data, you don't want to bias it.</p>

          <h3>9. Major Advantages</h3>
          <p><strong>Lasso (L1)</strong> acts as an automatic Feature Selector! If you give it 100 variables, it will multiply the useless ones by exactly zero, effectively deleting them from the model.</p>

          <h3>10. Major Disadvantages</h3>
          <p>You have to manually guess and test the perfect "Alpha" value. Too much penalty, and the model becomes stupid. Too little, and it overfits.</p>

          <h3>11. Data Assumptions</h3>
          <p>Requires strict Feature Scaling! If "Income" is in the 100,000s and "Age" is in the 30s, the penalty will unfairly crush the "Age" variable unless you scale them both to be between 0 and 1 first.</p>

          <h3>12. How it Overfits</h3>
          <p>It doesn't! The entire point of Regularization is to *prevent* overfitting.</p>

          <h3>13. How it Underfits</h3>
          <p>If you crank Alpha to 50.0, the penalty for having any math weights is so high that the AI gives up, sets all weights to near-zero, and draws a flat line.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * d² + d³)</code> (Similar to standard regression, plus penalty calculations).</li>
            <li><strong>Training Space:</strong> <code>O(n * d + d²)</code>.</li>
            <li><strong>Prediction Time:</strong> <code>O(d)</code> (Even faster in Lasso since many weights become 0).</li>
            <li><strong>Prediction Space:</strong> <code>O(d)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Use <strong>Ridge (L2)</strong> when you want to keep all variables but shrink their influence. Use <strong>Lasso (L1)</strong> when you have too much junk data and want the AI to delete the useless columns automatically.</p>
        `
      },
      {
        id: "knn",
        title: "K-Nearest Neighbors (KNN)",
        path: "/knn",
        content: `
          <h3>1. What is it?</h3>
          <p>KNN is the simplest algorithm in AI. It doesn't actually "learn" any math equations. It just memorizes the entire dataset. When you ask it to predict a new point, it looks at the 'K' closest dots around it and takes a majority vote!</p>

          <h3>2. Real-World Analogy</h3>
          <p>If you move to a new neighborhood and want to know what political party the neighborhood belongs to, you knock on the doors of your 5 closest neighbors. If 4 are Democrats and 1 is Republican, you assume it's a Democrat area. That is exactly K=5 KNN!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Memorize all training data.</li>
            <li><strong>Step 2:</strong> Take a new, unknown point.</li>
            <li><strong>Step 3:</strong> Calculate the literal ruler-distance from this new point to every single point in the database.</li>
            <li><strong>Step 4:</strong> Pick the K closest points. Whichever color is most common among those K points wins.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Euclidean Distance:</strong> <code>d = √[(x₂ - x₁)² + (y₂ - y₁)²]</code><br/>
          It's just the Pythagorean theorem you learned in middle school geometry!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>K (Number of Neighbors):</strong> How many nearby points to include in the vote. K=1 only looks at the absolute closest dot. K=15 looks at a massive crowd.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Validation Score:</strong> The graph shows the sweet spot. You will see accuracy is usually terrible at K=1, rises to a peak around K=5, and then falls back down as K gets too high.</p>

          <h3>7. Best Use Cases</h3>
          <p>Simple recommendation systems (e.g., "Users with similar viewing history to you also liked this movie").</p>

          <h3>8. Worst Use Cases</h3>
          <p>Massive datasets or real-time applications. Because KNN has no "brain" (no equation), it must re-measure the distance to a million points every single time you ask for a prediction. It is horribly slow.</p>

          <h3>9. Major Advantages</h3>
          <p>Zero training time. The "training" phase is literally just saving the data to memory.</p>

          <h3>10. Major Disadvantages</h3>
          <p>Requires massive memory storage, very slow at prediction time, and suffers heavily from the "Curse of Dimensionality" (distance measurements break down in 3D/4D/100D spaces).</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes similar things exist close to each other. Requires strict feature scaling (if one axis is in millimeters and one is in kilometers, the distance math is completely ruined).</p>

          <h3>12. How it Overfits</h3>
          <p><strong>Setting K=1.</strong> The boundary will draw a tiny circle around every single stray noise dot, memorizing the noise instead of the global trend.</p>

          <h3>13. How it Underfits</h3>
          <p><strong>Setting K to the total dataset size.</strong> If you have 100 points, and you set K=100, the AI will just guess the most common color in the entire dataset every single time, totally ignoring local patterns.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(1)</code> (Zero training! It just saves the dataset to memory).</li>
            <li><strong>Training Space:</strong> <code>O(n * d)</code> (Must store the entire training dataset forever).</li>
            <li><strong>Prediction Time:</strong> <code>O(n * d)</code> (Painfully slow. Must measure the distance to every single point in the database).</li>
            <li><strong>Prediction Space:</strong> <code>O(1)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Always use an ODD number for K in binary classification (e.g., K=3, 5, 7). If you use K=4, you can end up with a 2-vs-2 tie in the voting!</p>
        `
      },
      {
        id: "dtrees",
        title: "Decision Trees",
        path: "/d-trees",
        content: `
          <h3>1. What is it?</h3>
          <p>A Decision Tree is an algorithm that writes a flowchart. It slices the data over and over again using simple Yes/No questions until it perfectly isolates the different classes into their own rectangular boxes.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Think of the "Akinator" game. It asks: "Is it an animal?" -> Yes. "Does it bark?" -> Yes. "Is it a Dog." The algorithm mathematically finds the absolute best questions to ask to win the game in the fewest steps.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Look at all features (X-axis, Y-axis) and find a split (e.g., "Is X > 5?") that separates the Red dots from the Blue dots the best.</li>
            <li><strong>Step 2:</strong> Draw a line to split the map in two.</li>
            <li><strong>Step 3:</strong> Take the two new boxes, and split them again.</li>
            <li><strong>Step 4:</strong> Repeat until every box (Leaf) contains only one color of dot, or you hit the Max Depth limit.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Gini Impurity:</strong> <code>G = 1 - Σ(p_i²)</code>. It measures how "mixed" a box is. A box with 50 Reds and 50 Blues has a high Gini score (Bad). A box with 100 Reds and 0 Blues has a Gini score of 0.0 (Perfect!).</p>
          <p><strong>Entropy:</strong> <code>H = -Σ(p_i * log₂(p_i))</code>. An alternative to Gini, this formula comes from Information Theory and measures the level of "chaos" in a box. A 50/50 mixed box has an Entropy of 1.0 (Maximum Chaos). A purely red box has an Entropy of 0.0.</p>
          <p><strong>Information Gain:</strong> The algorithm calculates the Entropy <em>before</em> a split, and subtracts the Entropy <em>after</em> a split. The tree always permanently chooses the line that yields the highest Information Gain!</p>
          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Max Depth:</strong> How many layers deep the flowchart can go.</li>
            <li><strong>Min Samples Split:</strong> A rule saying "Don't split a box if it only has 2 dots left in it."</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Total Nodes vs Leaves:</strong> A "Node" is a question. A "Leaf" is an endpoint where a final prediction is made. Deeper trees have exponentially more nodes.</p>

          <h3>7. Best Use Cases</h3>
          <p>Medical diagnosis logic, loan approval systems, and any time you need an AI decision to be 100% human-readable and legally explainable.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Extrapolating future trends (like predicting stock prices next year). Trees cannot predict numbers higher than what they saw in training; they just draw flat, blocky staircases.</p>

          <h3>9. Major Advantages</h3>
          <p>No data scaling required! It doesn't care if X is in thousands and Y is in decimals. It also handles missing data and categorical text naturally.</p>

          <h3>10. Major Disadvantages</h3>
          <p>They are wildly unstable. Changing just one dot in the training data can cause the entire tree to completely redraw a totally different flowchart.</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes the best way to divide data is with perpendicular, straight lines (box cuts). It cannot draw diagonal lines!</p>

          <h3>12. How it Overfits</h3>
          <p>If you don't set a Max Depth, the tree will grow infinitely deep until it draws a microscopic, hyper-specific box around every single noisy outlier.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting Max Depth = 1 (A "Stump"). It will draw exactly one line across the screen and give up.</p>

          <h3>14. Time & Space Complexity (based on CART)</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * d * log(n))</code> (Must sort every feature to find the best Gini split).</li>
            <li><strong>Training Space:</strong> <code>O(nodes)</code> (Memory scales with how deep the tree grows).</li>
            <li><strong>Prediction Time:</strong> <code>O(depth)</code> (Blazing fast. Just answering a few Yes/No questions).</li>
            <li><strong>Prediction Space:</strong> <code>O(depth)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Never use a single Decision Tree in production. Always upgrade to a Random Forest or Gradient Boosted Tree to fix the instability issues!</p>
        `
      },
      {
        id: "svm",
        title: "Support Vector Machines (SVM)",
        path: "/svm",
        content: `
          <h3>1. What is it?</h3>
          <p>SVM is the "Wide Street" algorithm. While other algorithms just draw a thin line between two classes, SVM tries to draw the widest possible multi-lane highway between them without touching any dots.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine trying to place a thick plank of wood between red and blue marbles on a table. You want to use the thickest plank possible without crushing any marbles. The marbles that actually touch the edges of the plank are physically holding it in place. Those specific marbles are the "Support Vectors"!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Draw a line to separate the classes.</li>
            <li><strong>Step 2:</strong> Expand two parallel "gutters" (margins) outward from the line until they bump into the closest data points.</li>
            <li><strong>Step 3:</strong> Mathematically adjust the angle of the line to maximize the width of that street.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Hyperplane Equation:</strong> <code>w·x + b = 0</code><br/>
          <strong>Margin Maximization:</strong> It tries to minimize <code>||w||²</code> subject to the constraint that all points are on the correct side of the street (<code>y_i(w·x_i + b) ≥ 1</code>).</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Margin Width (C-Parameter):</strong> A "Soft" margin allows some dots to step inside the street to get a better overall angle. A "Hard" margin is strictly unforgiving and shrinks the street to avoid a single outlier.</li>
            <li><strong>Kernels (RBF, Poly):</strong> The magic trick! If data can't be separated by a straight line, kernels warp the 2D paper into a 3D bowl so you can slice it with a flat plane.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Number of Support Vectors:</strong> Look at the visualizer. The gold-circled dots are the Support Vectors. The AI literally threw away the rest of the dataset because only the points touching the margins matter to the math!</p>

          <h3>7. Best Use Cases</h3>
          <p>High-dimensional data (like text classification or image recognition) where the number of features exceeds the number of rows.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Massive datasets (over 100,000 rows). Because SVM involves heavy quadratic programming, it scales terribly and takes forever to train on big data.</p>

          <h3>9. Major Advantages</h3>
          <p>Memory efficient! Because the final model only saves the coordinates of the "Support Vectors" and deletes the rest of the training data, it takes up very little RAM.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It doesn't naturally output probabilities (like Logistic Regression does). It just outputs a hard "Class A" or "Class B" based on which side of the street you land on.</p>

          <h3>11. Data Assumptions</h3>
          <p>Data MUST be perfectly scaled. If X is large and Y is small, the "widest street" will be mathematically distorted.</p>

          <h3>12. How it Overfits</h3>
          <p>Using a Hard Margin (High C) or a complex RBF Kernel with high Gamma. The boundary will warp into incredibly tight, jagged islands around specific points.</p>

          <h3>13. How it Underfits</h3>
          <p>Using a Linear Kernel on data that forms concentric circles. A straight line can never solve a circle.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n²)</code> to <code>O(n³)</code> (Horrible for big data. The quadratic solver explodes as rows increase).</li>
            <li><strong>Training Space:</strong> <code>O(n²)</code> (To store the Kernel matrix).</li>
            <li><strong>Prediction Time:</strong> <code>O(S * d)</code> (Where <em>S</em> is the number of Support Vectors. Extremely fast!).</li>
            <li><strong>Prediction Space:</strong> <code>O(S * d)</code> (Only stores the exact Support Vector coordinates).</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>The "Kernel Trick" allows SVMs to calculate distance in infinite-dimensional space without ever actually doing the heavy math of rendering the infinite dimensions!</p>
        `
      },
      {
        id: "nb",
        title: "Gaussian Naive Bayes",
        path: "/nb",
        content: `
          <h3>1. What is it?</h3>
          <p>Naive Bayes doesn't draw boundaries; it builds 3D probability clouds (Bell Curves) over the data. When a new point drops in, it calculates the exact odds of that point belonging to each cloud, and bets on the highest probability.</p>

          <h3>2. Real-World Analogy</h3>
          <p>If you see someone who is 7 feet tall, are they more likely an NBA player or an Accountant? Even though the NBA has taller people, there are 2 million accountants and only 400 NBA players. Bayes' Theorem multiplies the <em>Height Probability</em> by the <em>Base Population Probability (Priors)</em> to realize they are still probably an accountant!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Look at Class 0. Find the mathematical mean (center) and variance (spread) of its points to build a Gaussian Bell Curve.</li>
            <li><strong>Step 2:</strong> Repeat for Class 1.</li>
            <li><strong>Step 3:</strong> Calculate "Priors" (did the user paint 80% of the dots Red and only 20% Blue?).</li>
            <li><strong>Step 4:</strong> For a new point, multiply the Bell Curve height by the Prior probability to get the final score.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Bayes' Theorem:</strong> <code>P(Class|Data) = [ P(Data|Class) * P(Class) ] / P(Data)</code><br/>
          The <code>P(Data|Class)</code> is calculated using the classic Gaussian PDF formula: <code>(1 / √2πσ²) * e^[-(x-μ)² / 2σ²]</code>.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <p>Practically none! That is the beauty of Naive Bayes. There are no learning rates, epochs, or tree depths to tune. It is pure, direct statistics.</p>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Priors:</strong> Shown in the Bar Chart. If you place 90 Red dots and 10 Blue dots, the Prior for Red is 90%. The AI will aggressively default to guessing Red unless the evidence overwhelmingly proves otherwise.</p>

          <h3>7. Best Use Cases</h3>
          <p>Spam Filtering and Text Classification. It is incredibly fast and scales beautifully to thousands of words in an email.</p>

          <h3>8. Worst Use Cases</h3>
          <p>When the features are deeply intertwined. For example, if X and Y represent Latitude and Longitude, Naive Bayes fails because it assumes X and Y have nothing to do with each other.</p>

          <h3>9. Major Advantages</h3>
          <p>Blazing fast. The "training" phase just requires finding the Mean and Variance (Standard Deviation) of the columns. It takes milliseconds.</p>

          <h3>10. Major Disadvantages</h3>
          <p>The "Naive" assumption. It assumes every single feature is completely independent of every other feature. In the real world, this is almost never true.</p>

          <h3>11. Data Assumptions</h3>
          <p><strong>Gaussian Naive Bayes</strong> assumes your data points form a normal distribution (a bell curve). If your data is heavily skewed or clustered in weird rings, the math breaks down.</p>

          <h3>12. How it Overfits</h3>
          <p>It rarely overfits. In fact, it is considered a "high bias, low variance" model, meaning it is more prone to underfitting than overfitting.</p>

          <h3>13. How it Underfits</h3>
          <p>If the true decision boundary between classes is a sharp zigzag or a diagonal line, Naive Bayes will fail because it can only generate smooth, curved parabolas based on circles/ovals.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * d)</code> (Incredibly fast. Just calculates the Mean and Variance of columns).</li>
            <li><strong>Training Space:</strong> <code>O(c * d)</code> (Stores the Mean/Variance for each class <em>c</em>).</li>
            <li><strong>Prediction Time:</strong> <code>O(c * d)</code> (Calculates the Bell Curve probability for each class).</li>
            <li><strong>Prediction Space:</strong> <code>O(c * d)</code>.</li>
          </ul>

          <h3>15. Fun Fact</h3>
          <p>If a new point contains a feature value the model has never seen before, the probability collapses to literally 0.0. We fix this by using "Laplace Smoothing" (adding +1 to everything so nothing is ever absolute zero)!</p>
        `
      }
    ]
  },
  {
    category: "Unsupervised Learning",
    algorithms:[
      {
        id: "kmeans",
        title: "K-Means Clustering",
        path: "/kmeans",
        content: `
          <h3>1. What is it?</h3>
          <p>K-Means is an unsupervised clustering algorithm. You dump a massive pile of unlabeled data on a table, tell the algorithm to find 'K' number of groups, and it mathematically organizes the chaos into distinct, color-coded clusters.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine a teacher trying to divide 30 students into 3 study groups based on where they are sitting in a classroom. She drops 3 desks randomly in the room. Students walk to the closest desk. The teacher then moves the desks to the exact center of where the students are standing. The students readjust. This repeats until the desks stop moving!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Randomly drop 'K' centroids (stars) onto the map.</li>
            <li><strong>Step 2:</strong> Calculate the distance from every data point to every centroid. Assign each point to its closest centroid.</li>
            <li><strong>Step 3:</strong> Calculate the mathematical mean (center) of all the points assigned to a cluster.</li>
            <li><strong>Step 4:</strong> Move the centroid to that exact center point. Repeat until the centroids stop moving (Convergence).</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Within-Cluster Sum of Squares (Inertia):</strong> <code>Σ ||x_i - μ_j||²</code><br/>
          It simply calculates the physical straight-line (Euclidean) distance between a point and its cluster center, squares it, and adds them all up. The goal is to minimize this total distance!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Number of Clusters (K):</strong> You must manually tell the AI how many groups to look for.</li>
            <li><strong>Max Iterations:</strong> A safety cutoff in case the centroids get stuck in an infinite loop vibrating between two spots.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Silhouette Score:</strong> Ranges from -1 to 1. A score of 0.9 means clusters are tight and far apart from each other. A score near 0 means clusters are overlapping and confusing.</p>

          <h3>7. Best Use Cases</h3>
          <p>Customer segmentation (grouping shoppers by purchasing habits), grouping news articles by topic, or basic image compression (reducing a photo to 16 colors).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Weirdly shaped data. If your data looks like interlocking rings, concentric circles, or crescent moons, K-Means will fail miserably because it can only draw straight, rigid borders (Voronoi cells).</p>

          <h3>9. Major Advantages</h3>
          <p>Lightning fast, computationally cheap, and incredibly easy to explain to non-technical stakeholders.</p>

          <h3>10. Major Disadvantages</h3>
          <p>You have to blindly guess 'K' beforehand. It also assumes clusters are roughly spherical and similar in size, which ruins it for complex datasets.</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes variance is spherical (equal in all directions). Standardizing your data (Scaling X and Y) is completely mandatory, or distances will be warped.</p>

          <h3>12. How it Overfits</h3>
          <p>If you set K equal to the number of data points. Every single dot becomes its own cluster. Inertia hits exactly 0.0, but the model is utterly useless.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting K=1. Everything is grouped into one massive blob, ignoring all underlying structure.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * K * d * i)</code> (Where <em>i</em> = iterations. Extremely fast and scalable).</li>
            <li><strong>Training Space:</strong> <code>O(n * d + K * d)</code> (Stores the data and the K centroids).</li>
            <li><strong>Prediction Time:</strong> <code>O(K * d)</code> (Just measures distance to the final K centroids).</li>
            <li><strong>Prediction Space:</strong> <code>O(K * d)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Never initialize centroids purely randomly! Modern implementations use <strong>K-Means++</strong>, a math trick that forces the initial centroids to spawn as far away from each other as possible, preventing the algorithm from getting stuck in bad local minima.</p>
        `
      },
      {
        id: "dbscan",
        title: "DBSCAN Clustering",
        path: "/DBScan",
        content: `
          <h3>1. What is it?</h3>
          <p>DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. Unlike K-Means, you don't need to tell it how many clusters exist! It finds dense groups of points, expands them like a contagion, and completely ignores isolated "noise" points.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine a zombie outbreak. If a person is sick, they infect anyone within 6 feet of them (Epsilon). Those newly sick people infect anyone within 6 feet of them. The cluster expands. If someone lives alone in a cabin in the woods far away from everyone, they never get sick (Noise).</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Pick a random unvisited point. Draw a circle (Epsilon radius) around it.</li>
            <li><strong>Step 2:</strong> If it has enough neighbors (MinPoints) in that circle, it becomes a <strong>Core Point</strong> and starts a cluster.</li>
            <li><strong>Step 3:</strong> Recursively check the neighbors' neighbors, adding them to the cluster until the density dies out.</li>
            <li><strong>Step 4:</strong> Any point that isn't connected to a dense region is labeled as <strong>Noise</strong>.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Density Reachability:</strong> A point 'A' is density-reachable from 'B' if there is a chain of points connecting them, where every point in the chain is within the Epsilon radius of the previous one.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Epsilon (ε):</strong> The physical radius of the "search circle" around each point.</li>
            <li><strong>MinPoints:</strong> The minimum number of dots required inside the Epsilon circle to legally declare it a "dense region."</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Area Chart (Classification Progress):</strong> Notice how it doesn't jump randomly in the visualizer. The Area chart grows smoothly as the algorithm traces along the physical contours of the data, marking Cores and Borders until it hits empty space.</p>

          <h3>7. Best Use Cases</h3>
          <p>Geospatial data (finding high-density Uber pickups), anomaly detection (finding fraudulent bank transactions sitting far away as Noise), and datasets with bizarre, non-circular shapes (Moons/S-curves).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Datasets with varying densities. If Cluster A is incredibly packed together, and Cluster B is very loose and spread out, a single Epsilon value won't work for both. One will be merged, or the other will be classified entirely as noise.</p>

          <h3>9. Major Advantages</h3>
          <p>You do not need to guess 'K'. It automatically figures out how many clusters exist. It is also one of the only algorithms that explicitly identifies and isolates Outliers (Noise)!</p>

          <h3>10. Major Disadvantages</h3>
          <p>It is incredibly sensitive to hyperparameters. A tiny change in Epsilon can instantly merge two distinct clusters into one giant blob.</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes that clusters are dense regions separated by clear regions of low density. Struggles heavily with the "Curse of Dimensionality" in high-D space because everything becomes mathematically far apart.</p>

          <h3>12. How it Overfits</h3>
          <p>If you set Epsilon very small and MinPoints very high, the algorithm gets too strict. It will refuse to make clusters and will classify almost your entire dataset as "Noise."</p>

          <h3>13. How it Underfits</h3>
          <p>If you set Epsilon very large, the search circle bridges the gap between different clusters, swallowing the entire map into a single massive Cluster 0.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * log(n))</code> with a spatial index (like KD-Tree), but <code>O(n²)</code> without one (Very slow for large datasets).</li>
            <li><strong>Training Space:</strong> <code>O(n * d)</code>.</li>
            <li><strong>Prediction Time:</strong> N/A (Transductive. Cannot predict new points without recalculating the whole map).</li>
            <li><strong>Prediction Space:</strong> N/A.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Senior engineers don't guess Epsilon. They plot a "K-Distance Graph" (distance to the k-th nearest neighbor for all points) and look for the mathematical "elbow" in the curve to find the exact perfect Epsilon radius!</p>
        `
      },
      {
        id: "pca",
        title: "Principal Component Analysis (PCA)",
        path: "/pca",
        content: `
          <h3>1. What is it?</h3>
          <p>PCA is the king of Dimensionality Reduction. It mathematically crushes massive datasets (like a 1000-pixel image) down into a tiny handful of variables (like 10 numbers) while retaining 99% of the original information!</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine shining a flashlight on a 3D teapot to cast a 2D shadow on a wall. If you point the light directly at the spout, the shadow just looks like a weird flat circle. But if you angle the light perfectly from the side, the shadow perfectly preserves the shape of the handle, lid, and spout. PCA is the math that finds that perfect "angle" (variance).</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Find the center of the dataset and move everything so the center is at (0,0).</li>
            <li><strong>Step 2:</strong> Calculate a Covariance Matrix to see how all the features correlate with each other.</li>
            <li><strong>Step 3:</strong> Perform Eigen-Decomposition to find the direction where the data is most stretched out (Principal Component 1).</li>
            <li><strong>Step 4:</strong> Find the next most stretched direction that is perfectly 90-degrees (orthogonal) to the first.</li>
            <li><strong>Step 5:</strong> Project the data onto these new lines!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Covariance Matrix:</strong> <code>C = (X^T * X) / (n-1)</code><br/>
          <strong>Eigen-Decomposition:</strong> <code>C * v = λ * v</code>. The Eigenvector (v) tells you the direction of the line. The Eigenvalue (λ) tells you how much information (variance) is captured on that line!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Number of Components:</strong> How many dimensions to keep. You usually select enough components to keep 95% of the total variance.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Explained Variance Ratio:</strong> This tells you how much "meaning" was retained. If crushing 3D data down to 2D gives you an Explained Variance of 98%, it means the 3rd dimension was mostly useless noise anyway!</p>

          <h3>7. Best Use Cases</h3>
          <p>Speeding up slow Machine Learning algorithms. If you have a dataset with 5,000 columns (like genetics), running an SVM will take days. PCA can crush it to 50 columns in seconds, letting the SVM train instantly with the exact same accuracy.</p>

          <h3>8. Worst Use Cases</h3>
          <p>When you need strict interpretability. PCA invents entirely new axes. If you pass in "Age" and "Income", PCA destroys them and creates "PC1" and "PC2", which represent a blended mathematical fraction of both. You can no longer explain it simply to a business client.</p>

          <h3>9. Major Advantages</h3>
          <p>Eliminates Multicollinearity (correlated features). If you have a dataset with "Price in Dollars" and "Price in Euros", PCA recognizes they are the same thing and instantly merges them into a single component.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It is strictly linear. If your data is arranged in a curved Swiss Roll or a circle, PCA can only draw a straight line through it, failing to unroll or understand the curved structure.</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes that Large Variance = Important Structure. It also strictly assumes the features have a linear relationship.</p>

          <h3>12. How it Overfits</h3>
          <p>Keeping 100% of the Principal Components. You haven't reduced the dimensionality at all; you've just rotated the dataset and kept all the noisy, useless variables.</p>

          <h3>13. How it Underfits</h3>
          <p>Crushing a massive 1000-dimensional dataset down to a 1D line. You will lose 99% of the variance, and the resulting data will be a useless, overlapping pile of mush.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(d³ + n * d²)</code> (Calculating and performing Eigen-decomposition on the Covariance matrix).</li>
            <li><strong>Training Space:</strong> <code>O(d²)</code> (Stores the Covariance matrix).</li>
            <li><strong>Prediction Time:</strong> <code>O(k * d)</code> (Where <em>k</em> is the number of kept components. Just a matrix multiplication!).</li>
            <li><strong>Prediction Space:</strong> <code>O(k * d)</code> (Stores the eigenvectors).</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Look at the <strong>Scree Plot</strong> in the dashboard! The height of the bar is the Eigenvalue. To find the optimal number of dimensions to keep, look for the "Elbow" in the plot, keep the tall bars and throw away the short ones!</p>
        `
      },
      {
        id: "gmm",
        title: "Gaussian Mixture Models (GMM)",
        path: "/gmm",
        content: `
          <h3>1. What is it?</h3>
          <p>GMM is "K-Means on steroids." While K-Means draws rigid, circular boundaries, GMM places soft, probabilistic Bell Curves (Gaussians) over the data that can stretch, rotate, and overlap to fit incredibly complex shapes!</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine mapping the physical height of athletes. Gymnasts are short, basketball players are tall, but there is overlap in the middle. Instead of drawing a hard line and saying "everyone over 6-foot is NBA", GMM draws overlapping probability curves. A 6'1" person might be 80% likely to be an NBA point guard, and 20% likely to be a very tall gymnast.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Randomly drop K number of Gaussian ellipses onto the map.</li>
            <li><strong>Step 2 (Expectation):</strong> For every point, calculate the probability that it belongs to each ellipse.</li>
            <li><strong>Step 3 (Maximization):</strong> Physically shift, stretch, and rotate the ellipses to maximize those probabilities.</li>
            <li><strong>Step 4:</strong> Repeat the E-M steps until the ellipses stop changing shape (Convergence).</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Expectation-Maximization (EM):</strong> It attempts to maximize the <strong>Log-Likelihood</strong> of the data. Because it uses Multivariate Normal Distributions, it updates the Mean (center), Covariance (stretch/rotation), and Weight (size) of each Gaussian on every iteration.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Number of Components:</strong> How many Gaussians to use.</li>
            <li><strong>Covariance Type:</strong> 'Spherical' acts exactly like K-Means. 'Full' allows the ellipses to stretch diagonally into ovals!</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Log-Likelihood:</strong> You want to maximize this! A higher Log-Likelihood means the mathematical ellipses are a highly accurate representation of the actual data distribution.</p>

          <h3>7. Best Use Cases</h3>
          <p>Any clustering problem where "Hard Clustering" isn't good enough. If you need to know exactly how confident the AI is about an assignment (e.g., Anomaly Detection where low-probability points are flagged as fraud).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Massive datasets with extremely high dimensions. The EM algorithm involves heavy matrix inversions for the Covariance, making it computationally brutal compared to K-Means.</p>

          <h3>9. Major Advantages</h3>
          <p>Incredible flexibility. The overlapping probabilities mean a point isn't forced into a cluster it barely belongs to; it retains a shared probabilistic relationship with all nearby clusters.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It is incredibly sensitive to the starting position. If the Gaussians initialize in a bad spot, the EM algorithm will get stuck in a "Local Minimum" and confidently draw ellipses in completely wrong places.</p>

          <h3>11. Data Assumptions</h3>
          <p>It assumes the entire dataset was literally generated by combining several hidden Gaussian distributions.</p>

          <h3>12. How it Overfits</h3>
          <p>Setting the Number of Components too high. The algorithm will create tiny, microscopic Gaussians that surround individual outlier data points, memorizing the noise.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting Components to 1. It just draws one massive, useless ellipse covering the entire screen.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n * K * d³ * i)</code> (The <em>d³</em> comes from inverting Covariance matrices. Brutally slow on high-dimensional data).</li>
            <li><strong>Training Space:</strong> <code>O(K * d²)</code> (Stores a full covariance matrix for every cluster).</li>
            <li><strong>Prediction Time:</strong> <code>O(K * d²)</code>.</li>
            <li><strong>Prediction Space:</strong> <code>O(K * d²)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Look at the <strong>AIC / BIC Curve</strong> in the dashboard! AIC and BIC mathematically penalize the model for adding too many Gaussians. The lowest point on the BIC curve mathematically proves the exact number of true clusters in the dataset!</p>
        `
      },
      ,{
        id: "hierarchical",
        title: "Hierarchical Clustering",
        path: "/hierarchical",
        content: `
          <h3>1. What is it?</h3>
          <p>Hierarchical Clustering builds a literal family tree (Dendrogram) of your data. Instead of randomly guessing clusters like K-Means, it starts by assuming every single dot is its own cluster, and then repeatedly merges the closest dots together until only one giant cluster remains.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Think of evolutionary biology. You start with individual species. Closely related species (dogs and wolves) merge into a genus. Those merge with foxes into a family (Canidae), which merges with cats and bears into an order (Carnivora). The algorithm draws this exact tree!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Calculate the distance between every single dot and every other dot.</li>
            <li><strong>Step 2:</strong> Find the two closest dots and permanently merge them into a "Cluster".</li>
            <li><strong>Step 3:</strong> Recalculate distances. (How far is a dot from a newly formed cluster?).</li>
            <li><strong>Step 4:</strong> Repeat until every dot is absorbed into a single root tree. Slice the tree at a certain height to get your final clusters.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Linkage Criteria:</strong> How do you measure the distance between two groups of dots?<br/>
          <strong>Ward's Method:</strong> Minimizes the variance of merged clusters. <code>ΔV = V_merged - (V_a + V_b)</code><br/>
          <strong>Complete Linkage:</strong> Measures the distance between the two <em>furthest</em> points in the clusters.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Target Clusters (The Cutoff):</strong> You mathematically slice the Dendrogram horizontally. A lower slice creates many small clusters; a higher slice creates few large clusters.</li>
            <li><strong>Linkage Type:</strong> Changes the math of how clusters attract each other (Ward, Average, Complete, Single).</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>The Dendrogram:</strong> The height of the U-shaped lines represents the mathematical distance between the clusters being merged. A very tall vertical line means the two clusters were incredibly far apart and very dissimilar!</p>

          <h3>7. Best Use Cases</h3>
          <p>Taxonomy (biology), genetic sequencing, and any dataset where the hierarchical relationship is just as important as the final clusters.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Big Data. Because it has to measure the distance from every dot to every other dot, it requires a massive distance matrix. Running this on 100,000 rows will instantly run out of RAM and crash your computer.</p>

          <h3>9. Major Advantages</h3>
          <p>You do not need to guess 'K' before running the algorithm! You run it once, look at the beautiful tree, and then decide where to chop it to get the perfect clusters.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It is strictly irreversible. If the algorithm makes a bad merge early on, those two dots are glued together forever and cannot be separated later.</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes that distance metrics (like Euclidean distance) correctly capture the similarity between your data points. Scaling the data is absolutely mandatory.</p>

          <h3>12. How it Overfits</h3>
          <p>Setting the Target Clusters too high (e.g., 50 clusters for 100 points). You just end up capturing noise instead of underlying structure.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting Target Clusters to 1. Everything merges into the root node, completely ignoring all subgroups.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n³)</code> (Calculates distances between every merged cluster. Fails on big data).</li>
            <li><strong>Training Space:</strong> <code>O(n²)</code> (Must hold the massive N x N distance matrix in RAM).</li>
            <li><strong>Prediction Time:</strong> N/A (Does not predict new points).</li>
            <li><strong>Prediction Space:</strong> N/A.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>If your clusters look like long, stringy snakes, you are using "Single Linkage" (which causes the Chaining Effect). Switch to "Ward Linkage" to force the algorithm to build tight, circular, spherical clusters!</p>
        `
      },
      {
        id: "tsne",
        title: "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
        path: "/tsne",
        content: `
          <h3>1. What is it?</h3>
          <p>t-SNE is a magical Dimensionality Reduction tool. While PCA tries to draw straight lines to crush data, t-SNE acts like a physics engine. It takes complex 3D (or 100D) shapes, maps the gravitational pull between neighbors, and lets the points physically push and pull each other until they settle flat on a 2D piece of paper.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine trying to make a flat 2D map of a 3D Earth. If you just squash it (PCA), Africa and South America get mangled. t-SNE mathematically unpeels the globe, ensuring that countries that are close to each other in 3D stay close to each other on the flat 2D paper.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Look at the 3D space. Calculate the probability that Point A would pick Point B as its neighbor.</li>
            <li><strong>Step 2:</strong> Drop all points randomly onto a flat 2D map.</li>
            <li><strong>Step 3:</strong> Calculate the same neighbor probabilities in the 2D map using a Heavy-Tailed Student-t distribution.</li>
            <li><strong>Step 4:</strong> Use Gradient Descent to physically move the 2D dots around until the 2D probabilities perfectly match the 3D probabilities!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Kullback-Leibler (KL) Divergence:</strong> <code>KL(P || Q) = Σ P_i * log(P_i / Q_i)</code><br/>
          It mathematically measures how badly the 2D map (Q) failed to capture the true 3D relationships (P). The algorithm calculates gradients to minimize this KL loss.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Perplexity:</strong> This is a knob that controls the "Gravity" of the points. It dictates how many close neighbors each point cares about. Low perplexity focuses entirely on local micro-clusters. High perplexity forces the AI to look at the global macro-shape.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Final KL Divergence:</strong> Shown in the Loss Curve. You will watch it plunge as the 2D points magically unravel themselves (like unrolling the 3D Swiss Roll) to match the original structure.</p>

          <h3>7. Best Use Cases</h3>
          <p>Visualizing incredibly high-dimensional datasets. It is the industry standard for plotting Neural Network embeddings (like grouping millions of facial recognition vectors on a 2D plot to see if they cluster correctly).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Clustering or Predictive Modeling. You should <strong>never</strong> run K-Means on top of a t-SNE plot, and you cannot use t-SNE to predict where a new incoming point should land.</p>

          <h3>9. Major Advantages</h3>
          <p>It is highly non-linear. Unlike PCA, it can easily unroll complex manifolds like spheres, S-curves, and twisted knots.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It is computationally massive, slow, and completely stochastic (random). If you run t-SNE twice on the exact same data, the resulting map might be flipped or rotated differently!</p>

          <h3>11. Data Assumptions</h3>
          <p>t-SNE purposely distorts global distances to preserve local distances. This means if Cluster A and Cluster B look far apart on the 2D map, it does NOT mean they are actually far apart in 3D space. Distance on a t-SNE plot is an illusion!</p>

          <h3>12. How it Overfits</h3>
          <p>Setting Perplexity too low (e.g., 2). The algorithm only cares about immediate neighbors, causing the data to fragment into dozens of tiny, fake, meaningless micro-clusters.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting Perplexity too high. The algorithm cares too much about the global shape and mashes all the distinct clusters into one giant, unreadable blob.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(n²)</code> originally, but modern Barnes-Hut approximations drop it to <code>O(n * log(n))</code>.</li>
            <li><strong>Training Space:</strong> <code>O(n * log(n))</code>.</li>
            <li><strong>Prediction Time:</strong> N/A (t-SNE cannot project new points into an existing map).</li>
            <li><strong>Prediction Space:</strong> N/A.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Always run PCA first! If you have 1000 dimensions, run PCA to crush it to 50 dimensions, and THEN run t-SNE to crush it to 2. This drastically reduces the noise and speeds up the math.</p>
        `
      },
      {
        id: "apriori",
        title: "Apriori (Association Rules)",
        path: "/apriori",
        content: `
          <h3>1. What is it?</h3>
          <p>Apriori is the "Shopping Cart" algorithm. It scans massive databases of transaction receipts to find hidden relationships and generate absolute "If-Then" rules (e.g., If they buy Peanut Butter and Jelly, they will buy Bread).</p>

          <h3>2. Real-World Analogy</h3>
          <p>Amazon's "Frequently Bought Together" section. Amazon doesn't have an AI predicting what you want; it literally just scanned 10 million past orders using Apriori and found that 85% of people who bought a flashlight also bought batteries.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Scan all receipts and delete items that are rarely bought (Min Support cutoff).</li>
            <li><strong>Step 2:</strong> Look at the surviving items and create 2-item pairs (e.g., Milk & Eggs). Delete pairs that are rarely bought together.</li>
            <li><strong>Step 3:</strong> Create 3-item groups. Delete rare ones. Repeat until no groups are left.</li>
            <li><strong>Step 4:</strong> Convert surviving groups into Rules (Milk -> Eggs) based on Confidence thresholds.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Support:</strong> P(A ∩ B) — What percentage of all receipts contain both A and B?<br/>
          <strong>Confidence:</strong> P(A ∩ B) / P(A) — Out of all the people who bought A, what percentage ALSO bought B?</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Minimum Support:</strong> The popularity threshold. High support = only look at mainstream items.</li>
            <li><strong>Minimum Confidence:</strong> The reliability threshold. High confidence = rules must be 90%+ accurate to be displayed.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Lift:</strong> The most important metric! Lift = Confidence / P(B). If Lift > 1, buying A actually *increases* the chance of buying B. If Lift = 1, the items are completely independent. If Lift < 1, buying A makes you *less* likely to buy B!</p>

          <h3>7. Best Use Cases</h3>
          <p>Retail store layout optimization (putting Cerelac next to Diapers), Medical symptom correlations (If symptom A and B, then Disease C), and cross-selling in E-Commerce.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Continuous numerical data (like Stock Prices or Temperatures). Apriori only works on categorical "Bucket/Basket" data.</p>

          <h3>9. Major Advantages</h3>
          <p>The rules are 100% human-readable and mathematically factual. You don't need a PhD to explain to a CEO why "Bread -> Butter" is a good rule.</p>

          <h3>10. Major Disadvantages</h3>
          <p>Computationally explosive. If Walmart has 100,000 unique items in a store, the number of possible item pairs and triplets is exponentially massive. It can freeze servers.</p>

          <h3>11. Data Assumptions</h3>
          <p>It relies entirely on the <strong>Downward Closure Property</strong>: The core assumption that if a pair of items (Milk & Bread) is infrequent, then any larger group containing them (Milk, Bread, & Eggs) MUST mathematically be infrequent too!</p>

          <h3>12. How it Overfits</h3>
          <p>Setting Minimum Support to 1%. The algorithm will analyze billions of rare, random coincidences and generate completely useless rules like "If buy Motor Oil -> Buy Banana."</p>

          <h3>13. How it Underfits</h3>
          <p>Setting Minimum Support to 99%. Almost no two items in the world appear together on 99% of all global receipts. The algorithm will return 0 rules.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(2^d)</code> (Exponentially horrifying. Evaluating every possible item combination will crash computers if min_support is too low).</li>
            <li><strong>Training Space:</strong> <code>O(2^d)</code> (To store the candidate itemsets).</li>
            <li><strong>Prediction Time:</strong> N/A (Generates static database rules, not predictions).</li>
            <li><strong>Prediction Space:</strong> <code>O(R)</code> (Where R is the number of rules generated).</li>
          </ul>
          <h3>15. Fun Fact</h3>
          <p>In the 1990s, an Apriori analysis on a grocery chain famously discovered that young fathers buying Diapers on Friday nights were highly likely to also buy Alcohol. The store moved the Alcohol aisle next to the diapers and sales skyrocketed!</p>
        `
      }
    ]
  },
  {
    category: "Ensemble Learning",
    algorithms:[
      {
        id: "rf",
        title: "Random Forest",
        path: "/rf",
        content: `
          <h3>1. What is it?</h3>
          <p>A single Decision Tree is like asking one person for advice; they might be biased. Random Forest is like asking a crowd of 500 diverse people for advice and taking a majority vote. It prevents overfitting by using the "Wisdom of the Crowds."</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine guessing the number of jellybeans in a jar. If one person guesses, they might be wildly wrong. But if you ask 100 random people and average their guesses, the final answer is almost always incredibly accurate. That is exactly what this algorithm does.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1 (Bootstrapping):</strong> Create 100 random mini-datasets by blindly sampling the original data with replacement.</li>
            <li><strong>Step 2 (Feature Randomness):</strong> Train a Decision Tree on each mini-dataset, but mathematically restrict them so they can only look at a random subset of the features (e.g., X1, but not X2).</li>
            <li><strong>Step 3:</strong> Let all 100 trees grow to their maximum depth.</li>
            <li><strong>Step 4 (Aggregation):</strong> For a new prediction, let all 100 trees vote. The most popular class wins!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Bagging (Bootstrap Aggregating):</strong> <code>Final Prediction = (1/B) * Σ f_b(x)</code>.<br/>
          By mathematically forcing the trees to be uncorrelated (different data, different features), the formula completely destroys the Variance (overfitting) of the model without hurting the Bias.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Number of Estimators (Trees):</strong> Adding more trees improves accuracy up to a point, then it just plateaus. It never hurts to add more trees (except for slowing down your computer).</li>
            <li><strong>Max Depth:</strong> Limits how deep the individual trees can grow to prevent them from memorizing heavy noise.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Feature Importance:</strong> Look at the Bar Chart in the visualizer! The AI calculates exactly how much Gini Impurity was destroyed by splitting on X1 versus X2 across all 100 trees. It tells you exactly which variable matters the most to your business!</p>

          <h3>7. Best Use Cases</h3>
          <p>Tabular data (Excel spreadsheets), banking fraud detection, medical diagnosis. It is widely considered the best out-of-the-box algorithm in all of Machine Learning.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Extrapolating outside the training data, and Computer Vision (CNNs are better) or Text Analysis (Transformers are better).</p>

          <h3>9. Major Advantages</h3>
          <p>It is almost immune to overfitting. It requires zero data scaling. It ignores useless variables natively. It handles missing data beautifully.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It is a "Black Box." Unlike a single Decision Tree where you can look at the flowchart, you cannot humanly read the logic of 500 overlapping trees. It lacks interpretability.</p>

          <h3>11. Data Assumptions</h3>
          <p>Makes virtually zero assumptions about data distribution, making it incredibly robust.</p>

          <h3>12. How it Overfits</h3>
          <p>It is very hard to overfit a Random Forest. However, if your Max Depth is infinite and you have too few estimators on a highly noisy dataset, the noise can survive the averaging process.</p>

          <h3>13. How it Underfits</h3>
          <p>If you restrict the Max Depth of the trees to 1 or 2, the individual trees are too stupid to learn anything. Averaging the votes of 100 stupid trees just gives you a stupid answer.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(M * n * d * log(n))</code> (Where <em>M</em> = number of trees. Can be easily parallelized on multiple CPU cores).</li>
            <li><strong>Training Space:</strong> <code>O(M * nodes)</code> (Stores hundreds of trees in RAM).</li>
            <li><strong>Prediction Time:</strong> <code>O(M * depth)</code> (Passes the data down M trees and tallies the votes).</li>
            <li><strong>Prediction Space:</strong> <code>O(M * nodes)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Random Forests calculate something called "Out-of-Bag (OOB) Error". Because each tree only trains on ~66% of the data, you can use the remaining 33% to test that tree. It basically gives you a free Validation Score without having to do a Train/Test split!</p>
        `
      },
      {
        id: "adaboost",
        title: "AdaBoost (Adaptive Boosting)",
        path: "/adaboost",
        content: `
          <h3>1. What is it?</h3>
          <p>If Random Forest is a committee of experts working independently, AdaBoost is a single student taking a test over and over again, focusing exclusively on the flashcards they got wrong the last time.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine preparing for a Math exam. On Day 1, you take a practice test. You get Algebra right but fail Calculus. On Day 2, you tell your tutor to multiply the importance (weight) of Calculus questions. You take the test again. You fail Geometry. On Day 3, you multiply Geometry's weight. By exam day, you have adapted to all your weaknesses!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Train a very weak, shallow model (a "Stump" - a tree with a depth of 1).</li>
            <li><strong>Step 2:</strong> Evaluate the stump. Find all the data points it got wrong.</li>
            <li><strong>Step 3:</strong> Mathematically increase the "Sample Weight" of those misclassified points so they look massive to the algorithm.</li>
            <li><strong>Step 4:</strong> Train a new stump that is forced to focus on the heavy points. Repeat sequentially!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Exponential Loss:</strong> <code>L(y, f(x)) = exp(-y * f(x))</code><br/>
          <strong>Alpha (Tree Weight):</strong> <code>α_t = 0.5 * ln((1 - error_t) / error_t)</code>. Highly accurate stumps get a high voting power in the final equation. Stumps that did worse than a coin-flip get negative voting power!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Estimators (Stumps):</strong> Because they are trained sequentially, setting this to 100 means you wait for 100 corrections.</li>
            <li><strong>Learning Rate:</strong> A fractional multiplier (e.g., 0.1) that shrinks the contribution of each new stump. A low learning rate requires more estimators, but prevents the model from over-correcting too wildly.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Sequential Boundary Evolution:</strong> Look at the Animation in the visualizer! You will see it literally draw straight lines (Stumps). Then it draws another straight line to fix the errors of the first. Slowly, those straight lines combine into a beautifully complex curve.</p>

          <h3>7. Best Use Cases</h3>
          <p>Binary classification problems like Face Detection (the famous Viola-Jones face detector that powered digital cameras in the 2000s relied entirely on AdaBoost).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Extremely noisy datasets. Because AdaBoost mathematically amplifies the weight of points it gets wrong, a completely wrong "Noise" point will be amplified infinitely, tricking the algorithm into obsessing over an outlier.</p>

          <h3>9. Major Advantages</h3>
          <p>It mathematically proves that a sequence of "Weak Learners" (models that are barely better than a 50/50 coin flip) can be combined to form an invincible "Strong Learner."</p>

          <h3>10. Major Disadvantages</h3>
          <p>Cannot be parallelized! Because Tree #2 mathematically requires the errors of Tree #1 to even begin training, you cannot split this across a 16-core CPU. It must train sequentially.</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes the data has extremely low noise. Outliers will utterly destroy AdaBoost's performance.</p>

          <h3>12. How it Overfits</h3>
          <p>Running too many estimators on noisy data. The model will eventually build highly specific rules just to capture the 3 random noise dots it couldn't solve in the first 100 epochs.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting the Estimators to 1. It will literally just draw a single straight line on the map (a stump).</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(M * n * d)</code> (Cannot be parallelized! Tree 2 must wait for Tree 1 to finish finding errors).</li>
            <li><strong>Training Space:</strong> <code>O(M * nodes)</code>.</li>
            <li><strong>Prediction Time:</strong> <code>O(M)</code> (Since it uses depth=1 stumps, prediction is near-instant).</li>
            <li><strong>Prediction Space:</strong> <code>O(M)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>There is an iron-clad mathematical relationship between Learning Rate and Number of Estimators in boosting. If you cut the Learning Rate in half (from 1.0 to 0.5), you generally must double the Number of Estimators to get the same level of training!</p>
        `
      },
      ,{
        id: "gb",
        title: "Gradient Boosting",
        path: "/gb",
        content: `
          <h3>1. What is it?</h3>
          <p>Gradient Boosting is arguably the most powerful algorithm for tabular data on Earth (powering XGBoost and LightGBM). It builds a sequence of trees, but instead of altering sample weights like AdaBoost, it forces every new tree to mathematically predict the exact Errors (Residuals) of the previous tree!</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine playing mini-golf. Your first swing (Tree 1) gets the ball close to the hole, but you miss by 5 feet. Your second swing (Tree 2) doesn't aim for the hole—it specifically aims to cover that missing 5 feet. Your third swing (Tree 3) covers the remaining 2 inches. Added together, your swings perfectly hit the target.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> The model makes a basic, average guess for every data point.</li>
            <li><strong>Step 2:</strong> It calculates the <em>Residuals</em> (Actual Value - Predicted Value).</li>
            <li><strong>Step 3:</strong> It trains a brand new Decision Tree where the "Target" is no longer the actual data, but the Residual Errors!</li>
            <li><strong>Step 4:</strong> It adds the new tree's predictions to the old predictions, shrinking the errors. Repeat!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The Update Rule:</strong> <code>F_m(x) = F_{m-1}(x) + ν * h_m(x)</code><br/>
          The new model <code>F_m</code> is equal to the old model, plus the new tree <code>h_m</code> scaled down by a Learning Rate <code>ν</code>. It is literally performing Gradient Descent, but in <em>Function Space</em> instead of Parameter Space!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Learning Rate:</strong> How much we trust each new tree. A high rate means the first few trees take massive swings. A low rate means we take hundreds of tiny, highly accurate putts.</li>
            <li><strong>Total Estimators:</strong> How many trees to add to the sequence.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Ensemble Learning Curve:</strong> Notice the gap between the Training Loss and Validation Loss in the visualization. Unlike Random Forest, Gradient Boosting <em>can</em> easily overfit. As you add more trees, the Training Error will eventually hit 0.0, but the Validation Error will U-turn and get worse!</p>

          <h3>7. Best Use Cases</h3>
          <p>Winning Kaggle competitions. Anytime you have highly structured spreadsheet data (Finance, Real Estate, User Churn), Gradient Boosting will almost always defeat Deep Learning Neural Networks.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Unstructured data like Raw Images, Audio, or Text. (You need CNNs and Transformers for that).</p>

          <h3>9. Major Advantages</h3>
          <p>Unmatched predictive accuracy. By combining the flexibility of trees with the optimization of Calculus (Gradient Descent), it molds perfectly to complex, non-linear data without requiring feature scaling.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It cannot be parallelized. Tree #50 cannot be built until Tree #49 finishes calculating its exact mathematical errors. This makes it much slower to train than a Random Forest.</p>

          <h3>11. Data Assumptions</h3>
          <p>Because it uses Decision Trees as its base, it doesn't assume any linear relationships and handles missing data or severe outliers better than standard regression.</p>

          <h3>12. How it Overfits</h3>
          <p>Too many trees! Because it relentlessly hunts down errors, if you let it run for 1,000 estimators, it will eventually build trees dedicated to predicting the random background noise in your dataset.</p>

          <h3>13. How it Underfits</h3>
          <p>If you use a tiny Learning Rate (e.g., 0.001) but only give it 10 Estimators. The trees take such microscopic steps toward the target that the sequence ends before it ever gets close to a good prediction.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(M * n * d * log(n))</code> (Slower than Random Forest because trees must be built sequentially to target residuals).</li>
            <li><strong>Training Space:</strong> <code>O(M * nodes)</code>.</li>
            <li><strong>Prediction Time:</strong> <code>O(M * depth)</code>.</li>
            <li><strong>Prediction Space:</strong> <code>O(M * nodes)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>There is a delicate mathematical dance between Learning Rate and Estimators. If you divide your Learning Rate by 2, you must double your Estimators to achieve the exact same boundary shape, but it will be much smoother and generalize better!</p>
        `
      }
    ]
  },
  {
    category: "Reinforcement Learning",
    algorithms:[
      {
        id: "ql",
        title: "Q-Learning (Tabular RL)",
        path: "/ql",
        content: `
          <h3>1. What is it?</h3>
          <p>Q-Learning is how an AI learns to play a video game. Instead of looking at a static dataset, an Agent explores a dynamic environment, makes decisions, hits walls (gets punished), finds goals (gets rewarded), and memorizes the best actions in a giant cheat sheet called a Q-Table.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine dropping a mouse in a maze with a piece of cheese. The first time, the mouse wanders blindly for an hour before finding the cheese. The second time, it remembers a few bad turns and takes 40 minutes. After 500 attempts, the mouse sprints straight to the cheese in 10 seconds. That is exactly what this visualizer does.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Build a blank spreadsheet (Q-Table) with rows for every tile on the map, and 4 columns (Up, Down, Left, Right) filled with zeros.</li>
            <li><strong>Step 2:</strong> The agent takes an action. If it hits a wall, it receives a -5 Reward.</li>
            <li><strong>Step 3:</strong> The agent updates its spreadsheet: "Tile (3,4), Action 'Up' = Terrible Idea."</li>
            <li><strong>Step 4:</strong> Through thousands of episodes, the spreadsheet fills up with the mathematically optimal moves for every single tile.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The Bellman Equation:</strong> <code>New Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s', a')) - Q(s,a)]</code><br/>
          This brilliant formula updates the current action's value by looking at the immediate reward (R) PLUS the highest possible reward of the <em>next</em> state it lands in!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Discount Factor (Gamma γ):</strong> If Gamma=0, the AI is greedy and only cares about immediate 1-step rewards. If Gamma=0.99, the AI plays 4D chess and cares about long-term future rewards.</li>
            <li><strong>Exploration Rate (Epsilon ε):</strong> If Epsilon=10%, the AI rolls a 10-sided dice. If it lands on 1, it ignores the Q-Table and takes a totally random move just to see what happens.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>The Q-Table Heatmap:</strong> Look at the final dashboard. The white arrows are the final "Brain" of the AI. You can drop the agent on ANY tile, and it will just follow the arrows straight to the goal.</p>

          <h3>7. Best Use Cases</h3>
          <p>Solving discrete puzzles, simple pathfinding, and board games like Tic-Tac-Toe or Checkers where the number of possible board states is relatively small.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Continuous environments like driving a real car, or games like Chess. If the board has millions of possible combinations, the Q-Table spreadsheet becomes so massive it exceeds all the RAM on Earth (State-Space Explosion).</p>

          <h3>9. Major Advantages</h3>
          <p>It is mathematically proven to converge to the absolute perfect optimal policy, given enough time to explore every state-action combination.</p>

          <h3>10. Major Disadvantages</h3>
          <p>It suffers fatally from the "Curse of Dimensionality." It cannot generalize. If it learns how to solve Maze A, and you move one wall, the Q-Table is useless and it must relearn the entire map from scratch.</p>

          <h3>11. Data Assumptions</h3>
          <p>It strictly assumes the environment is a Markov Decision Process (MDP)—meaning the outcome of an action depends ONLY on the current state, not on the history of how the agent got there.</p>

          <h3>12. How it Fails (Over-Exploitation)</h3>
          <p>If Epsilon is set to 0.0, the agent takes the first path it finds to the goal and locks it in. It will never try taking an alternate route, meaning it might miss a massive shortcut.</p>

          <h3>13. How it Fails (Over-Exploration)</h3>
          <p>If Epsilon is set to 1.0 (100%), the agent ignores all of its learning and just wanders around taking random moves forever like a drunkard.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Episodes * Steps)</code> (Scales with the exploration rate of the agent).</li>
            <li><strong>Training Space:</strong> <code>O(S * A)</code> (Where <em>S</em>=States and <em>A</em>=Actions. The Q-Table crashes your RAM if the game is too complex).</li>
            <li><strong>Prediction Time:</strong> <code>O(1)</code> (Literally just looking up a number in a spreadsheet).</li>
            <li><strong>Prediction Space:</strong> <code>O(S * A)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Watch the <strong>Reward Convergence Curve</strong>. It always starts in the deep negatives because the dumb agent is constantly crashing into walls. As it learns to avoid walls and beeline for the goal, the reward physically skyrockets onto the graph!</p>
        `
      },
      ,{
        id: "dqn",
        title: "Deep Q-Networks (DQN)",
        path: "/dqn",
        content: `
          <h3>1. What is it?</h3>
          <p>Deep Q-Networks (DQN) combine the trial-and-error of Reinforcement Learning with the pattern-recognition power of Deep Learning. Instead of memorizing a massive spreadsheet of every possible move (like basic Q-Learning), DQN uses a Neural Network to intuitively <em>understand</em> physics and predict the best move in real-time!</p>

          <h3>2. Real-World Analogy</h3>
          <p>Standard Q-Learning is like memorizing the answers to a specific math test; if the numbers change, you fail. DQN is like learning the actual formulas. Once you understand the physics of balance, you can balance the CartPole no matter what angle or speed it starts at!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> The environment feeds the AI raw physical states (Cart Position, Velocity, Pole Angle, Angular Velocity).</li>
            <li><strong>Step 2:</strong> The Neural Network crunches these numbers and outputs the predicted "Q-Values" (the expected future reward) for pushing Left vs. pushing Right.</li>
            <li><strong>Step 3:</strong> The AI takes the action with the highest Q-Value.</li>
            <li><strong>Step 4:</strong> It measures its actual success (Reward) against what it predicted, calculates the mathematical Error (Loss), and Backpropagates to make its brain smarter.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>DQN Loss Function:</strong> <code>Loss = [ (Reward + γ * max Q(s', a')) - Q(s, a) ]²</code><br/>
          This is Mean Squared Error (MSE). The Neural Network is trying to minimize the difference between its current prediction <code>Q(s, a)</code> and the actual reality of the reward it received.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Epsilon Decay Rate:</strong> Controls the "Exploration vs Exploitation" trade-off. It dictates how fast the AI stops experimenting with random moves and starts trusting its Neural Network.</li>
            <li><strong>Discount Factor (Gamma):</strong> Forces the AI to care about keeping the pole up for 100 steps in the future, rather than just surviving the next 1 step.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Loss vs. Reward Curve:</strong> Look at the graph in the visualizer. Early on, the Reward is near zero (the pole falls instantly). Then, the Neural Network Loss massively spikes! Why? Because the AI confidently predicted it would survive, and it died. That massive Error forces the network to update its weights rapidly, causing the Reward to skyrocket shortly after!</p>

          <h3>7. Best Use Cases</h3>
          <p>Continuous state spaces. Self-driving cars, robotic arm manipulation, and beating human champions at complex video games (like Atari and Dota 2).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Simple, grid-based board games (like Tic-Tac-Toe or Checkers). Using a Neural Network for a simple game is massive overkill and takes 100x longer to train than a simple Q-Table.</p>

          <h3>9. Major Advantages</h3>
          <p>It can generalize! A DQN can encounter a completely unique physical state it has never seen before and still mathematically guess the correct action by interpolating its past experiences.</p>

          <h3>10. Major Disadvantages</h3>
          <p>Highly unstable training. Because the AI is generating its own training data as it moves, a string of bad luck can completely overwrite its good memories, causing the AI to suddenly "forget" how to balance the pole entirely (Catastrophic Forgetting).</p>

          <h3>11. Data Assumptions</h3>
          <p>It assumes the environment physics are consistent (Markovian). If gravity randomly changed every 5 seconds, the Neural Network's weights would constantly invalidate themselves.</p>

          <h3>12. How it Overfits</h3>
          <p>If Epsilon decays too fast (e.g., dropping to 0% in 5 episodes), the AI will never fully explore the environment. It will lock into a terrible, suboptimal strategy just because it worked once.</p>

          <h3>13. How it Underfits</h3>
          <p>If the Neural Network Learning Rate is too high, the weights will bounce around wildly, and the Loss will explode to infinity. The cart will just violently vibrate back and forth.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Episodes * Steps * W)</code> (Must run Neural Net backprop at every single step of the game).</li>
            <li><strong>Training Space:</strong> <code>O(Replay_Buffer_Size + W)</code> (Must save millions of past memories to RAM to prevent Catastrophic Forgetting).</li>
            <li><strong>Prediction Time:</strong> <code>O(W)</code> (A fast forward-pass through the network to pick an action).</li>
            <li><strong>Prediction Space:</strong> <code>O(W)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Modern production DQNs use "Experience Replay." Instead of learning from an action immediately and forgetting it, the AI saves its memories to a database, and randomly samples batches of past memories to train on. This completely destroys the Catastrophic Forgetting problem!</p>
        `
      }
    ]
  },
  {
    category: "Semi-Supervised",
    algorithms:[
      {
        id: "pl",
        title: "Pseudo-Labeling (Proxy Labeling)",
        path: "/pl",
        content: `
          <h3>1. What is it?</h3>
          <p>In the real world, data is free, but human labels are incredibly expensive to hire. Pseudo-labeling lets you train a model on a tiny amount of human-labeled data, and then sets that model loose to automatically "guess" the labels for the millions of unclassified data points!</p>

          <h3>2. Real-World Analogy</h3>
          <p>A doctor diagnoses 5 X-Rays as Cancer and 5 as Healthy. We train a basic AI on those 10 images. We then feed the AI 1,000,000 blank X-Rays. If the AI is 99% confident that an X-Ray is cancer, it permanently stamps a "Cancer" label on it. It uses these newly labeled images to make itself smarter, creating a compounding snowball effect.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Train a baseline model on the handful of true, colored dots.</li>
            <li><strong>Step 2:</strong> Ask the model to predict the probabilities of all the Gray (Unlabeled) dots.</li>
            <li><strong>Step 3:</strong> If the model's confidence exceeds a strict threshold (e.g., >85%), permanently color that dot!</li>
            <li><strong>Step 4:</strong> Combine the True labels with the new Pseudo-labels, and retrain the model entirely from scratch. Repeat!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Entropy Minimization:</strong> By forcing the model to make hard predictions on unlabeled data, we mathematically force the decision boundary to move away from dense regions of data and settle into the low-density gaps.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Confidence Threshold:</strong> The strictness limit. Setting it to 51% means the AI will recklessly label everything. Setting it to 99% means it will only label the most obvious cases.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Data Utilization Chart:</strong> Look at the Bar Chart in the visualizer! It visually proves how much money you saved. You started with 6 labeled points, and the AI safely turned that into 100 labeled points for free.</p>

          <h3>7. Best Use Cases</h3>
          <p>Medical imaging, speech recognition, and web-scraping pipelines where downloading raw text/images is fast, but paying humans to categorize them is impossible.</p>

          <h3>8. Worst Use Cases</h3>
          <p>When the initial human labels are heavily biased. If the human only labeled small dogs, the AI will confidently mislabel every large dog as a cat.</p>

          <h3>9. Major Advantages</h3>
          <p>Massive cost savings. It dramatically improves the accuracy of a weak model by allowing it to map out the underlying physical density of the unlabeled data.</p>

          <h3>10. Major Disadvantages</h3>
          <p><strong>Confirmation Bias:</strong> If the AI makes a wrong guess early on, it permanently bakes that mistake into its dataset and trains itself to be even more confidently wrong on the next iteration!</p>

          <h3>11. Data Assumptions</h3>
          <p>It relies heavily on the <strong>Cluster Assumption:</strong> Points that are physically clustered together must belong to the same class, and the decision boundary should naturally fall in the empty gaps between clusters.</p>

          <h3>12. How it Overfits</h3>
          <p>If the Confidence Threshold is too low, the boundary rapidly collapses into junk shapes as the AI confidently hallucinates incorrect labels.</p>

          <h3>13. How it Underfits</h3>
          <p>If the Threshold is too high (99.9%), the AI becomes paralyzed with anxiety. It refuses to label any gray dots, completely failing to utilize the unlabeled data.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(I * [T_train + T_predict])</code> (Where <em>I</em> = Iterations, and <em>T</em> is the time complexity of the underlying Base Model). Pseudo-labeling is a "Wrapper" algorithm. It forces you to completely retrain the base model from scratch, and then run a prediction on every unlabeled point, over and over again!</li>
            <li><strong>Training Space:</strong> <code>O(n * d + S_base)</code> (Must hold the expanding dataset in RAM along with the Base Model's memory footprint).</li>
            <li><strong>Prediction Time:</strong> <code>O(T_predict)</code> (Exactly the same as the Base Model. Once training is done, the Pseudo-Labeling wrapper is thrown away!).</li>
            <li><strong>Prediction Space:</strong> <code>O(S_base)</code> (Just the size of the final trained Base Model).</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Notice the Confidence Margins (the dotted lines) in the animation. The AI will only pseudo-label a gray dot if it physically falls outside those lines. As dots get labeled, they pull the boundary closer to the true center of the clusters!</p>
        `
      },
      ,{
        id: "cl",
        title: "Contrastive Learning (Self-Supervised)",
        path: "/cl",
        content: `
          <h3>1. What is it?</h3>
          <p>Contrastive Learning is the magic behind modern AI like FaceID and ChatGPT. It doesn't use human labels. Instead, it mathematically forces similar data points together in a hidden "Latent Space" while violently pushing random points far apart, naturally creating perfect clusters out of thin air.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine trying to organize a massive, messy room full of unlabeled items. You pick up two items. If they look similar (two shoes), you tie them together with a rubber band. You pick a random third item (a book) and push it to the other side of the room. Do this a million times, and your room perfectly sorts itself into piles of shoes, books, and clothes!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Take a data point (an Anchor). Find a highly similar point (a Positive match) and select a random point (a Negative match).</li>
            <li><strong>Step 2:</strong> Map all three into a 2D Latent Space. (They start in a messy pile at the origin [0,0]).</li>
            <li><strong>Step 3:</strong> Calculate gradients to pull the Anchor and Positive together, while pushing the Anchor and Negative apart.</li>
            <li><strong>Step 4:</strong> Repeat across the dataset until the points explode into clearly separated clusters.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Contrastive Margin Loss (InfoNCE):</strong> <code>Loss = D(Anchor, Positive)² + max(0, Margin - D(Anchor, Negative))²</code><br/>
          It penalizes the AI if the Positive point is too far away, and penalizes it if the Negative point is closer than your chosen Margin radius.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Repulsion Margin:</strong> The physical distance the AI demands between different clusters. A margin of 5.0 means the AI will violently shove clusters to the absolute edges of the map.</li>
            <li><strong>Epochs:</strong> How many times the AI applies the push/pull physics to the dataset.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>The Latent Space Animation:</strong> You are literally watching the "Brain" of the AI form. By the final frame, the AI has invented a coordinate system where physical distance equals semantic meaning!</p>

          <h3>7. Best Use Cases</h3>
          <p>Pre-training massive foundation models. If you have 10 billion uncaptioned images from the internet, you use Contrastive Learning to organize them into a Latent Dictionary (like OpenAI's CLIP model). Then, you only need 100 human labels to map text onto the clusters!</p>

          <h3>8. Worst Use Cases</h3>
          <p>When you have a small dataset with perfectly clean, tabular labels (like Excel data). Using Self-Supervised learning here is a massive waste of computational power compared to a simple Random Forest.</p>

          <h3>9. Major Advantages</h3>
          <p>Infinite training data. You never have to pay humans to label anything. The AI automatically finds the underlying structure of reality.</p>

          <h3>10. Major Disadvantages</h3>
          <p><strong>Collapsing:</strong> If the AI is poorly configured (e.g., no negative points are used), it will find a "cheat code" and map every single point in the dataset to the exact same [0,0] coordinate. The loss drops to zero, but the model is completely useless.</p>

          <h3>11. Data Assumptions</h3>
          <p>It assumes you have a mathematically valid way to define a "Positive" pair. In images, a Positive pair is made by cropping or flipping the same image (Data Augmentation).</p>

          <h3>12. How it Overfits</h3>
          <p>If your batch sizes are too small, the AI will push away points that are actually in the same class just because they were randomly selected as "Negatives," shattering a true cluster into a dozen useless micro-clusters.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting the Repulsion Margin to 0.0. The AI will pull similar points together, but without a repulsion force, all the clusters will bleed into one giant, overlapping blob in the center of the map.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * (W + B * d))</code> (Where <em>W</em> = Neural Net Weights, <em>B</em> = Batch Size). To find "Negatives", the AI must calculate the distance between every single image and <em>every other image</em> in the current Batch. This quadratic <code>B²</code> math makes large batch sizes computationally explosive!</li>
            <li><strong>Training Space:</strong> <code>O(W + B * d)</code> (Must store the massive Neural Network gradients and the embedding coordinates for the entire current batch in VRAM).</li>
            <li><strong>Prediction Time:</strong> <code>O(W)</code> (To embed a new image, it just does one fast forward-pass through the Encoder network).</li>
            <li><strong>Prediction Space:</strong> <code>O(W)</code> (Stores the frozen Encoder weights. The Projection Head used during training is usually deleted!).</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Look at the colors in the Latent Space graph. The AI is entirely colorblind! It only sees X/Y coordinates. The fact that all the Red points landed together proves that the math successfully identified their hidden relationship.</p>
        `
      }
    ]
  },
  {
    category: "Deep Learning",
    algorithms:[
      {
        id: "ann",
        title: "Artificial Neural Networks (ANN)",
        path: "/ann",
        content: `
          <h3>1. What is it?</h3>
          <p>ANNs are computing systems inspired by the human brain. Instead of using a single math equation, they use a massive web of interconnected "Neurons" organized in layers to map incredibly complex relationships between inputs and outputs.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine an assembly line of art critics trying to identify a painting. The first row of critics only looks for straight lines. They pass their notes to the second row, who combine those lines into shapes. The final row combines those shapes to yell "It's a house!"</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1 (Forward Pass):</strong> Data flows into the Input Layer, multiplies by mathematical Weights, and passes through Activation Functions in the Hidden Layers.</li>
            <li><strong>Step 2:</strong> The Output Layer spits out a prediction.</li>
            <li><strong>Step 3 (Loss):</strong> The AI calculates how wrong its prediction was.</li>
            <li><strong>Step 4 (Backpropagation):</strong> Using Calculus (the Chain Rule), the error is sent backward through the network, tweaking every single weight to be slightly more accurate next time.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Neuron Math:</strong> <code>y = σ(W*x + b)</code>. The neuron multiplies inputs by Weights (W), adds a Bias (b), and passes it through an Activation Function (σ) like ReLU or Sigmoid to introduce non-linearity.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Hidden Layers & Neurons:</strong> Dictates the "Brain Size." More neurons mean it can solve harder problems, but take longer to train.</li>
            <li><strong>Epochs:</strong> One full cycle of passing the entire dataset through the network forward and backward.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Training vs Validation Loss Curve:</strong> Look at the graph in the visualization. If both lines go down, the AI is learning. If the Training line goes down to zero but the Validation line shoots up into the sky, the AI is overfitting (memorizing the test answers).</p>

          <h3>7. Best Use Cases</h3>
          <p>Highly complex tabular data, non-linear classification, and as the foundational building block for all other Deep Learning models.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Images (use CNNs) and Time-Series data (use RNNs). Standard ANNs flatten everything into a 1D list, destroying spatial and temporal awareness.</p>

          <h3>9. Major Advantages</h3>
          <p>They are "Universal Function Approximators." Given enough neurons and time, an ANN can mathematically learn to replicate ANY pattern or formula in the universe.</p>

          <h3>10. Major Disadvantages</h3>
          <p>They are the ultimate "Black Box." Unlike a Decision Tree, you cannot look inside an ANN and easily explain to a CEO *why* it made a specific prediction. The logic is hidden across millions of floating-point numbers.</p>

          <h3>11. Data Assumptions</h3>
          <p>Data MUST be strictly scaled (between 0 and 1) or Normalized. If you feed an ANN raw numbers like $450,000, the gradients will explode to infinity and crash the math.</p>

          <h3>12. How it Overfits</h3>
          <p>Using 5 massive Hidden Layers on a tiny dataset of 10 points. The network has so much brainpower it will memorize the exact coordinates of the noise dots.</p>

          <h3>13. How it Underfits</h3>
          <p>Using no Hidden Layers. An ANN with 0 hidden layers mathematically collapses into a basic Logistic Regression model, incapable of drawing curves.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * W)</code> (Where <em>W</em> = total number of weights in all layers).</li>
            <li><strong>Training Space:</strong> <code>O(W + batch_size * Neurons)</code> (Must store weights and backprop gradients in VRAM).</li>
            <li><strong>Prediction Time:</strong> <code>O(W)</code> (Fast forward-pass matrix multiplication).</li>
            <li><strong>Prediction Space:</strong> <code>O(W)</code> (Stores the frozen weights).</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Always use the <strong>ReLU</strong> activation function in your hidden layers instead of Sigmoid. Sigmoid causes a fatal math error called "Vanishing Gradients" in deep networks, freezing the learning process entirely!</p>
        `
      },
      {
        id: "cnn",
        title: "Convolutional Neural Networks (CNN)",
        path: "/cnn",
        content: `
          <h3>1. What is it?</h3>
          <p>A CNN is a neural network designed specifically for vision. Instead of looking at an entire image at once, it slides mathematical "Filters" (like a magnifying glass) over the pixels to detect local patterns like edges, textures, and ultimately, complex objects.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine searching for a missing person in a massive crowd. You don't look at the whole crowd at once; you scan a pair of binoculars across the crowd, row by row, looking specifically for a red hat. The CNN's "Filters" are those binoculars!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1 (Convolution):</strong> Slide a 3x3 mathematical matrix (Filter) across the image to highlight physical features (like vertical edges).</li>
            <li><strong>Step 2 (Activation):</strong> Pass the feature map through a ReLU function to drop negative values.</li>
            <li><strong>Step 3 (Pooling):</strong> Shrink the image (Max Pooling) to compress it and focus only on the brightest features.</li>
            <li><strong>Step 4 (Flatten & Dense):</strong> Unroll the final tiny feature maps into a 1D list and feed it to a standard Neural Network to guess the final label.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The Convolution:</strong> It performs a dot product between the 3x3 Filter weights and the 3x3 pixel grid it is currently sitting on. If the pixels match the filter's pattern, the math outputs a massive positive number (Activation)!</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Filter Size (Kernel):</strong> Usually 3x3. Larger filters (5x5) look at broader shapes but lose micro-details.</li>
            <li><strong>Stride:</strong> How many pixels the filter shifts when it slides. Stride 1 moves smoothly; Stride 2 skips pixels and instantly halves the image size.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Feature Maps:</strong> Look at the visualizer! Notice how the original drawn digit gets transformed into weird, glowing, ghost-like shapes. The CNN literally created new images that only contain the edges and lines it deemed important!</p>

          <h3>7. Best Use Cases</h3>
          <p>Image classification (Cats vs Dogs), Facial Recognition, Self-Driving Car vision, and Medical Tumor detection.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Tabular spreadsheet data. If Column 1 is "Age" and Column 2 is "Income", they have no physical spatial relationship, making a sliding 3x3 Convolutional Filter completely useless.</p>

          <h3>9. Major Advantages</h3>
          <p><strong>Translation Invariance:</strong> Because the filter slides across the whole image, a CNN can recognize a cat whether it is in the top-left corner or the bottom-right corner!</p>

          <h3>10. Major Disadvantages</h3>
          <p>They are blind to rotation and scale. If you train a CNN on right-side-up cats, and show it an upside-down cat, it will fail completely unless you trained it with "Data Augmentation."</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes that pixels close to each other are highly related, and that local patterns (like an eye or an ear) are the building blocks of global patterns (a face).</p>

          <h3>12. How it Overfits</h3>
          <p>Using too many layers with millions of filters on a small dataset. The CNN will memorize the exact background noise (like a watermark or a timestamp) of the training images instead of the actual subject.</p>

          <h3>13. How it Underfits</h3>
          <p>Using only 1 Convolutional layer. Early layers only learn simple edges and lines. You need deep, subsequent layers to combine those lines into complex shapes like "Noses" and "Tires."</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * F * K² * Image_Size²)</code> (Where <em>F</em>=Filters, <em>K</em>=Kernel size. Incredibly heavy, requires GPUs).</li>
            <li><strong>Training Space:</strong> <code>O(W + Feature_Maps_Memory)</code> (Feature maps eat massive amounts of RAM).</li>
            <li><strong>Prediction Time:</strong> <code>O(F * K² * Image_Size²)</code>.</li>
            <li><strong>Prediction Space:</strong> <code>O(W)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Notice the <strong>Max Pooling</strong> layer in the visualizer. Max Pooling literally throws away 75% of the pixels, keeping only the brightest one in a 2x2 grid. This compresses the image, making the network exponentially faster without losing the core features!</p>
        `
      },
      {
        id: "rnn",
        title: "Recurrent Neural Networks (RNN)",
        path: "/rnn",
        content: `
          <h3>1. What is it?</h3>
          <p>RNNs are Neural Networks with a short-term memory. Unlike normal AI that processes entire images at once, an RNN reads data sequentially (step-by-step over time), updating a "Hidden State" to remember what happened in the past so it can predict the future.</p>

          <h3>2. Real-World Analogy</h3>
          <p>If you watch a movie frame-by-frame, a normal Neural Network looks at Frame #45 and tries to guess what is happening with zero context. An RNN watches Frames 1 through 44, builds a memory of the plot, and uses that context to perfectly understand Frame #45.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> The RNN looks at Time Step 1. It saves some information to its Hidden State (Memory).</li>
            <li><strong>Step 2:</strong> It moves to Time Step 2. It mathematically mixes the new data with its Memory of Step 1.</li>
            <li><strong>Step 3:</strong> It slides across the "Lookback Window," continuously rolling its memory forward.</li>
            <li><strong>Step 4:</strong> At the final step, it outputs a prediction (e.g., tomorrow's stock price) based on the accumulated context.</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The Recurrent Cell:</strong> <code>h_t = tanh(W_x * X_t + W_h * h_{t-1} + bias)</code><br/>
          This brilliant formula dictates that the Current Memory (h_t) is an exact mixture of the New Input (X_t) and the Previous Memory (h_{t-1}).</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Lookback Window:</strong> How many timesteps into the past the AI is allowed to read. Lookback=2 gives the AI amnesia. Lookback=30 allows it to see grand historical trends.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Memory Cell Activity (The Purple Bars):</strong> Look at the Composed Chart in the visualization! The purple bars show the physical mathematical firing of the Hidden State. Notice how the memory cell spikes heavily whenever the dataset makes a sudden, unpredictable zigzag. It is literally "thinking" harder to memorize the anomaly!</p>

          <h3>7. Best Use Cases</h3>
          <p>Time-series forecasting (Stock Market, Weather), Electrocardiograms (ECG heartbeats), and foundational Natural Language Processing (reading sentences word by word).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Static, independent data. Using an RNN to predict house prices based on Square Footage makes no sense because there is no "timeline" linking House A to House B.</p>

          <h3>9. Major Advantages</h3>
          <p>It can handle variable-length inputs. Whether you feed it a 5-word sentence or a 50-word sentence, the RNN cell just keeps rolling forward until it reaches the end.</p>

          <h3>10. Major Disadvantages</h3>
          <p><strong>The Vanishing Gradient Problem.</strong> Because the math multiplies the memory state by a weight matrix at every step, if you look back 100 days, the math multiplies 100 times. The signal quickly shrinks to 0.0000001, causing the RNN to completely "forget" what happened on Day 1.</p>

          <h3>11. Data Assumptions</h3>
          <p>It fundamentally assumes temporal dependency—that the data at <code>T=5</code> is directly caused by the events at <code>T=4</code>, <code>T=3</code>, etc.</p>

          <h3>12. How it Overfits</h3>
          <p>If you train an RNN on a highly volatile stock market, it will memorize the exact noise of the past 10 days and fail completely when real-world macro-economics change tomorrow.</p>

          <h3>13. How it Underfits</h3>
          <p>Setting the Lookback Window too short. If the data operates on a 7-day weekly cycle, but you set Lookback=3, the AI will never realize that weekends exist.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * Timesteps * W)</code> (Cannot be parallelized across time. BPTT is very slow).</li>
            <li><strong>Training Space:</strong> <code>O(Timesteps * W)</code> (Must store the hidden state for every step to calculate gradients backward).</li>
            <li><strong>Prediction Time:</strong> <code>O(Timesteps * W)</code>.</li>
            <li><strong>Prediction Space:</strong> <code>O(W)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Vanilla RNNs are almost dead in the real world. Engineers use an upgraded version called <strong>LSTM (Long Short-Term Memory)</strong>, which adds chemical "Forget Gates" to actively protect long-term memories from vanishing!</p>
        `
      },
      {
        id: "transformer",
        title: "Transformers (LLMs)",
        path: "/transformer",
        content: `
          <h3>1. What is it?</h3>
          <p>Transformers are the absolute pinnacle of AI. They are the engine behind ChatGPT. Instead of reading words one-by-one like an RNN, Transformers process the <em>entire paragraph at the exact same time</em> using a mechanism called "Self-Attention" to understand the profound context between words.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine reading the sentence: "The bank of the river." An RNN reads "bank" and thinks of money until it reaches "river" 3 seconds later and has to rewrite its memory. A Transformer looks at all 5 words simultaneously. The word "river" shines a bright spotlight backward onto the word "bank", instantly clarifying the context!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1 (Tokenize & Embed):</strong> Words are mapped to X/Y coordinates in a Latent Dictionary.</li>
            <li><strong>Step 2 (Positional Encoding):</strong> Sine/Cosine math is added so the AI knows the order of the words.</li>
            <li><strong>Step 3 (Self-Attention):</strong> Every word asks questions (Queries) and provides answers (Keys) to every other word to find relationships.</li>
            <li><strong>Step 4 (Softmax Prediction):</strong> The final contextual web spits out probabilities for what the next logical word should be!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Scaled Dot-Product Attention:</strong> <code>Attention(Q, K, V) = softmax(Q * K^T / √d) * V</code><br/>
          This is the most famous equation in modern AI. It takes your Queries, multiplies them by the Keys of other words to create a Heatmap score, and applies that score to the actual Value of the word.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Temperature:</strong> Controls the "Creativity." Temp 0.1 makes the AI robotic and boring. Temp 2.0 flattens the probability, forcing the AI to hallucinate crazy words.</li>
            <li><strong>Causal Masking:</strong> An absolute necessity for generating text (GPT style). It blinds the AI, preventing a word from "looking into the future" to see the next word.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>The Self-Attention Matrix:</strong> Look at the top-left visualizer. Dark blue squares mean the AI found a massive mathematical connection between those two specific words! The white upper-right triangle is the Causal Mask physically blocking the future.</p>

          <h3>7. Best Use Cases</h3>
          <p>Large Language Models (LLMs), language translation, code generation, and analyzing DNA/Protein sequences.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Simple numeric tasks. Using a 175-Billion parameter Transformer to predict house prices in a 500-row Excel sheet is astronomically inefficient.</p>

          <h3>9. Major Advantages</h3>
          <p><strong>Parallelization!</strong> Because they don't read words sequentially, you can throw a 10,000-word essay into a GPU and process all 10,000 words at the exact same millisecond. This speed allows us to train on the entire Internet.</p>

          <h3>10. Major Disadvantages</h3>
          <p>Transformers suffer from quadratic memory scaling O(N²). If you double the length of your input prompt, the Self-Attention Matrix takes '4x' the RAM. This is why ChatGPT has "Context Window" limits.</p>

          <h3>11. Data Assumptions</h3>
          <p>The model assumes absolutely nothing about word order until the Positional Encodings are manually injected into the math.</p>

          <h3>12. How it Overfits</h3>
          <p>If you train a 100-Billion parameter model on a small dataset, it will literally memorize the exact paragraphs of the training data and regurgitate them perfectly instead of actually learning grammar.</p>

          <h3>13. How it Underfits</h3>
          <p>Insufficient training time. A Transformer starts completely ignorant. It takes months on supercomputers for the Q, K, and V weight matrices to learn the complexities of human language.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * (L² * d + L * d²))</code> (Where <em>L</em> is Context Window Length. The L² is the self-attention matrix exploding quadratically).</li>
            <li><strong>Training Space:</strong> <code>O(L² + W)</code> (The attention heatmap requires terrifying amounts of memory for long books).</li>
            <li><strong>Prediction Time:</strong> <code>O(L² * d)</code> (Auto-regressive generation requires recalculating attention for the whole sequence per word).</li>
            <li><strong>Prediction Space:</strong> <code>O(L² + W)</code> (The KV-Cache).</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Hover over the dots in the <strong>Latent Embedding Space</strong> chart. The AI doesn't know English. It just mapped "Cat" and "Dog" close together in physical space because they appear in similar sentences. Language is just geometry to an AI!</p>
        `
      },
      {
        id: "ae",
        title: "Autoencoders",
        path: "/ae",
        content: `
          <h3>1. What is it?</h3>
          <p>An Autoencoder is an unsupervised Neural Network shaped like an hourglass. It learns to violently compress data into a tiny "bottleneck," and then uses a Decoder network to mathematically rebuild the original data from that crushed state.</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine reading a 500-page book. Your boss asks you to summarize it in exactly 3 words (The Bottleneck). You write "Boy meets wizard." Later, your friend takes those 3 words and attempts to rewrite the entire 500-page book from scratch. If they succeed, you have built a perfect Autoencoder!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1 (Encoder):</strong> Pass a complex 2D data point into the network. Hidden layers shrink it down to a 1D number.</li>
            <li><strong>Step 2 (Bottleneck):</strong> The data is trapped in the "Latent Space" (the narrowest point).</li>
            <li><strong>Step 3 (Decoder):</strong> The network expands the 1D number back into 2D coordinates.</li>
            <li><strong>Step 4:</strong> Calculate the difference (MSE) between the input and the reconstructed output. Backpropagate to improve!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Reconstruction Loss:</strong> <code>Loss = || X - X_hat ||²</code><br/>
          The AI measures the literal physical distance between where the true point is (X), and where the Decoder drew the fake reconstructed point (X_hat).</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Latent Dimension Size:</strong> How wide the bottleneck is. If your data is 2D, and your bottleneck is 1D, the Compression Ratio is 2:1.</li>
            <li><strong>Epochs:</strong> Autoencoders require hundreds of passes to slowly figure out the underlying manifold (shape) of the data.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>The Latent Space Graph:</strong> Look at the 1D Bottleneck chart! You passed the AI massive 2D coordinates. The AI mathematically crushed all of them onto a single 1D number line! This proves the AI discovered the hidden underlying structure of your shape.</p>

          <h3>7. Best Use Cases</h3>
          <p>Image Compression, Denoising (removing static from old photos), and Anomaly Detection (if a credit card transaction compresses badly, it's probably fraud!).</p>

          <h3>8. Worst Use Cases</h3>
          <p>Predictive Classification. Autoencoders don't output "Yes/No" answers, they output reconstructed coordinates/images.</p>

          <h3>9. Major Advantages</h3>
          <p>It is basically <strong>Non-Linear PCA</strong>. While PCA can only draw straight lines to reduce dimensions, Autoencoders can use Activation Functions (ReLU, Tanh) to bend, twist, and curve the latent space to perfectly map things like S-Curves and Moons!</p>

          <h3>10. Major Disadvantages</h3>
          <p>Standard Autoencoders do not create a "continuous" latent space. If you pick a random empty spot in the bottleneck and ask the Decoder to build an image from it, it will just spit out garbage.</p>

          <h3>11. Data Assumptions</h3>
          <p>It fundamentally assumes that your High-Dimensional data actually lives on a Low-Dimensional manifold (e.g., that a 1-Megapixel image of a face can actually be summarized by just 50 numbers like "eye distance" and "nose length").</p>

          <h3>12. How it Overfits</h3>
          <p>If you make the Bottleneck the same size as the Input. The AI will act like an "Identity Function". It will just copy the input, paste it to the output, achieve a Loss of 0.0, and learn absolutely nothing of value.</p>

          <h3>13. How it Underfits</h3>
          <p>If you compress a 4K movie into a bottleneck of 1 single neuron. The AI cannot physically store that much information in one number, and the decoded output will be a blurry, gray mess.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * W)</code> (Standard neural network backprop).</li>
            <li><strong>Training Space:</strong> <code>O(W)</code>.</li>
            <li><strong>Prediction Time:</strong> <code>O(W_encoder)</code> (If compressing) or <code>O(W_decoder)</code> (If generating).</li>
            <li><strong>Prediction Space:</strong> <code>O(W)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>To fix the garbage-generation problem, engineers upgraded this to the <strong>Variational Autoencoder (VAE)</strong>. VAEs force the bottleneck to be a smooth Gaussian distribution, allowing you to sample random points to magically generate brand-new, realistic images!</p>
        `
      },
      {
        id: "gan",
        title: "Generative Adversarial Networks (GANs)",
        path: "/gan",
        content: `
          <h3>1. What is it?</h3>
          <p>GANs are two neural networks locked in a war. The <strong>Generator</strong> acts like an art forger trying to create fake data. The <strong>Discriminator</strong> acts like a cop trying to catch the fakes. By fighting each other, the Generator learns to create hyper-realistic data from pure static noise!</p>

          <h3>2. Real-World Analogy</h3>
          <p>A counterfeiter prints a fake $100 bill and gives it to a bank teller. The teller catches it and says, "Fake! The watermark is wrong." The counterfeiter goes home, fixes the watermark, and tries again. Eventually, the counterfeiter becomes so perfect that the teller has exactly a 50/50 chance of guessing correctly.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> Feed pure random noise into the Generator. It outputs a "Fake" 2D dot.</li>
            <li><strong>Step 2:</strong> Hand the Fake dot and a Real dot to the Discriminator. The Discriminator guesses which is which.</li>
            <li><strong>Step 3:</strong> If the Discriminator is fooled, it updates its weights to get smarter.</li>
            <li><strong>Step 4:</strong> If the Discriminator catches the fake, the Generator updates its weights to become a better liar!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The Min-Max Game:</strong> <code>min(G) max(D) V(D,G) = E[log(D(x))] + E[log(1 - D(G(z)))]</code><br/>
          The Discriminator wants to maximize this equation (confidently identify Real vs Fake). The Generator wants to minimize this equation (make the Discriminator guess zero).</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Learning Rates:</strong> You must perfectly balance the learning speeds of the two networks. If one learns faster than the other, the entire GAN collapses.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Min-Max Loss War (The Graph):</strong> In normal ML, you want Loss to hit 0.0. In a GAN, if Loss hits 0.0, the model is broken! You want to see the Purple and Green lines oscillating and fighting forever, eventually stabilizing around 0.69 (which is the mathematical representation of a 50/50 coin flip!).</p>

          <h3>7. Best Use Cases</h3>
          <p>Creating Deepfakes, generating photorealistic AI art (StyleGAN), increasing the resolution of blurry images (Super-Resolution GANs), and generating synthetic medical data for research.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Predicting discrete numbers or performing standard classification. GANs are strictly for Generative tasks.</p>

          <h3>9. Major Advantages</h3>
          <p>They produce the sharpest, crispest, most highly realistic images in the entire field of Artificial Intelligence, often surpassing Autoencoders.</p>

          <h3>10. Major Disadvantages</h3>
          <p>They are notoriously brutal to train. The math is highly unstable. If the Discriminator gets too smart too fast, the Generator's gradients drop to zero, and it completely stops learning.</p>

          <h3>11. Data Assumptions</h3>
          <p>Assumes that the "Real Data" exists on a complex, high-dimensional manifold, and that we can mathematically map a simple random noise vector onto that exact manifold.</p>

          <h3>12. How it Fails (Mode Collapse)</h3>
          <p><strong>Mode Collapse</strong> is the ultimate GAN failure. The Generator finds one single image (e.g., one specific cat face) that perfectly fools the Discriminator. Instead of learning to draw all cats, it becomes lazy and just outputs that exact same cat face forever!</p>

          <h3>13. How it Fails (Vanishing Gradients)</h3>
          <p>If the Discriminator is perfect, it outputs a probability of 0.0000001 for fakes. When the Generator tries to Backpropagate that tiny number to improve itself, the math vanishes, and the Generator is frozen forever.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * (W_gen + W_disc))</code> (You are training two separate deep networks simultaneously).</li>
            <li><strong>Training Space:</strong> <code>O(W_gen + W_disc)</code>.</li>
            <li><strong>Prediction Time:</strong> <code>O(W_gen)</code> (Discriminator is thrown away after training!).</li>
            <li><strong>Prediction Space:</strong> <code>O(W_gen)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Look at the <strong>Discriminator Heatmap</strong> in the visualization! The deep green areas show where the "Cop" is 100% confident the data is real. Notice how the purple dots (the Generator) actively hunt down and hide exactly inside those green zones to avoid being caught!</p>
        `
      },
      {
        id: "gcn",
        title: "Graph Convolutional Networks (GCN)",
        path: "/gcn",
        content: `
          <h3>1. What is it?</h3>
          <p>Standard Neural Networks look at isolated rows in a spreadsheet. GCNs look at <strong>Networks</strong>. A GCN mathematically allows a data point to "talk" to its friends, updating its own identity based on who it is connected to in the graph!</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine a Social Network. You have a blank profile, but 5 of your friends are tagged as "Software Engineers," and 1 is a "Designer." A GCN looks at the "Edges" (friendships) connecting you, averages out their labels, and mathematically concludes you are probably a Software Engineer too.</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1:</strong> The AI builds an Adjacency Matrix (a grid showing exactly which dots are connected by lines).</li>
            <li><strong>Step 2 (Message Passing):</strong> Every node gathers the features of its direct neighbors and adds them to its own features.</li>
            <li><strong>Step 3:</strong> The blended features are passed through a standard Neural Network layer.</li>
            <li><strong>Step 4:</strong> The network predicts the color/label of the unlabeled nodes based on the newly blended information!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>The GCN Update Rule:</strong> <code>H_new = σ( D^{-1/2} * A * D^{-1/2} * H_old * W )</code><br/>
          This brilliant matrix math takes the Adjacency Matrix (A), normalizes it so popular nodes don't overpower the math (D), multiplies it by the Node Features (H), and learns the optimal Weights (W).</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Epochs:</strong> Every layer (epoch) in a GCN allows a message to travel one "hop" further. Epoch 1 = Friends. Epoch 2 = Friends of Friends!</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>GCN Latent Embeddings (The Scatter Plot):</strong> Notice how the animation pulls the dots together. Because the nodes passed messages to their neighbors, the mathematical gravity of the GCN physically yanked connected nodes into the exact same spot in the Latent Space!</p>

          <h3>7. Best Use Cases</h3>
          <p>Social network analysis, discovering new molecular drugs (where atoms are nodes and bonds are edges), and Pinterest/Amazon Recommendation Engines.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Standard tabular data. If your data points have no relationships or links between them, multiplying by an empty Adjacency Matrix does absolutely nothing.</p>

          <h3>9. Major Advantages</h3>
          <p>It fundamentally exploits topological structure. It doesn't just know *what* a data point is; it knows *where* it sits in the fabric of the network.</p>

          <h3>10. Major Disadvantages</h3>
          <p><strong>Over-Smoothing.</strong> If you use too many layers (e.g., 10 layers deep), the messages travel so far that every single node in the entire network averages out into the exact same gray mush. The AI goes completely blind.</p>

          <h3>11. Data Assumptions</h3>
          <p>It heavily relies on <strong>Homophily</strong>: the assumption that connected nodes tend to share the same properties and labels (Birds of a feather flock together).</p>

          <h3>12. How it Overfits</h3>
          <p>If the graph is very sparse (few edges), the GCN will overfit to the tiny handful of connections it sees, memorizing the specific graph structure and failing if applied to a new, different network.</p>

          <h3>13. How it Underfits</h3>
          <p>Using only 1 Layer. The nodes will only be able to see their absolute direct neighbors, completely missing out on the vast global structure of the network.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * |E| * d)</code> (Where <em>|E|</em> is the number of Edges. Sparse matrices make this very fast).</li>
            <li><strong>Training Space:</strong> <code>O(|V| * d + |E|)</code> (Stores the Nodes and Adjacency Matrix).</li>
            <li><strong>Prediction Time:</strong> <code>O(|E| * d)</code>.</li>
            <li><strong>Prediction Space:</strong> <code>O(|V| * d + |E|)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>GCNs are uniquely <strong>Transductive</strong>. Notice in the Sandbox that the AI can "see" the gray, unlabeled dots during training! It uses the gray dots as bridges to pass messages between the colored dots. It feels like cheating, but it is mathematically brilliant!</p>
        `
      },
      {
        id: "diffusion",
        title: "Diffusion Models",
        path: "/diffusion",
        content: `
          <h3>1. What is it?</h3>
          <p>Diffusion Models are the bleeding-edge technology behind Midjourney and DALL-E. They work by taking a perfectly good piece of data, systematically destroying it with static noise, and then training a Neural Network to act as a "Denoising Engine" to reverse the process and generate brand new data out of thin air!</p>

          <h3>2. Real-World Analogy</h3>
          <p>Imagine dropping a single drop of ink into a glass of water. Over time, the ink diffuses and spreads until the water is just a cloudy, gray mess (The Forward Process). Now, imagine you have a magical pair of tweezers that can grab water molecules and pull the ink back together into a single, perfect drop (The Reverse Process). That magical tweezer is the AI!</p>

          <h3>3. How does it Work</h3>
          <ul>
            <li><strong>Step 1 (Forward Process):</strong> Take a real data point (e.g., an S-Curve) and mathematically inject Gaussian noise into it step-by-step until it becomes pure, unrecognizable static.</li>
            <li><strong>Step 2 (The Neural Network):</strong> Feed the static to the AI. Ask it: <em>"How much noise was just added in the last millisecond?"</em></li>
            <li><strong>Step 3 (Training):</strong> The AI predicts the noise, subtracts it, and checks if it got closer to the original shape.</li>
            <li><strong>Step 4 (Generation):</strong> Once fully trained, you throw away the original data. You hand the AI a completely random pile of TV static, and it iteratively carves a beautiful new shape out of it!</li>
          </ul>

          <h3>4. The Core Math & Formulas</h3>
          <p><strong>Markov Chain Denoising:</strong> <code>x_{t-1} = x_t - ε_θ(x_t, t)</code><br/>
          This formula says: The cleaner image (<code>x_{t-1}</code>) is equal to the current noisy image (<code>x_t</code>) minus the exact noise the Neural Network (<code>ε_θ</code>) predicts is currently in the image at time step <code>t</code>.</p>

          <h3>5. Key Hyperparameters to Tune</h3>
          <ul>
            <li><strong>Markov Timesteps (T):</strong> How many steps it takes to destroy/rebuild the data. T=1000 makes a gorgeous, highly detailed image but takes forever. T=10 is blazing fast but looks blurry and rushed.</li>
            <li><strong>Noise Schedule (Beta):</strong> The mathematical rate at which noise is added. Linear, Cosine, or Sigmoid schedules dictate whether the noise is added aggressively upfront or smoothly over time.</li>
          </ul>

          <h3>6. What the Outputs/Metrics Mean</h3>
          <p><strong>Signal-to-Noise Ratio (The Graph):</strong> Look at the Beta Noise Schedule graph in the visualizer! At T=0, you have 100% Signal (the pure shape). At T=Max, you have 100% Noise. The model learns how to navigate the exact crossover point where the signal gets lost in the noise.</p>

          <h3>7. Best Use Cases</h3>
          <p>Generative AI. Generating photorealistic images (Stable Diffusion), creating 3D molecules for drug discovery, and generating hyper-realistic synthetic voice audio.</p>

          <h3>8. Worst Use Cases</h3>
          <p>Real-time applications (like video games) or categorical predictions. Diffusion models are inherently slow and are strictly designed for generation, not classification.</p>

          <h3>9. Major Advantages</h3>
          <p><strong>Unprecedented Quality and Stability.</strong> Unlike GANs, which are incredibly unstable and suffer from "Mode Collapse" (only generating one type of image), Diffusion Models are mathematically stable to train and capture the entire diversity of a dataset!</p>

          <h3>10. Major Disadvantages</h3>
          <p>They are <strong>astronomically slow</strong> to run. A GAN generates an image in 1 step (0.01 seconds). A Diffusion Model requires running the image through the Neural Network 50 to 1,000 times sequentially to slowly carve away the noise.</p>

          <h3>11. Data Assumptions</h3>
          <p>It assumes the data distribution is continuous and can be smoothly degraded into a standard Gaussian (Normal) distribution.</p>

          <h3>12. How it Overfits</h3>
          <p>If the Neural Network is too large and the dataset is too small, the AI won't learn <em>how</em> to denoise. Instead, it will literally memorize the exact static patterns of the training data and just spit out identical copies of the training images (plagiarism!).</p>

          <h3>13. How it Underfits</h3>
          <p>If you don't use enough Timesteps, the AI is forced to take massive, sloppy jumps while trying to remove the noise, resulting in images that look like blurry, melted blobs.</p>

          <h3>14. Time & Space Complexity</h3>
          <ul>
            <li><strong>Training Time:</strong> <code>O(Epochs * n * Timesteps * W)</code> (Network must learn the score function for hundreds of noise levels).</li>
            <li><strong>Training Space:</strong> <code>O(W)</code>.</li>
            <li><strong>Prediction Time:</strong> <code>O(Timesteps * W)</code> (Incredibly slow. Must run the Neural Network 1,000 times just to generate 1 image!).</li>
            <li><strong>Prediction Space:</strong> <code>O(W)</code>.</li>
          </ul>

          <h3>15. Pro Tip</h3>
          <p>Want to know how Midjourney understands your text prompts? It uses <strong>Classifier-Free Guidance (CFG)</strong>. The text prompt is mathematically injected into the U-Net at every single timestep, acting like a magnet that pulls the denoising process toward the specific shape of a "Dog" or a "Spaceship"!</p>
        `
      }
    ]
  }
];

