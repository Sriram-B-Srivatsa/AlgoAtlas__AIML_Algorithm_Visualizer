import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import base64
from io import BytesIO, StringIO
import pydot
from scipy import sparse
import re

# Custom Node class for decision trees
class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

# Custom TreeStructure class to mimic sklearn's tree_ attribute
class TreeStructure:
    def __init__(self):
        self.node_count = 0
        self.children_left =[]
        self.children_right = []
        self.feature = []
        self.threshold =[]
        self.value = []
        self.impurity = []
        self.n_node_samples =[]
        self.weighted_n_node_samples =[]
        self.n_outputs = 1
        self.n_classes = None

# Inherit from BaseEstimator to fix export_graphviz _sklearn_tags_ errors
class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.root = None
        self._node_count = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_features_in_ = X.shape[1]
        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_classes_ = len(self.classes_)
        self.tree_ = TreeStructure()
        self._node_count = 0
        self.root = self._build_tree(X, y, depth=0)
        self._build_sklearn_tree_structure()
        self._calculate_feature_importances()
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            class_counts = np.zeros((1, self.n_classes_))
            for i, cls in enumerate(self.classes_):
                class_counts[0, i] = np.sum(y == cls)
            return Node(value=class_counts)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            class_counts = np.zeros((1, self.n_classes_))
            for i, cls in enumerate(self.classes_):
                class_counts[0, i] = np.sum(y == cls)
            return Node(value=class_counts)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            value=self._node_value(y)
        )

    def _node_value(self, y):
        class_counts = np.zeros((1, self.n_classes_))
        for i, cls in enumerate(self.classes_):
            class_counts[0, i] = np.sum(y == cls)
        return class_counts

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return 1 - np.sum(probabilities**2)

    def _entropy(self, y):
        m = len(y)
        if m == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _calculate_impurity(self, y):
        if self.criterion == 'gini': return self._gini(y)
        else: return self._entropy(y)

    def _find_best_split(self, X, y):
        m, n = X.shape
        best_gain = -np.inf
        best_feature, best_threshold = None, None
        current_impurity = self._calculate_impurity(y)

        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                gain = current_impurity - (n_left/m * left_impurity + n_right/m * right_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_sklearn_tree_structure(self):
        self.tree_ = TreeStructure()
        self.tree_.n_outputs = 1
        if hasattr(self, 'n_classes_'):
            self.tree_.n_classes = np.array([self.n_classes_], dtype=np.intp)
        else:
            self.tree_.n_classes = np.array([1], dtype=np.intp)

        self.tree_.children_left = []
        self.tree_.children_right =[]
        self.tree_.feature = []
        self.tree_.threshold = []
        self.tree_.value =[]
        self.tree_.impurity = []
        self.tree_.n_node_samples =[]
        self.tree_.weighted_n_node_samples = []

        nodes =[]
        self._collect_nodes_depth_first(self.root, nodes)
        node_to_id = {node: i for i, node in enumerate(nodes)}

        for i, node in enumerate(nodes):
            if node.feature is not None:
                self.tree_.feature.append(node.feature)
                self.tree_.threshold.append(node.threshold)
                counts = node.value[0]
                total = np.sum(counts)
                if total > 0:
                    probs = counts / total
                    if self.criterion == 'gini': impurity = 1 - np.sum(probs * probs)
                    else: impurity = -np.sum(probs * np.log2(probs + 1e-10))
                else:
                    impurity = 0.0
            else:
                self.tree_.feature.append(-2)
                self.tree_.threshold.append(-2)
                counts = node.value[0]
                total = np.sum(counts)
                if total > 0:
                    probs = counts / total
                    if self.criterion == 'gini': impurity = 1 - np.sum(probs * probs)
                    else: impurity = -np.sum(probs * np.log2(probs + 1e-10))
                else:
                    impurity = 0.0

            self.tree_.value.append(node.value)
            self.tree_.impurity.append(impurity)
            sample_count = np.sum(node.value)
            self.tree_.n_node_samples.append(int(sample_count))
            self.tree_.weighted_n_node_samples.append(float(sample_count))

            if node.left is not None: self.tree_.children_left.append(node_to_id[node.left])
            else: self.tree_.children_left.append(-1)
            if node.right is not None: self.tree_.children_right.append(node_to_id[node.right])
            else: self.tree_.children_right.append(-1)

        self.tree_.node_count = len(nodes)

    def _collect_nodes_depth_first(self, node, nodes_list):
        if node is None: return
        nodes_list.append(node)
        self._collect_nodes_depth_first(node.left, nodes_list)
        self._collect_nodes_depth_first(node.right, nodes_list)

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: return self._predict_single(X)
        else: return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node = self.root
        while node.feature is not None:
            if x[node.feature] <= node.threshold: node = node.left
            else: node = node.right
        return self.classes_[np.argmax(node.value)]

    def decision_path(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        indices = []
        indptr =[0]
        for i in range(n_samples):
            path = self._get_decision_path(X[i])
            indices.extend(path)
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=np.int8)
        n_nodes = self.tree_.node_count
        result = sparse.csr_matrix((data, indices, indptr), shape=(n_samples, n_nodes))
        return result

    def _get_decision_path(self, x):
        path =[]
        node_id = 0
        while node_id != -1:
            path.append(node_id)
            if self.tree_.feature[node_id] < 0: break
            if x[self.tree_.feature[node_id]] <= self.tree_.threshold[node_id]:
                node_id = self.tree_.children_left[node_id]
            else:
                node_id = self.tree_.children_right[node_id]
        return path

    def apply(self, X):
        X = np.array(X)
        if X.ndim == 1: return np.array([self._get_leaf_id(X)])
        else: return np.array([self._get_leaf_id(x) for x in X])

    def _get_leaf_id(self, x):
        node_id = 0
        while True:
            if self.tree_.feature[node_id] < 0: return node_id
            if x[self.tree_.feature[node_id]] <= self.tree_.threshold[node_id]:
                node_id = self.tree_.children_left[node_id]
            else:
                node_id = self.tree_.children_right[node_id]

    def get_depth(self):
        def _get_node_depth(node_id):
            if node_id == -1: return 0
            left_depth = _get_node_depth(self.tree_.children_left[node_id]) if self.tree_.children_left[node_id] >= 0 else 0
            right_depth = _get_node_depth(self.tree_.children_right[node_id]) if self.tree_.children_right[node_id] >= 0 else 0
            return max(left_depth, right_depth) + 1
        return _get_node_depth(0) if self.tree_ is not None else 0

    def get_n_leaves(self):
        if self.tree_ is None: return 0
        count = 0
        for i in range(self.tree_.node_count):
            if self.tree_.children_left[i] == -1 and self.tree_.children_right[i] == -1:
                count += 1
        return count

    def _calculate_feature_importances(self):
        feature_counts = np.zeros(self.n_features_in_)
        for feature_idx in self.tree_.feature:
            if feature_idx >= 0: feature_counts[feature_idx] += 1
        if np.sum(feature_counts) > 0: self.feature_importances_ = feature_counts / np.sum(feature_counts)
        else: self.feature_importances_ = np.ones(self.n_features_in_) / self.n_features_in_


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.root = None
        self._node_count = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_features_in_ = X.shape[1]
        self.tree_ = TreeStructure()
        self._node_count = 0
        self.root = self._build_tree(X, y, depth=0)
        self._build_sklearn_tree_structure()
        self._calculate_feature_importances()
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           np.all(y == y[0]):
            return Node(value=np.array([[np.mean(y)]]))

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return Node(value=np.array([[np.mean(y)]]))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            value=np.array([[np.mean(y)]])
        )

    def _mse(self, y): return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0
    def _mae(self, y): return np.mean(np.abs(y - np.mean(y))) if len(y) > 0 else 0

    def _calculate_impurity(self, y):
        if self.criterion == 'mse': return self._mse(y)
        else: return self._mae(y)

    def _find_best_split(self, X, y):
        m, n = X.shape
        best_loss_reduction = -np.inf
        best_feature, best_threshold = None, None
        current_impurity = self._calculate_impurity(y)

        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                loss_reduction = current_impurity - (n_left/m * left_impurity + n_right/m * right_impurity)
                if loss_reduction > best_loss_reduction:
                    best_loss_reduction = loss_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_sklearn_tree_structure(self):
        self.tree_ = TreeStructure()
        self.tree_.n_outputs = 1
        self.tree_.n_classes = np.array([1], dtype=np.intp)
        self.tree_.children_left = []
        self.tree_.children_right = []
        self.tree_.feature =[]
        self.tree_.threshold = []
        self.tree_.value = []
        self.tree_.impurity =[]
        self.tree_.n_node_samples =[]
        self.tree_.weighted_n_node_samples = []

        nodes =[]
        self._collect_nodes_depth_first(self.root, nodes)
        node_to_id = {node: i for i, node in enumerate(nodes)}

        for node in nodes:
            if node.feature is not None:
                self.tree_.feature.append(node.feature)
                self.tree_.threshold.append(node.threshold)
            else:
                self.tree_.feature.append(-2)
                self.tree_.threshold.append(-2)
            self.tree_.value.append(node.value)
            self.tree_.impurity.append(0.0)
            sample_count = np.sum(node.value)
            self.tree_.n_node_samples.append(int(sample_count))
            self.tree_.weighted_n_node_samples.append(float(sample_count))

            if node.left is not None: self.tree_.children_left.append(node_to_id[node.left])
            else: self.tree_.children_left.append(-1)
            if node.right is not None: self.tree_.children_right.append(node_to_id[node.right])
            else: self.tree_.children_right.append(-1)

        self.tree_.node_count = len(nodes)

    def _collect_nodes_depth_first(self, node, nodes_list):
        if node is None: return
        nodes_list.append(node)
        self._collect_nodes_depth_first(node.left, nodes_list)
        self._collect_nodes_depth_first(node.right, nodes_list)

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1: return self._predict_single(X)
        else: return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node = self.root
        while node.feature is not None:
            if x[node.feature] <= node.threshold: node = node.left
            else: node = node.right
        return node.value[0][0]

    def _calculate_feature_importances(self):
        feature_counts = np.zeros(self.n_features_in_)
        for feature_idx in self.tree_.feature:
            if feature_idx >= 0: feature_counts[feature_idx] += 1
        if np.sum(feature_counts) > 0: self.feature_importances_ = feature_counts / np.sum(feature_counts)
        else: self.feature_importances_ = np.ones(self.n_features_in_) / self.n_features_in_

    def decision_path(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        indices = []
        indptr = [0]
        for i in range(n_samples):
            path = self._get_decision_path(X[i])
            indices.extend(path)
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=np.int8)
        n_nodes = self.tree_.node_count
        result = sparse.csr_matrix((data, indices, indptr), shape=(n_samples, n_nodes))
        return result

    def _get_decision_path(self, x):
        path =[]
        node_id = 0
        while node_id != -1:
            path.append(node_id)
            if self.tree_.feature[node_id] < 0: break
            if x[self.tree_.feature[node_id]] <= self.tree_.threshold[node_id]:
                node_id = self.tree_.children_left[node_id]
            else: node_id = self.tree_.children_right[node_id]
        return path

    def apply(self, X):
        X = np.array(X)
        if X.ndim == 1: return np.array([self._get_leaf_id(X)])
        else: return np.array([self._get_leaf_id(x) for x in X])

    def _get_leaf_id(self, x):
        node_id = 0
        while True:
            if self.tree_.feature[node_id] < 0: return node_id
            if x[self.tree_.feature[node_id]] <= self.tree_.threshold[node_id]:
                node_id = self.tree_.children_left[node_id]
            else: node_id = self.tree_.children_right[node_id]

    def get_depth(self):
        def _get_node_depth(node_id):
            if node_id == -1: return 0
            left_depth = _get_node_depth(self.tree_.children_left[node_id]) if self.tree_.children_left[node_id] >= 0 else 0
            right_depth = _get_node_depth(self.tree_.children_right[node_id]) if self.tree_.children_right[node_id] >= 0 else 0
            return max(left_depth, right_depth) + 1
        return _get_node_depth(0) if self.tree_ is not None else 0

    def get_n_leaves(self):
        if self.tree_ is None: return 0
        count = 0
        for i in range(self.tree_.node_count):
            if self.tree_.children_left[i] == -1 and self.tree_.children_right[i] == -1: count += 1
        return count


# --- API ENDPOINT FUNCTIONS WITH ADDED HISTORY LOOP ---

def run_decision_tree_classification(data, max_depth=3, min_samples_split=2, criterion='gini'):
    try:
        X = np.array(data['X'])
        y = np.array(data['y']).astype(str)

        history =[]
        final_model = None

        # Build trees incrementally from depth 1 to max_depth to create the Validation Curve history
        for d in range(1, max_depth + 1):
            clf = DecisionTreeClassifier(
                max_depth=d,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=42
            )
            clf.fit(X, y)

            y_pred = clf.predict(X)
            accuracy = accuracy_score(y, y_pred)

            tree_img = generate_tree_visualization(clf, feature_names=['x1', 'x2'], class_names=np.unique(y), X=X, y=y)
            boundary_img = generate_optimized_decision_boundary(clf, X, y)

            history.append({
                'depth': d,
                'accuracy': float(accuracy),
                'n_nodes': clf.tree_.node_count,
                'n_leaves': clf.get_n_leaves(),
                'decision_boundary': boundary_img,
                'tree_visualization': tree_img
            })

            if d == max_depth:
                final_model = clf

        feature_importances = final_model.feature_importances_.tolist()

        decision_paths = []
        for i, sample in enumerate(X[:5]):
            path = get_decision_path(final_model, sample)
            decision_paths.append({
                'sample_idx': i,
                'path': path,
                'prediction': final_model.predict([sample])[0]
            })

        return {
            'model_type': 'classification',
            'accuracy': float(history[-1]['accuracy']),
            'n_nodes': final_model.tree_.node_count,
            'n_leaves': final_model.get_n_leaves(),
            'max_depth': max_depth,
            'feature_importances': feature_importances,
            'tree_visualization': history[-1]['tree_visualization'],
            'decision_boundary': history[-1]['decision_boundary'],
            'example_paths': decision_paths,
            'history': history
        }
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}

def run_decision_tree_regression(data, max_depth=3, min_samples_split=2):
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])

        if len(X) < 2: return {"error": "Need at least 2 training points"}

        history =[]
        final_model = None

        # Build trees incrementally from depth 1 to max_depth
        for d in range(1, max_depth + 1):
            regressor = DecisionTreeRegressor(
                max_depth=d,
                min_samples_split=min_samples_split,
                random_state=42
            )
            regressor.fit(X, y)

            y_pred = regressor.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            tree_img = generate_tree_visualization(regressor, feature_names=['x1', 'x2'], regression=True, X=X, y=y)
            decision_boundary = generate_regression_surface(regressor, X, y)

            history.append({
                'depth': d,
                'mse': float(mse),
                'accuracy': float(r2), # Re-using accuracy field for R2 score in graphs
                'n_nodes': regressor.tree_.node_count,
                'n_leaves': regressor.get_n_leaves(),
                'decision_boundary': decision_boundary,
                'tree_visualization': tree_img
            })

            if d == max_depth:
                final_model = regressor

        return {
            'model_type': 'regression',
            'mse': float(history[-1]['mse']),
            'r2': float(history[-1]['accuracy']),
            'n_nodes': final_model.tree_.node_count,
            'n_leaves': final_model.get_n_leaves(),
            'max_depth': max_depth,
            'feature_importances': final_model.feature_importances_.tolist(),
            'tree_visualization': history[-1]['tree_visualization'],
            'decision_boundary': history[-1]['decision_boundary'],
            'history': history
        }
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}

def predict_data_points(data, predict_points, max_depth=3, min_samples_split=2, criterion='gini'):
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        predict_X = np.array(predict_points)

        is_regression = False
        try:
            float_y = np.array([float(val) for val in y])
            is_regression = True
            y = float_y
        except (ValueError, TypeError):
            is_regression = False

        if is_regression:
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        else:
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=42)

        model.fit(X, y)
        predictions = model.predict(predict_X)

        prediction_paths =[]
        for i, point in enumerate(predict_X):
            path = get_decision_path(model, point)
            prediction_paths.append({
                'point_idx': i,
                'path': path,
                'prediction': str(predictions[i]) if not is_regression else float(predictions[i])
            })

        if is_regression:
            tree_img = generate_tree_visualization(model, feature_names=['x1', 'x2'], regression=True, X=X, y=y)
            decision_boundary = generate_regression_surface(model, X, y)
            predictions_list = predictions.tolist()
        else:
            tree_img = generate_tree_visualization(model, feature_names=['x1', 'x2'], class_names=np.unique(y), X=X, y=y)
            decision_boundary = generate_optimized_decision_boundary(model, X, y)
            predictions_list = [str(p) for p in predictions]

        class_mapping = {}
        if not is_regression:
            unique_classes = np.unique(y)
            for i, cls in enumerate(unique_classes):
                class_mapping[i] = str(cls)

        return {
            'predictions': predictions_list,
            'prediction_paths': prediction_paths,
            'tree_visualization': tree_img,
            'decision_boundary': decision_boundary,
            'model_type': 'regression' if is_regression else 'classification',
            'class_mapping': class_mapping
        }
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}

def get_decision_path(tree_model, sample):
    node_indicator = tree_model.decision_path([sample])
    leaf_id = tree_model.apply([sample])[0]
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    path_info =[]
    for i, node_id in enumerate(node_index):
        if i < len(node_index) - 1:
            feature = tree_model.tree_.feature[node_id]
            threshold = tree_model.tree_.threshold[node_id]
            path_info.append({
                'node_id': int(node_id),
                'feature': int(feature),
                'threshold': float(threshold),
                'goes_left': sample[feature] <= threshold
            })
        else:
            value = tree_model.tree_.value[node_id]
            path_info.append({
                'node_id': int(node_id),
                'is_leaf': True,
                'value': value.tolist()
            })
    return path_info

def generate_tree_visualization(model, feature_names=None, class_names=None, regression=False, X=None, y=None):
    from sklearn.tree import DecisionTreeClassifier as SkDTC
    from sklearn.tree import DecisionTreeRegressor as SkDTR
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    import io
    import base64

    # We silently train a native tree with the exact same parameters just to draw the graph
    if regression:
        fallback_model = SkDTR(max_depth=model.max_depth, min_samples_split=model.min_samples_split, random_state=42)
        fallback_model.fit(X, y)
    else:
        fallback_model = SkDTC(max_depth=model.max_depth, min_samples_split=model.min_samples_split, criterion=model.criterion, random_state=42)
        fallback_model.fit(X, y)

    # Create a nice large figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    # Use scikit-learn's built-in matplotlib tree plotter
    plot_tree(fallback_model, filled=True, feature_names=feature_names, class_names=class_names,
              ax=ax, rounded=True, precision=2, fontsize=10)

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=False)
    plt.close(fig)
    buf.seek(0)

    # FIX: Return raw base64 string without the HTML prefix!
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_optimized_decision_boundary(tree_model, X, y):
    try:
        h = 0.05
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        Z = tree_model.predict(mesh_points)

        if not np.issubdtype(Z.dtype, np.number):
            unique_z = np.unique(Z)
            z_map = {val: i for i, val in enumerate(unique_z)}
            Z = np.array([z_map[val] for val in Z], dtype=np.float64)

        Z = Z.reshape(xx.shape)
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        colors =['#3B82F6', '#EF4444', '#22C55E', '#F59E0B', '#8B5CF6']
        if n_classes <= len(colors): cmap_light = ListedColormap(colors[:n_classes])
        else: cmap_light = plt.cm.rainbow

        plt.figure(figsize=(6, 4.5), dpi=80)
        plt.contourf(xx, yy, Z.astype(float), alpha=0.3, cmap=cmap_light)

        for i, cls in enumerate(unique_classes):
            color = colors[i] if i < len(colors) else plt.cm.rainbow(i / n_classes)
            idx = np.where(y == cls)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=f'Class {cls}', alpha=0.8, edgecolor='k', s=30)

        plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())
        plt.title('Decision Boundary')
        plt.xlabel('x1'); plt.ylabel('x2')
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()

        buffer = BytesIO()
        # FIXED: Removed optimize=True
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1, transparent=False)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    except Exception as e:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Error generating boundary:\n{str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

def generate_decision_boundary(tree_model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    try:
        mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        Z = tree_model.predict(mesh_points)
        if not np.issubdtype(Z.dtype, np.number):
            unique_z = np.unique(Z)
            z_map = {val: i for i, val in enumerate(unique_z)}
            Z = np.array([z_map[val] for val in Z], dtype=np.float64)
        Z = Z.reshape(xx.shape)
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        colors =['#3B82F6', '#EF4444', '#22C55E', '#F59E0B', '#8B5CF6']
        if n_classes <= len(colors): cmap_light = ListedColormap(colors[:n_classes])
        else: cmap_light = plt.cm.rainbow

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z.astype(float), alpha=0.3, cmap=cmap_light)

        for i, cls in enumerate(unique_classes):
            color = colors[i] if i < len(colors) else plt.cm.rainbow(i / n_classes)
            idx = np.where(y == cls)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=f'Class {cls}', alpha=0.8, edgecolor='k')

        plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
        plt.legend(); plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error generating decision boundary:\n{str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

def generate_regression_surface(tree_model, X, y):
    try:
        h = 0.05
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        Z = tree_model.predict(mesh_points).reshape(xx.shape)

        plt.figure(figsize=(6, 4.5), dpi=80)
        contour = plt.contourf(xx, yy, Z, 20, cmap='viridis', alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=30, alpha=1.0)
        cbar = plt.colorbar(contour, shrink=0.8); cbar.set_label('Predicted Value')

        plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())
        plt.title('Regression Surface')
        plt.xlabel('x1'); plt.ylabel('x2'); plt.tight_layout()

        buffer = BytesIO()
        # FIXED: Removed optimize=True
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1, transparent=False)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Error generating regression surface:\n{str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

def generate_regression_scatter_plot(X, y, predictions):
    try:
        plt.figure(figsize=(6, 4.5), dpi=80)
        norm = plt.Normalize(min(y), max(y))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm); sm.set_array([])
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, alpha=0.8, edgecolor='k', s=30)

        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = predictions.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
        plt.colorbar(sm, label='Value')
        plt.xlim(xx.min(), xx.max()); plt.ylim(yy.min(), yy.max())
        plt.title('Regression Predictions'); plt.xlabel('x1'); plt.ylabel('x2'); plt.tight_layout()

        buffer = BytesIO()
        # FIXED: Removed optimize=True
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1, transparent=False)
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Error generating regression visualization:\n{str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

def generate_sample_classification_data(dataset_type='blobs', n_samples=40, n_clusters=3, variance=0.5):
    np.random.seed(42)
    from sklearn.datasets import make_blobs, make_moons, make_circles

    if (dataset_type == 'moons'):
        X, y = make_moons(n_samples=n_samples*2, noise=variance*0.1, random_state=42)
        X = X * 4.5 - 2
        y = [str(int(i)) for i in y]
    elif (dataset_type == 'circles'):
        X, y = make_circles(n_samples=n_samples*2, noise=variance*0.1, factor=0.5, random_state=42)
        X = X * 6.5
        y =[str(int(i)) for i in y]
    else:
        centers =[]
        for i in range(n_clusters):
            angle = i * (2 * np.pi / n_clusters)
            centers.append([4 * np.cos(angle), 4 * np.sin(angle)])
        X, y_numeric = make_blobs(n_samples=n_samples*n_clusters, centers=centers, cluster_std=variance*1.5, random_state=42)
        y =[str(int(i)) for i in y_numeric]
    return {'X': X.tolist(), 'y': y}

def generate_sample_regression_data(dataset_type='nonlinear', n_samples=40, variance=0.5, sparsity=1.0):
    np.random.seed(42)
    if sparsity > 1.0:
        centers =[]
        num_clusters = min(5, int(sparsity * 2))
        for _ in range(num_clusters): centers.append([np.random.uniform(-7, 7), np.random.uniform(-7, 7)])
        X =[]
        points_per_cluster = n_samples // num_clusters
        remainder = n_samples % num_clusters
        for i, center in enumerate(centers):
            cluster_points = points_per_cluster + (1 if i < remainder else 0)
            cluster_variance = 2.0 / sparsity
            cluster_x = center[0] + np.random.normal(0, cluster_variance, cluster_points)
            cluster_y = center[1] + np.random.normal(0, cluster_variance, cluster_points)
            for j in range(cluster_points): X.append([cluster_x[j], cluster_y[j]])
        X = np.array(X)
    else:
        range_scale = 16.0 * sparsity
        X = np.random.uniform(-range_scale/2, range_scale/2, (n_samples, 2))

    if dataset_type == 'linear':
        y = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1]
        y += np.random.normal(0, variance * 0.5, n_samples)
    else:
        y = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * (X[:, 0]**2) - 0.1 * (X[:, 0] * X[:, 1])
        y += np.random.normal(0, variance * 2.0, n_samples)
    return {'X': X.tolist(), 'y': y.tolist()}

def generate_tree_with_highlighted_path(tree_model, X, path_indices, feature_names=None, class_names=None, regression=False):
    dotfile = StringIO()
    export_graphviz(tree_model, out_file=dotfile, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, special_characters=True, impurity=not regression, precision=2)
    dot_data = dotfile.getvalue()
    for idx in path_indices:
        dot_data = dot_data.replace(f'node[shape=box, style="filled', f'node{idx}[shape=box, color=red, penwidth=3.0, style="filled')
    (graph,) = pydot.graph_from_dot_data(dot_data)
    png_bytes = graph.create_png()
    return base64.b64encode(png_bytes).decode('utf-8')
