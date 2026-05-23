import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from sklearn.cluster import DBSCAN
import base64
from io import BytesIO
import time
import traceback

def run_dbscan(points, eps=0.5, min_samples=5,
               show_core_points=True, show_border_points=True,
               show_noise_points=True, show_epsilon_radius=True):
    """
    Run the DBSCAN algorithm and return the results along with step-by-step iterations and graph data.
    """
    try:
        # Convert input points to numpy array
        X = np.array([[point['x'], point['y']] for point in points])

        # Create an interactive DBSCAN object
        interactive_dbscan = InteractiveDBSCAN(eps=eps, min_samples=min_samples)

        # Run the algorithm and collect iterations
        iterations = interactive_dbscan.run(X)

        # Calculate cluster sizes for BarChart
        cluster_sizes =[]
        if hasattr(interactive_dbscan, 'labels_'):
            labels = interactive_dbscan.labels_
            unique_labels = set(labels)
            for lbl in unique_labels:
                if lbl == -2: continue # Skip unvisited
                count = int(np.sum(labels == lbl))
                name = "Noise" if lbl == -1 else f"Cluster {lbl}"
                cluster_sizes.append({"cluster": name, "size": count})

            # Sort clusters by index, putting Noise at the end
            cluster_sizes.sort(key=lambda x: float('inf') if x["cluster"] == "Noise" else int(x["cluster"].split(" ")[1]))

        # Calculate progression history for AreaChart
        progress_history =[]
        for i, it in enumerate(iterations):
            progress_history.append({
                "iteration": i,
                "Core": len(it["core_points"]),
                "Border": len(it["border_points"]),
                "Noise": len(it["noise_points"])
            })

        # Return the results
        result = {
            "status": "success",
            "iterations": iterations,
            "progress_history": progress_history,
            "cluster_sizes": cluster_sizes,
            "final_labels": interactive_dbscan.labels_.tolist() if hasattr(interactive_dbscan, 'labels_') else[],
            "num_clusters": len(set(interactive_dbscan.labels_)) - (1 if -1 in interactive_dbscan.labels_ else 0) if hasattr(interactive_dbscan, 'labels_') else 0,
            "core_points": [int(x) for x in interactive_dbscan.core_points],
            "border_points": [int(x) for x in interactive_dbscan.border_points],
            "noise_points": [int(x) for x in interactive_dbscan.noise_points]
        }

        return result

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in run_dbscan: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return {"status": "error", "message": str(e), "traceback": error_traceback}

class InteractiveDBSCAN:
    """
    A custom implementation of DBSCAN algorithm that captures intermediate steps for visualization.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.X = None
        self.labels_ = None
        self.core_points = []
        self.border_points = []
        self.noise_points =[]

    def run(self, X):
        self.X = X
        n_samples = X.shape[0]

        # Initialize all points as unvisited (-2 means unvisited)
        self.labels_ = np.full(n_samples, -2, dtype=int)
        iterations =[]

        iterations.append(self._create_iteration_data(
            phase="initialization", labels=self.labels_.tolist(), current_point=None, neighbors=[],
            message="Algorithm initialized. Ready to start clustering."
        ))

        cluster_id = 0

        for point_idx in range(n_samples):
            if self.labels_[point_idx] != -2: continue

            neighbors = self._find_neighbors(point_idx)
            iterations.append(self._create_iteration_data(
                phase="evaluating_point", labels=self.labels_.tolist(), current_point=point_idx, neighbors=neighbors.tolist(),
                message=f"Evaluating point {point_idx}. Found {len(neighbors)} neighbors."
            ))

            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
                self.noise_points.append(point_idx)
                iterations.append(self._create_iteration_data(
                    phase="noise_identified", labels=self.labels_.tolist(), current_point=point_idx, neighbors=neighbors.tolist(),
                    message=f"Point {point_idx} classified as noise (insufficient neighbors)."
                ))
                continue

            cluster_id += 1
            self.labels_[point_idx] = cluster_id - 1
            self.core_points.append(point_idx)

            iterations.append(self._create_iteration_data(
                phase="new_cluster_formed", labels=self.labels_.tolist(), current_point=point_idx, neighbors=neighbors.tolist(),
                message=f"New cluster {cluster_id-1} started with core point {point_idx}."
            ))

            seed_points = list(neighbors)
            processed_indices =[]

            while seed_points:
                current_seed = seed_points.pop(0)
                if current_seed in processed_indices: continue
                processed_indices.append(current_seed)

                if self.labels_[current_seed] != -2: continue
                self.labels_[current_seed] = cluster_id - 1

                neighbor_neighbors = self._find_neighbors(current_seed)
                iterations.append(self._create_iteration_data(
                    phase="evaluating_neighbor", labels=self.labels_.tolist(), current_point=current_seed, neighbors=neighbor_neighbors.tolist(),
                    message=f"Evaluating neighbor point {current_seed}. Found {len(neighbor_neighbors)} neighbors."
                ))

                if len(neighbor_neighbors) >= self.min_samples:
                    self.core_points.append(current_seed)
                    for secondary_neighbor in neighbor_neighbors:
                        if self.labels_[secondary_neighbor] == -2 and secondary_neighbor not in processed_indices:
                            seed_points.append(secondary_neighbor)
                    iterations.append(self._create_iteration_data(
                        phase="expanding_cluster", labels=self.labels_.tolist(), current_point=current_seed, neighbors=neighbor_neighbors.tolist(),
                        message=f"Expanding cluster {cluster_id-1} with core point {current_seed}."
                    ))
                else:
                    self.border_points.append(current_seed)
                    iterations.append(self._create_iteration_data(
                        phase="border_point_identified", labels=self.labels_.tolist(), current_point=current_seed, neighbors=neighbor_neighbors.tolist(),
                        message=f"Point {current_seed} classified as border point in cluster {cluster_id-1}."
                    ))

        self.core_points = list(set(self.core_points))
        self.border_points = list(set(self.border_points))
        self.noise_points = list(set(self.noise_points))

        iterations.append(self._create_iteration_data(
            phase="completed", labels=self.labels_.tolist(), current_point=None, neighbors=[],
            message=f"Clustering complete! Found {cluster_id} clusters, {len(self.core_points)} core points, {len(self.border_points)} border points, and {len(self.noise_points)} noise points."
        ))

        return iterations

    def _find_neighbors(self, point_idx):
        distances = np.sqrt(np.sum((self.X - self.X[point_idx])**2, axis=1))
        return np.where(distances <= self.eps)[0]

    def _create_iteration_data(self, phase, labels, current_point, neighbors, message=""):
        processed_labels =[int(l) for l in labels]
        processed_neighbors = [int(n) for n in neighbors] if neighbors is not None else[]
        processed_count = sum(1 for l in labels if l != -2)
        cluster_labels = set(l for l in labels if l >= 0)

        return {
            "phase": phase,
            "labels": processed_labels,
            "current_point": int(current_point) if current_point is not None else None,
            "neighbors": processed_neighbors,
            "core_points": [int(cp) for cp in self.core_points],
            "border_points":[int(bp) for bp in self.border_points],
            "noise_points":[int(np) for np in self.noise_points],
            "num_clusters": int(len(cluster_labels)),
            "num_processed": int(processed_count),
            "message": message
        }

def generate_dbscan_data(dataset_type='blobs', n_samples=100, n_clusters=3, noise_level=0.05):
    from sklearn.datasets import make_blobs, make_moons, make_circles
    np.random.seed(42)

    if dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
        X = X * 5 - 2.5
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise_level, factor=0.5, random_state=42)
        X = X * 5
    elif dataset_type == 'anisotropic':
        X, _ = make_blobs(n_samples=n_samples, centers=1, random_state=42)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation) * 4
    elif dataset_type == 'noisy_circles':
        X, _ = make_circles(n_samples=n_samples, factor=0.5, noise=noise_level*2, random_state=42)
        n_outliers = int(n_samples * 0.1)
        outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
        X = np.vstack([X * 5, outliers])
    else:
        centers =[]
        for i in range(n_clusters):
            angle = i * (2 * np.pi / n_clusters)
            centers.append([4 * np.cos(angle), 4 * np.sin(angle)])
        X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=noise_level*5 + 0.3, random_state=42)

    return {"points":[{"x": float(x), "y": float(y)} for x, y in X]}
