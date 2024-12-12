import optuna
from datetime import datetime
import os
import hdbscan
import pickle
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import pylab as plt
from loguru import logger
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, make_scorer, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.decorators import deprecated


# Tienen que estar fuera, si no GridSearch no funciona bien. Definir archivo con estas funciones fuera, o en utils
def silhouette_scorer(estimator, X, y=None):
    # Obtiene las etiquetas predichas por el modelo y filtra el ruido (-1)
    if hasattr(estimator, 'labels_'):
        labels = estimator.labels_
    else:
        labels = estimator.fit_predict(X)
    if len(set(labels)) > 1 and np.all(labels != -1):
        return silhouette_score(X, labels)
    return -1  # Valor por defecto si no se puede calcular

def davies_bouldin_scorer(estimator, X, y=None):
    # Obtiene las etiquetas predichas por el modelo y filtra el ruido (-1)
    if hasattr(estimator, 'labels_'):
        labels = estimator.labels_
    else:
        labels = estimator.fit_predict(X)
    if len(set(labels)) > 1 and np.all(labels != -1):
        return davies_bouldin_score(X, labels)
    return float('inf')  # Valor alto si no se puede calcular


class ClusteringModel(ABC):
    """
    Base abstract class for clustering models.

    This class provides a template for implementing clustering models with methods
    for running the clustering, saving clustering results and plots, and performing
    basic PCA for 2D representation of clusters.
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 model_name: str):
        """
        Initialize the clustering model with data.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset on which clustering will be performed.
        """
        self.data = data
        self.model_name = model_name
        # Setting up directories for saving results based on model_name
        # Theese are base directories. Every model will define their results
        # directory tree        
            
    
    def run_single_experiment(self, params, eval_method):
        """
        Run a single clustering experiment based on the model specified by self.model_name.

        Parameters
        ----------
        params : dict
            Hyperparameters for the clustering model.
        eval_method : str
            Evaluation method for the clustering ('silhouette' or 'davies_bouldin').

        Returns
        -------
        labels : np.ndarray
            Cluster labels assigned to each data point.
        centers : np.ndarray
            Calculated cluster centers.
        score : float
            Clustering score based on the specified evaluation method.
        """
        np.random.seed(42)
        if self.model_name == "kmeans":
            # KMeans clustering
            model = KMeans(**params)
        elif self.model_name == "hdbscan":
            # HDBSCAN clustering
            model = hdbscan.HDBSCAN(**params)
        elif self.model_name == "dbscan":
            # HDBSCAN clustering
            model = DBSCAN(**params)
        elif self.model_name == "agglomerative":
            # Agglomerative clustering
            model = AgglomerativeClustering(**params)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported. Choose from 'kmeans', 'hdbscan', or 'agglomerative'.")
        
        labels = model.fit_predict(self.data)
        centers = self.get_cluster_centers(labels)

        # Calculate the score based on the specified evaluation method
        if eval_method == "silhouette" and len(set(labels)) > 1:
            score = silhouette_score(self.data[labels != -1], labels[labels != -1])
        elif eval_method == "davies_bouldin" and len(set(labels)) > 1:
            score = davies_bouldin_score(self.data[labels != -1], labels[labels != -1])
        else:
            score = None  # Not enough clusters for valid scoring or unsupported eval_method

        return labels, centers, score
    
    
    def get_cluster_centers(self, labels):
        """
        Calculate the centers of clusters given data points and their cluster labels.

        For density-based clustering, this function finds the densest point or a representative point
        near the cluster's central area. For centroid-based methods, it calculates the mean of the 
        points within each cluster to approximate the centroid.

        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Cluster labels assigned to each data point, with each unique label representing 
            a separate cluster. Points labeled as -1 are typically considered noise.

        Returns
        -------
        centers : numpy.ndarray of shape (n_clusters, n_features)
            An array containing the calculated center of each cluster.
        """
        unique_labels = np.unique(labels)
        centers = []
        for label in unique_labels:
            if label == -1:  # Ignore noise
                continue
            
            cluster_points = self.data.values[labels == label]
                
            if self.model_name in ["hdbscan", "dbscan"]:  # Density-based methods
                # Find the point with highest density within the cluster
                # Check if cluster has less values than neighbors
                n_neighbors_cluster = min(5, max(1, len(cluster_points) - 1))
                nbrs = NearestNeighbors(n_neighbors=n_neighbors_cluster).fit(cluster_points)
                densities = np.mean(nbrs.kneighbors()[0], axis=1)
                densest_point_idx = np.argmin(densities)  # Lower distance to neighbors = higher density
                cluster_center = cluster_points[densest_point_idx]
            else:  # For centroid-based methods like KMeans or Agglomerative
                cluster_center = np.mean(cluster_points, axis=0)

            centers.append(cluster_center)

        if centers:
            centers = np.array(centers)
        
        return centers
            
     
    
    def run_optuna_generic(self, model_builder, evaluation_method="silhouette", n_trials=100, penalty="linear", penalty_range=(2,8)):
        """
        Generic Optuna optimization for clustering models with configurable penalty types.

        Parameters
        ----------
        model_builder : Callable[[optuna.trial.Trial], clustering_model]
            Function that builds the clustering model with hyperparameters suggested by the Optuna trial.
        evaluation_method : str
            The evaluation metric to optimize ('silhouette' or 'davies_bouldin').
        n_trials : int
            The number of trials for Optuna. Default is 50.
        penalty : str, optional
            The type of penalty to apply based on `n_clusters`. Options are:
            - "linear": Linear penalty based on `n_clusters`.
            - "proportional": Proportional penalty inversely related to `n_clusters`.
            - "range": Proportional penalty only when `n_clusters` is outside a specified range.
        """
        
        # Define acceptable range for "range" penalty type
        if penalty_range is not None and penalty == "range":
            min_clusters, max_clusters = penalty_range
        
        # Objective function
        def objective(trial):
            # Build the model with suggested hyperparameters
            model = model_builder(trial)
            np.random.seed(42)
            # Fit and predict
            labels = model.fit_predict(self.data)
            
            # Get number of clusters (excluding noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            trial.set_user_attr("n_clusters", n_clusters)
            
            # Calculate cluster centers
            centers = self.get_cluster_centers(labels)
            trial.set_user_attr("centers", centers)  # Store centers in trial attributes
            trial.set_user_attr("labels", labels)

            
            # Evaluate model
            if n_clusters > 1:
                # Calculate the original score without penalty
                if evaluation_method in ["silhouette", "silhouette_noise"]:
                    score_original = silhouette_score(self.data[labels != -1], labels[labels != -1])
                elif evaluation_method in ["davies_bouldin", "davies_noise"]:
                    score_original = davies_bouldin_score(self.data[labels != -1], labels[labels != -1])
                else:
                    raise ValueError("Evaluation method not supported. Use 'silhouette' or 'davies_bouldin' instead.")

                # Noise points
                noise_points = (labels == -1).sum()
                # Calcular la proporción de ruido
                noise_ratio = noise_points / len(self.data)

                # If we choose to take noise as metric, calculate that score
                alpha = 0.3  # Ajustable según la importancia del ruido
                if evaluation_method == "silhouette_noise":
                    # Penalización proporcional al ruido
                    score_original = score_original - alpha * noise_ratio
                    # Garantizar que el score penalizado se mantenga dentro del rango de Silhouette [-1, 1]
                    score_original = max(score_original, -1.0)
                elif evaluation_method == "davies_noise":
                    # Penalización proporcional al ruido
                    score_original = score_original + alpha * noise_ratio


                # Check eval method has range if noise
                if (evaluation_method in ("silhouette_noise","davies_noise")):
                    if penalty == "" or penalty is None:
                        raise ValueError("When using eval methods optimizing noise, make sure to provide a penalty range.")


                if penalty == "linear":
                    adjustment = 0.1 * n_clusters
                    score_penalized = score_original - adjustment if evaluation_method in ["silhouette", "silhouette_noise"] else score_original + adjustment
                elif penalty == "proportional":
                    penalty_factor = 1 - (1 / n_clusters) if evaluation_method in ["silhouette", "silhouette_noise"] else 1 + (1 / n_clusters)
                    score_penalized = score_original * penalty_factor
                elif penalty == "range":
                    if n_clusters < min_clusters:
                        adjustment = 0.1 * (min_clusters - n_clusters)
                        score_penalized = score_original - adjustment if evaluation_method in ["silhouette", "silhouette_noise"] else score_original + adjustment
                    elif n_clusters > max_clusters:
                        adjustment = 0.1 * (n_clusters - max_clusters)
                        score_penalized = score_original - adjustment if evaluation_method in ["silhouette", "silhouette_noise"] else score_original + adjustment
                    else:
                        score_penalized = score_original
                else:
                    score_penalized = score_original
            else:
                score_original = -1 if evaluation_method in ["silhouette", "silhouette_noise"] else float('inf')
                score_penalized = score_original

            trial.set_user_attr("score_original", score_original)
            return score_penalized
        
        # Set optimization direction
        direction = "maximize" if "silhouette" in evaluation_method else "minimize"
        
        # Execute Optuna optimization with tqdm
        pbar = tqdm(total=n_trials, desc="Optuna Optimization")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda study, trial: pbar.update(1)])
        pbar.close()

        return study



    def run_grid_search_generic(self, param_grid, evaluation_method="silhouette"):
        """
        Generic GridSearchCV for clustering models with a specified evaluation method.

        Parameters
        ----------
        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter settings 
            to try as values.
        evaluation_method : str, optional
            The evaluation metric to optimize. Can be either 'silhouette' for maximizing 
            the silhouette score, or 'davies_bouldin' for minimizing the Davies-Bouldin score.
            Defaults to 'silhouette'.

        Returns
        -------
        GridSearchCV
            The GridSearchCV object after fitting, containing details of the best model 
            and its evaluation score.
        """

        if self.model_name == "kmeans":
            # KMeans clustering
            model = KMeans()
        elif self.model_name == "hdbscan":
            # HDBSCAN clustering
            model = hdbscan.HDBSCAN()
        elif self.model_name == "dbscan":
            # DBSCAN clustering
            model = DBSCAN()
        elif self.model_name == "agglomerative":
            # Agglomerative clustering
            model = AgglomerativeClustering()
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported. Choose from 'kmeans', 'hdbscan', or 'agglomerative'.")


        # Create and fit the GridSearchCV object
        grid_search = GridSearchCV(
            estimator=model,  # Se espera que cada clase específica defina `self.model_instance`
            param_grid=param_grid,
            scoring=silhouette_scorer if evaluation_method == "silhouette" else davies_bouldin_scorer,
            cv=[(slice(None), slice(None))],  # Usamos todos los datos sin CV
            n_jobs=-1
        )

        # Ejecuta la búsqueda
        grid_search.fit(self.data)
        
        # Log the best parameters and score
        # logger.info(f"Best Parameters: {grid_search.best_params_}")
        # logger.info(f"Best Score ({evaluation_method}): {grid_search.best_score_}")

        return grid_search





    def plot_single_experiment(
        self,
        X: pd.DataFrame, 
        c: Optional[np.ndarray] = None, 
        centroids: Optional[np.ndarray] = None,
        i: int = 0, 
        j: int = 0, 
        figs: Tuple[int, int] = (9, 7)):
        """
        Plots a 2D representation of the dataset and its associated clusters.

        Parameters
        ----------
        X : pd.DataFrame
            Data points to plot, with each row representing a sample in 2D space.
        c : Optional[np.ndarray]
            Cluster labels for each point.
        centroids : Optional[np.ndarray]
            Coordinates of cluster centroids in 2D space.
        i : int
            Index of the feature for the x-axis.
        j : int
            Index of the feature for the y-axis.
        figs : Tuple[int, int]
            Size of the figure in inches.
        """

        # color mapping for clusters
        colors = ['#FF0000', '#00FF00', '#FFFF00', '#0000FF', '#FF9D0A', '#00B6FF', '#F200FF', '#FF6100']
        cmap_bold = ListedColormap(colors)
        # Plotting frame
        plt.figure(figsize=figs)
        # Plotting points with seaborn
        sns.scatterplot(x=X.iloc[:, i], y=X.iloc[:, j], hue=c, palette=cmap_bold.colors, s=30, hue_order=sorted(set(c)))  # Ensures that -1 appears first in the legend if present)
        # Plotting centroids
        if centroids is not None:
            sns.scatterplot(x=centroids[:, i], y=centroids[:, j], marker='D',palette=colors[1:] if -1 in set(c) else colors[:], hue=range(centroids.shape[0]), s=100,edgecolors='black')
        plt.show()


