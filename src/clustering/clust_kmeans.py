import os
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_model import ClusteringModel
from src.utils.decorators import deprecated


class KMeansClustering(ClusteringModel):
    """
    KMeans clustering model class inheriting from ClusteringModel.

    This class implements the KMeans clustering algorithm on a dataset and 
    provides methods to run clustering, calculate metrics, and save results 
    including plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the KMeansClustering model.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="kmeans")
        
        
    def run_optuna(self, evaluation_method: str = "silhouette", n_trials: int = 50,penalty: str="linear", penalty_range: tuple =(2,8)):
        """
        Run Optuna optimization for the KMeans clustering model with a specified evaluation method.

        This method performs an Optuna hyperparameter optimization to tune the KMeans clustering 
        algorithm. It defines a model builder function to set the hyperparameter ranges specific to 
        KMeans, including the number of clusters (`n_clusters`), initialization method (`init`), 
        number of centroid seeds (`n_init`), and maximum number of iterations (`max_iter`). 
        The optimization goal is to maximize the silhouette score or minimize the Davies-Bouldin score 
        based on the specified `evaluation_method`.

        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. Can be either 'silhouette' for maximizing the 
            silhouette score, or 'davies_bouldin' for minimizing the Davies-Bouldin score. 
            Defaults to 'silhouette'.
        n_trials : int, optional
            The number of trials to run in Optuna optimization. Defaults to 50.

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing details of the optimization process, including 
            the best hyperparameters found and the associated evaluation score.

        Notes
        -----
        - The method defines a `model_builder` function that constructs a KMeans model with 
        hyperparameters suggested by each Optuna trial.
        - Calls the `run_optuna_generic` method from the base class `ClusteringModel`, which 
        manages the Optuna optimization process and progress tracking.
        - The optimization direction is set based on the specified evaluation metric.

        Example
        -------
        >>> kmeans_clustering = KMeansClustering(data)
        >>> study = kmeans_clustering.run_optuna(evaluation_method="davies_bouldin", n_trials=100)
        >>> print("Best parameters:", study.best_params)
        >>> print("Best score:", study.best_value)
        """
        
        # Define the model builder function for KMeans
        def model_builder(trial):
            return KMeans(
                n_clusters=trial.suggest_int('n_clusters', 40, 180),
                init=trial.suggest_categorical('init', ['k-means++', 'random']),
                n_init=trial.suggest_int('n_init', 10, 50),
                max_iter=trial.suggest_int('max_iter', 100, 150)
            )

        # Call the generic Optuna optimization method
        return self.run_optuna_generic(model_builder, evaluation_method, n_trials,penalty, penalty_range)
        



    def run_gridsearch(self, evaluation_method="silhouette"):
        """
        Run GridSearchCV for the KMeans clustering model with a specified evaluation method.

        This method defines the hyperparameter grid specific to KMeans clustering and 
        calls the generic run_grid_search_generic method to perform the grid search.

        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. Can be either 'silhouette' for maximizing 
            the silhouette score, or 'davies_bouldin' for minimizing the Davies-Bouldin score.
            Defaults to 'silhouette'.
        
        Returns
        -------
        GridSearchCV
            The GridSearchCV object containing details of the best hyperparameters found and 
            the associated evaluation score.
        """
        # Define the parameter grid for KMeans
        param_grid = {
            'n_clusters': [10,15,17,19,23,26,28,30,35,37,40,45,50,55],
            'init': ['k-means++', 'random'],
            'n_init': [10, 20],
            'max_iter': [100, 300]
        }
    
        
        # Call the generic grid search method
        return self.run_grid_search_generic(param_grid, evaluation_method)


if __name__ == "__main__":
    # Test the KMeansClustering class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    kmeans_clustering = KMeansClustering(data)
    kmeans_clustering.run()
    print("KMeans clustering complete. Results and plots saved.")
