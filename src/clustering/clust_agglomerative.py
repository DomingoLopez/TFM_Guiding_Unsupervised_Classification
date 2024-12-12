from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_model import ClusteringModel
from src.utils.decorators import deprecated


class AgglomerativeClusteringModel(ClusteringModel):
    """
    Agglomerative Clustering model class inheriting from ClusteringModel.

    This class implements the Agglomerative Clustering algorithm on a dataset 
    and provides methods to run clustering, calculate metrics, and save results 
    including plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the AgglomerativeClusteringModel.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="agglomerative")
  
    
    
    def run_optuna(self, evaluation_method="silhouette", n_trials=50,penalty: str="linear", penalty_range: tuple =(2,8)):
        """
        Run Optuna optimization for the Agglomerative Clustering model with a specified evaluation method.

        This method sets up and executes an Optuna hyperparameter optimization for the Agglomerative 
        Clustering algorithm. It defines the range of hyperparameters specific to Agglomerative Clustering, 
        including `n_clusters`, `linkage`, and `metric`. The optimization process seeks to maximize the 
        silhouette score or minimize the Davies-Bouldin score, depending on the selected `evaluation_method`.
        
        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. It can be either 'silhouette' (to maximize the silhouette score) 
            or 'davies_bouldin' (to minimize the Davies-Bouldin score). Defaults to "silhouette".
        n_trials : int, optional
            The number of optimization trials to run. Defaults to 50.

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing details of the optimization process, including the best 
            hyperparameters found and associated evaluation score.
        
        Notes
        -----
        - This method calls the generic `run_optuna_generic` method from the base class `ClusteringModel`, 
        which manages the Optuna optimization process and the model evaluation.
        - `model_builder` is a nested function that constructs an Agglomerative Clustering model using 
        hyperparameters suggested by each Optuna trial. Note that if `linkage` is set to 'ward', 
        the `metric` is automatically set to 'euclidean' as required by the Agglomerative Clustering algorithm.
        """
         # Param/model builder for Agglomerative
        def model_builder(trial):
            n_clusters = trial.suggest_int('n_clusters', 40, 200)
            linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])
            custom_metric = metric if linkage != "ward" else "euclidean"
            
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=custom_metric
            )

         # Call generic class method
        return self.run_optuna_generic(model_builder, evaluation_method, n_trials, penalty, penalty_range)
        
   
    def run_gridsearch(self, evaluation_method="silhouette"):
        """
        Run GridSearchCV for the Agglomerative Clustering model with a specified evaluation method.

        This method sets up a grid search for hyperparameter tuning for the Agglomerative Clustering 
        algorithm, using either the silhouette score or Davies-Bouldin score as the evaluation metric. 
        The search includes parameters specific to Agglomerative Clustering such as `n_clusters`, 
        `linkage`, and `metric`.

        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. Can be either 'silhouette' for maximizing 
            the silhouette score or 'davies_bouldin' for minimizing the Davies-Bouldin score.
            Defaults to 'silhouette'.
        
        Returns
        -------
        GridSearchCV
            The GridSearchCV object containing details of the best hyperparameters found and 
            the associated evaluation score.

        Notes
        -----
        - If the `linkage` parameter is set to 'ward', the `metric` is forced to 'euclidean', as 
        required by the Agglomerative Clustering algorithm.
        - Calls the `run_grid_search_generic` method from the base class `ClusteringModel` to 
        manage the grid search process.
        """
        # Define the parameter grid for Agglomerative Clustering
        param_grid = {
            'n_clusters': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            'linkage': ['ward', 'complete', 'average', 'single'],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }

        # Remove invalid metric options for 'ward' linkage (must use 'euclidean' for 'ward')
        valid_params = []
        for linkage in param_grid['linkage']:
            for metric in param_grid['metric']:
                if linkage == 'ward' and metric != 'euclidean':
                    continue
                valid_params.append({'linkage': linkage, 'metric': metric})

        # Create a new parameter grid that respects the 'ward'-'euclidean' requirement
        new_param_grid = {'n_clusters': param_grid['n_clusters'], 'linkage_metric': valid_params}

        # Call the generic grid search method
        return self.run_grid_search_generic(new_param_grid, evaluation_method)
   
   

if __name__ == "__main__":
    # Test the AgglomerativeClusteringModel class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    agglomerative_clustering = AgglomerativeClusteringModel(data)
    agglomerative_clustering.run()
    print("Agglomerative clustering complete. Results and plots saved.")
