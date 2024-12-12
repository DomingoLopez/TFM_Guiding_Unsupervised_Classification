from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_model import ClusteringModel
import optuna
import pandas as pd


class DBSCANClustering(ClusteringModel):
    """
    DBSCAN clustering model class inheriting from ClusteringModel.

    This class implements the DBSCAN clustering algorithm on a dataset and 
    provides methods to run clustering, calculate metrics, and save results 
    including plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the DBSCANClustering model.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="dbscan")

    def run_optuna(self, evaluation_method="silhouette", n_trials=50, penalty: str = "linear", penalty_range: tuple = (2, 8)):
        """
        Run Optuna optimization for the DBSCAN clustering model with a specified evaluation method.

        This method sets up and executes an Optuna hyperparameter optimization for the DBSCAN 
        clustering algorithm. It defines the range of hyperparameters specific to DBSCAN, 
        including `eps` and `min_samples`, and passes these parameters to the generic Optuna 
        optimization method inherited from the base class.
        
        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. It can be either 'silhouette' (for maximizing 
            the silhouette score) or 'davies_bouldin' (for minimizing the Davies-Bouldin score). 
            Defaults to "silhouette".
        n_trials : int, optional
            The number of optimization trials to run. Defaults to 50.

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing details of the optimization process, including 
            the best hyperparameters found and associated evaluation score.
        """
        def model_builder(trial):
            return DBSCAN(
                eps=trial.suggest_float("eps", 0.5, 2, log=True),
                min_samples=trial.suggest_int("min_samples", 2, 20),
                metric=trial.suggest_categorical("metric", ["cosine","euclidean", "manhattan"])
            )
        
        return self.run_optuna_generic(model_builder, evaluation_method, n_trials, penalty, penalty_range)

    def run_gridsearch(self, evaluation_method="silhouette"):
        """
        Run GridSearchCV for the DBSCAN clustering model with a specified evaluation method.

        This method defines the hyperparameter grid specific to DBSCAN clustering and 
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
        param_grid = {
            'eps': [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0],
            'min_samples': [2, 5, 7, 10, 15],
            'metric': ['euclidean', 'manhattan', 'chebyshev']
        }
        
        return self.run_grid_search_generic(param_grid, evaluation_method)


if __name__ == "__main__":
    # Test the DBSCANClustering class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    dbscan_clustering = DBSCANClustering(data)
    dbscan_clustering.run()
    print("DBSCAN clustering complete. Results and plots saved.")
