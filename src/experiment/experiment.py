from collections import Counter
from itertools import product
import os
from pathlib import Path
import pickle
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.preprocess.preprocess import Preprocess


class Experiment():
    """
    Experiment Class where we can define which kind of methods, algorithms, 
    scalers and optimizers we should experiment with.
    
    This is the main class for setting up and running clustering experiments 
    using specified optimizers, dimensionality reduction methods, and evaluation metrics.
    """

    def __init__(self, 
                 id:int = 0,
                 dino_model = "small",
                 data:pd.DataFrame = None, 
                 optimizer:str = "optuna",
                 optuna_trials: int = 100,
                 normalization:bool = True,
                 dim_red:str = None, 
                 reduction_params:dict = None,
                 scaler:str = None, 
                 clustering:str = "hdbscan",
                 eval_method:str = "silhouette",
                 penalty = None,
                 penalty_range = None,
                 cache= True, 
                 verbose= False,
                 **kwargs):
    
        """
        Initializes an experiment with the specified configuration.

        Args:
            data (list): The data to be used for the experiment.
            optimizer (str): The optimization method to use, e.g., 'optuna' or 'gridsearch'.
            dim_red (str): Dim reduction
            reduction_parameters (dict): parameters of reduction
            scalers (list): List of scalers to normalize the data.
            clustering (str): Clustering algorithm to apply.
            eval_method (str): Evaluation metric for clustering quality.
            penalty (str): Penalty type to be applied in optimization.
            penalty_range (tuple): Range of penalty values.
            cache (bool): If True, caching is enabled.
            verbose (bool): If True, enables verbose logging.
            **kwargs: Additional keyword arguments.
        """
        # Setup attrs
        self._id = id
        self._dino_model = dino_model
        self._data = data
        self._optimizer = optimizer
        self._optuna_trials = optuna_trials
        self._normalization = normalization
        self._dim_red = dim_red
        self._reduction_params = reduction_params
        self._scaler = scaler
        self._clustering = clustering
        self._eval_method = eval_method
        self._penalty = penalty
        self._penalty_range = penalty_range
        self._cache = cache
        self._verbose = verbose
        self._results_df = None

        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        self._main_result_dir = (
            Path(__file__).resolve().parent
            # TODO: Add penalty to silhouette folder name
            / f"results/experiment_{self._id}/{self._eval_method}"
        )
        self._result_path_csv = os.path.join(self._main_result_dir, "result.csv")
        self._result_path_pkl = os.path.join(self._main_result_dir, "result.pkl")
        os.makedirs(self._main_result_dir, exist_ok=True)



    # Getters and Setters
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def dino_model(self):
        return self._dino_model

    @dino_model.setter
    def dino_model(self, value):
        self._dino_model = value
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def optuna_trials(self):
        return self._optuna_trials

    @optuna_trials.setter
    def optuna_trials(self, value):
        self._optuna_trials = value

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        self._normalization = value

    @property
    def dim_red(self):
        return self._dim_red

    @dim_red.setter
    def dim_red(self, value):
        self._dim_red = value

    @property
    def reduction_params(self):
        return self._reduction_params

    @reduction_params.setter
    def reduction_params(self, value):
        self._reduction_params = value

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def clustering(self):
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        self._clustering = value

    @property
    def eval_method(self):
        return self._eval_method

    @eval_method.setter
    def eval_method(self, value):
        self._eval_method = value

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        self._penalty = value

    @property
    def penalty_range(self):
        return self._penalty_range

    @penalty_range.setter
    def penalty_range(self, value):
        self._penalty_range = value

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def results_df(self):
        return self._results_df

    @results_df.setter
    def results_df(self, value):
        self._results_df = value
    
       

    def run_experiment(self):
        """
        Executes the experiment based on the chosen optimizer.
        
        Calls the appropriate internal method for running an experiment using 
        either Optuna or Grid Search as specified in the optimizer attribute.
        
        Raises:
            ValueError: If the optimizer specified is not supported.
        """
        logger.info(f"STARTING EXPERIMENT USING {self._optimizer.upper()} OPTIMIZER")
        if self._optimizer == "optuna":
            self.__run_experiment_optuna()
        elif self._optimizer == "gridsearch":
            self.__run_experiment_gridsearch()
        else:
            raise ValueError("optimizer not supported. Valid options are 'optuna' or 'gridsearch' ")
    



    def __run_experiment_optuna(self):
        """
        Runs the experiment using the Optuna optimizer, performs optimization, 
        and saves results to CSV and pickle.
        """
        if self.__load_cached_results():
            return

        results = []
        param_combinations = self.__get_param_combinations()

        for reduction_params in param_combinations:
            embeddings = self.__apply_preprocessing(reduction_params)
            clustering_model = ClusteringFactory.create_clustering_model(self._clustering, embeddings)
            study = clustering_model.run_optuna(
                evaluation_method=self._eval_method, n_trials=self._optuna_trials, penalty=self._penalty, penalty_range=self._penalty_range
            )
            best_trial = study.best_trial
            n_clusters_best = best_trial.user_attrs.get("n_clusters", None)
            centers_best = best_trial.user_attrs.get("centers", None)
            labels_best = best_trial.user_attrs.get("labels", None)
            label_counter = Counter(labels_best)
            score_best = best_trial.user_attrs.get("score_original", None)
            noise_not_noise = {
                -1: label_counter.get(-1, 0),
                1: sum(v for k, v in label_counter.items() if k != -1)
            }
            # Depending on eval_method type
            if self._eval_method == "silhouette":
                score_noise_ratio = score_best / (noise_not_noise.get(-1) + 1)
            elif self._eval_method == "davies_bouldin":
                score_noise_ratio = (noise_not_noise.get(-1) + 1) / score_best
            elif self._eval_method == "silhouette_noise":
                score_noise_ratio = score_best
            elif self._eval_method == "davies_noise":
                score_noise_ratio = score_best
            else:
                raise ValueError(f"Unsupported evaluation method: {self._eval_method}")
            

            # Append results
            results.append({
                "id": self._id,
                "clustering": self._clustering,
                "eval_method": self._eval_method,
                "optimization": self._optimizer,
                "optuna_trials": self._optuna_trials,
                "normalization": self._normalization,
                "scaler": self._scaler,
                "dim_red": self._dim_red,
                "reduction_params": reduction_params,
                "dimensions": reduction_params.get("n_components", None),  
                "embeddings": embeddings,
                "n_clusters": n_clusters_best,
                "best_params": str(study.best_params),
                "centers": centers_best,
                "labels": labels_best,
                "label_counter": label_counter,
                "noise_not_noise": noise_not_noise,
                "score_noise_ratio": score_noise_ratio,
                "penalty": self._penalty,
                "penalty_range": self._penalty_range if self._penalty is not None else None,
                "score_w_penalty": study.best_value,
                "score_w/o_penalty": score_best
            })

        self.store_results(results)



    def __load_cached_results(self):
        """
        Checks if results are already cached; if yes, loads and returns them.
        """
        if os.path.isfile(self._result_path_pkl) and self._cache:
            try:
                results_df = pickle.load(open(str(self._result_path_pkl), "rb"))
                results_df.to_csv(self._result_path_csv, sep=";")
                self._results_df = results_df
                return True
            except FileNotFoundError:
                logger.error("Cached results file not found.")
        return False


    def __get_param_combinations(self):
        """
        Generates parameter combinations based on dimensionality reduction and reduction parameters.
        """
        if self._dim_red and self._reduction_params:
            param_names = list(self._reduction_params.keys())
            param_values = list(self._reduction_params.values())
            return [dict(zip(param_names, combination)) for combination in product(*param_values)]
        return [{}]


    def __apply_preprocessing(self, reduction_params):
        """
        Applies preprocessing steps including normalization, scaling, and dimensionality reduction.
        """
        preprocces_obj = Preprocess(embeddings=self._data, 
                                    dino_model = self._dino_model,
                                    scaler=self._scaler, 
                                    normalization=self._normalization,
                                    dim_red=self._dim_red,
                                    reduction_params=reduction_params)
        return preprocces_obj.run_preprocess()



    def store_results(self, results):
        """
        Stores results in CSV and pickle formats.
        """
        logger.info(f"ENDING EXPERIMENT...STORING RESULTS.")
        results_df = pd.DataFrame(results)
        results_df.to_csv(self._result_path_csv, sep=";")
        self._results_df = results_df
        pickle.dump(results_df, open(str(self._result_path_pkl), "wb"))
        logger.info(f"EXPERIMENT ENDED.")


    

    def __run_experiment_gridsearch(self):
        """
        Runs the experiment using Grid Search.

        If cache is enabled and results exist, it loads them from a pickle file.
        Otherwise, it performs the grid search for each parameter combination in 
        dimensionality reduction and stores only the best result from each grid search.
        """
        if self.__load_cached_results():
            return

        results = []
        param_combinations = self.__get_param_combinations()
        
        for reduction_params in param_combinations:
            embeddings = self.__apply_preprocessing(reduction_params)
            clustering_model = ClusteringFactory.create_clustering_model(self._clustering, embeddings)
            grid_search = clustering_model.run_gridsearch(evaluation_method=self._eval_method)
            
            # Best result for the current grid search
            best_index = grid_search.best_index_
            best_params = grid_search.cv_results_['params'][best_index]
            best_score_curr = grid_search.cv_results_['mean_test_score'][best_index]
            # Fit the best estimator to get labels and centers
            best_estimator = grid_search.best_estimator_.set_params(**best_params).fit(embeddings)
            labels = getattr(best_estimator, 'labels_', None)
            if "n_clusters" in best_params:
                n_clusters = best_params["n_clusters"]
            else:
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise for density-based clustering
            centers = clustering_model.get_cluster_centers(labels)
            # Calculate additional metrics
            label_counter = Counter(labels)
            noise_not_noise = {
                -1: label_counter.get(-1, 0),
                1: sum(v for k, v in label_counter.items() if k != -1)
            }
            
            # Depending on eval_method type
            if self._eval_method == "silhouette":
                score_noise_ratio = best_score_curr / (noise_not_noise.get(-1) + 1)
            elif self._eval_method == "davies_bouldin":
                score_noise_ratio = (noise_not_noise.get(-1) + 1) / best_score_curr
            else:
                raise ValueError(f"Unsupported evaluation method: {self._eval_method}")
            
            results.append({
            "id": self._id,
            "clustering": self._clustering,
            "eval_method": self._eval_method,
            "optimization": self._optimizer,
            "optuna_trials": self._optuna_trials,
            "normalization": self._normalization,
            "scaler": self._scaler,
            "dim_red": self._dim_red,
            "reduction_params": reduction_params,
            "dimensions": reduction_params.get("n_components", None),  
            "embeddings": embeddings,
            "n_clusters": n_clusters,
            "best_params": str(best_params),
            "centers": centers,
            "labels": labels,
            "label_counter": label_counter,
            "noise_not_noise": noise_not_noise,
            "score_noise_ratio": score_noise_ratio,
            "penalty": self._penalty,
            "penalty_range": self._penalty_range if self._penalty is not None else None,
            "score_w_penalty": None,
            "score_w/o_penalty": best_score_curr
            })
            
            


        self.store_results(results)



if __name__ == "__main__":
    pass
