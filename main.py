import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from loguru import logger

from src.clustering.clust_hdbscan import HDBSCANClustering
from src.clustering.clustering_factory import ClusteringFactory
from src.experiment.experiment import Experiment
from src.experiment.experiment_result_controller import ExperimentResultController
from src.llava_inference.llava_inference import LlavaInference
from src.multimodal_clustering_metric.multimodal_clustering_metric import MultiModalClusteringMetric
from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference
from src.preprocess.preprocess import Preprocess

import matplotlib.pyplot as plt
import cv2


def load_images(path) -> list:
    # Finding images
    # image_loader = ImageLoader(folder="./data/Small_Data")
    image_loader = ImageLoader(folder=path)
    images = image_loader.find_images()
    return images

def generate_embeddings(images, model) -> list:
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name=model, images=images, disable_cache=False)
    embeddings = dinomodel.run()
    return embeddings


def run_experiments(file, images) -> None:
   
    # Load json file with all experiments
    with open(file, 'r') as f:
        experiments_config = json.load(f)

    for config in experiments_config:
        id = config.get("id")
        dino_model = config.get("dino_model","small")
        optimizer = config.get("optimizer", "optuna")
        optuna_trials = config.get("optuna_trials", None)
        normalization = config.get("normalization", True)
        scaler = config.get("scaler", None)
        dim_red = config.get("dim_red", None)
        reduction_parameters = config.get("reduction_parameters", None)
        clustering = config.get("clustering", "hdbscan")
        eval_method = config.get("eval_method", "silhouette")
        penalty = config.get("penalty", None)
        penalty_range = config.get("penalty_range", None)
        cache = config.get("cache", True)
        # Make and Run Experiment
        logger.info(f"LOADING EXPERIMENT: {id}")

        # Generate embeddings based on experiment model
        embeddings = generate_embeddings(images, model=dino_model)
        experiment = Experiment(
            id,
            dino_model,
            embeddings,
            optimizer,
            optuna_trials,
            normalization,
            dim_red,
            reduction_parameters,
            scaler,
            clustering,
            eval_method,
            penalty,
            penalty_range,
            cache
        )
        experiment.run_experiment()


if __name__ == "__main__": 
    
    # ###################################################################
    images = load_images("./data/Data")
    experiments_file = "src/experiment/json/experiments_optuna_all.json"
    run_experiments(experiments_file, images)
    # Classification level to analyze
    classification_lvl = [3]
    prompts = [1,2]
    llava_models = ("llava1-6_7b", "llava1-6_13b", "llava1-5_7b")
    # Cluster Range to filter
    n_cluster_range = (40,300)


    # Obtain experiments results
    with open(experiments_file, 'r') as f:
        experiments_config = json.load(f)

    result_list = []
    result_list_top_trials = []
    for config in experiments_config:
        eval_method = config.get("eval_method", "silhouette")
        id = config.get("id",1)
        dino_model = config.get("dino_model")
        dim_red = config.get("dim_red","umap")

        # APPLY FILTERS FROM REDUCTION HIPERPARAMS
        if dim_red == "umap":
            reduction_params = {
                "n_components": (2,25),
                "n_neighbors": (3,60),
                "min_dist": (0.1, 0.8)
            }
        elif dim_red == "tsne":
            reduction_params = {
                "n_components": (2,25),
                "perplexity": (4,60),
                "early_exaggeration": (7, 16)
            }
        else:
            reduction_params = {
                "n_components": (2,25)
            }

        experiment_controller = ExperimentResultController(eval_method, 
                                                           dino_model,
                                                           experiment_id=id, 
                                                           n_cluster_range=n_cluster_range,
                                                           reduction_params=reduction_params)
        experiments_filtered = experiment_controller.get_top_k_experiments(top_k=5)
        best_experiment = experiment_controller.get_best_experiment_data(experiments_filtered)
        experiment_controller.create_cluster_dirs(images=images, experiment=best_experiment)
        experiment_controller.plot_all(best_experiment)
        

        for class_lvl in classification_lvl:
            for model in llava_models:
                for prompt in prompts:
                    llava = LlavaInference(images=images, classification_lvl=class_lvl, n_prompt=prompt, model=model)
                    llava.run()
                    # Get Llava Results from llava-model i 
                    llava_results_df = llava.get_results(model)
                    # Obtain categories from classification_lvl
                    categories = llava.get_categories(class_lvl)
                    best_experiment_index = best_experiment["original_index"]

                    # for every experiment in top 5, I want to store that, so I would get best 5 results from every experiment.
                    # Also keep storing best experiment results
                    for idx, row in experiments_filtered.iterrows():
                        img_cluster_dict = experiment_controller.get_cluster_images_dict(images,row,None,False)
                        # Quality metrics
                        lvm_lvlm_metric = MultiModalClusteringMetric(class_lvl,
                                                                    categories,
                                                                    model, 
                                                                    prompt, 
                                                                    row, 
                                                                    img_cluster_dict, 
                                                                    llava_results_df)
                        lvm_lvlm_metric.generate_stats()
                        # Obtain results
                        quality_results = pd.DataFrame()
                        for i in (True, False):
                            # Calculate metrics
                            results = lvm_lvlm_metric.calculate_clustering_quality(use_noise=i)
                            # Join results (in columns)
                            quality_results = pd.concat([quality_results, pd.DataFrame([results])], axis=1)


                        
                        # Save results in list
                        result_list_top_trials.append({
                            "experiment_id" : id,
                            "trial_index": row["original_index"],
                            "dino_model" : dino_model,
                            "normalization" : row["normalization"],
                            "scaler" : row["scaler"],
                            "dim_red" : row["dim_red"],
                            "reduction_parameters" : row["reduction_params"],
                            "clustering" : row["clustering"],
                            "n_clusters": row["n_clusters"],
                            "best_params": row["best_params"],
                            "penalty" : row["penalty"],
                            "penalty_range" : row["penalty_range"],
                            "noise_not_noise" : row["noise_not_noise"],
                            # Important things
                            "classification_lvl": class_lvl,
                            "lvlm": model,
                            "prompt": prompt,
                            "eval_method": eval_method,
                            "best_score": row["score_w_penalty"] if "noise" in row["eval_method"] else row["score_w/o_penalty"], 
                            # Metrics
                            "homogeneity_global": quality_results["homogeneity_global"].iloc[0],
                            "entropy_global": quality_results["entropy_global"].iloc[0],
                            "quality_metric":quality_results["quality_metric"].iloc[0],
                            "homogeneity_global_w_noise": quality_results["homogeneity_global_w_noise"].iloc[0],
                            "entropy_global_w_noise": quality_results["entropy_global_w_noise"].iloc[0],
                            "quality_metric_w_noise":quality_results["quality_metric_w_noise"].iloc[0]
                        })


                        # Store best results only
                        if row["original_index"] == best_experiment_index:
                            # Save results in list
                            result_list.append({
                                "experiment_id" : id,
                                "best_trial_index": best_experiment["original_index"],
                                "dino_model" : dino_model,
                                "normalization" : best_experiment["normalization"],
                                "scaler" : best_experiment["scaler"],
                                "dim_red" : best_experiment["dim_red"],
                                "reduction_parameters" : best_experiment["reduction_params"],
                                "clustering" : best_experiment["clustering"],
                                "n_clusters": best_experiment["n_clusters"],
                                "best_params": best_experiment["best_params"],
                                "penalty" : best_experiment["penalty"],
                                "penalty_range" : best_experiment["penalty_range"],
                                "noise_not_noise" : best_experiment["noise_not_noise"],
                                # Important things
                                "classification_lvl": class_lvl,
                                "lvlm": model,
                                "prompt": prompt,
                                "eval_method": eval_method,
                                "best_score": best_experiment["score_w_penalty"] if "noise" in best_experiment["eval_method"] else best_experiment["score_w/o_penalty"], 
                                # Metrics
                                "homogeneity_global": quality_results["homogeneity_global"].iloc[0],
                                "entropy_global": quality_results["entropy_global"].iloc[0],
                                "quality_metric":quality_results["quality_metric"].iloc[0],
                                "homogeneity_global_w_noise": quality_results["homogeneity_global_w_noise"].iloc[0],
                                "entropy_global_w_noise": quality_results["entropy_global_w_noise"].iloc[0],
                                "quality_metric_w_noise":quality_results["quality_metric_w_noise"].iloc[0]
                            })
                            lvm_lvlm_metric.plot_cluster_categories_3()


    df_results = pd.DataFrame(result_list)
    df_results.to_csv("results.csv",sep=";")

    df_results_top_k = pd.DataFrame(result_list_top_trials)
    df_results_top_k.to_csv("results_top_trials.csv",sep=";")
