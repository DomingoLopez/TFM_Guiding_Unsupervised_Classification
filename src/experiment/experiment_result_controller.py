from collections import Counter
from itertools import product
import os
from pathlib import Path
import pickle
import shutil
import sys
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Optional
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import umap
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.preprocess.preprocess import Preprocess


class ExperimentResultController():

    def __init__(self, 
                 eval_method="silhouette",
                 dino_model="small",
                 experiment_id=None,
                 n_cluster_range=None,
                 reduction_params=None,
                 cache= True, 
                 verbose= False,
                 **kwargs):
    
        # Setup attrs
        self.eval_method = eval_method
        self.dino_model = dino_model
        self.experiment_id = experiment_id
        self.n_cluster_range = n_cluster_range
        self.reduction_params = reduction_params
        self.cache = cache
        self.verbose = verbose
        

        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Results dir
        self.results_dir = (
            Path(__file__).resolve().parent
            / f"results"
        )

        # Clusters dir
        self.cluster_dir = (
            Path(__file__).resolve().parent
            / f"clusters"
        )
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

        # Load all experiments for given eval_method
        self.results_df = None
        self.cluster_images_dict = None
        self.__load_all_experiments(experiment_id)

        # Get original embeddings (Just for representation)
        # This is a bit messy. Refactor asap.
        if self.dino_model == "small":
            embeddings_name = "embeddings_dinov2_vits14_5066.pkl"
        else:
            embeddings_name = "embeddings_dinov2_vitb14_5066.pkl"
        original_embeddings_path = Path(__file__).resolve().parent.parent / f"dinov2_inference/cache/{embeddings_name}"
        with open(original_embeddings_path, "rb") as f:
            self.original_embeddings = pickle.load(f)
        

    
    def __load_all_experiments(self, experiment_id = None):
        """
        Loads all experiment of given experiment id.
        """
        experiment_files = Path(self.results_dir).rglob("*.pkl") if experiment_id is None else Path(os.path.join(self.results_dir, f"experiment_{experiment_id}")).rglob("*.pkl")
        experiments = []
        for file in experiment_files:
            try:
                with open(file, "rb") as f:
                    result = pickle.load(f)
                    
                # Check if the loaded result is a valid DataFrame
                if isinstance(result, pd.DataFrame) and not result.empty:
                    result = result.reset_index().rename(columns={"index": "original_index"})
                    experiments.append(result)
                else:
                    logger.warning(f"Invalid or empty result file: {file}")
            
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")

        # Combine all valid results
        if experiments:
            self.results_df = pd.concat(experiments, ignore_index=True)
        else:
            self.results_df = pd.DataFrame()
            logger.warning("No experiments found with the specified eval_method.")



    def get_top_k_experiments(self, top_k: int) -> pd.DataFrame:
        """
        Returns the top_k experiments based on the specified criteria.

        Parameters:
            top_k (int): Number of top experiments to return.

        Returns:
            pd.DataFrame: Filtered DataFrame with the top_k experiments.
        """
        
        # Validate n_cluster_range
        min_n_cluster, max_n_cluster = self.n_cluster_range
        if min_n_cluster < 2 or max_n_cluster > 800:
            raise ValueError("n_cluster_range values must be between 2 and 800.")
        if min_n_cluster > max_n_cluster:
            raise ValueError("min_n_cluster cannot be greater than max_n_cluster.")
        
        # Validate reduction_params
        for key, value_range in self.reduction_params.items():
            if not isinstance(value_range, tuple) or len(value_range) != 2:
                raise ValueError(f"Parameter {key} in reduction_params must be a tuple (min, max).")
            if value_range[0] > value_range[1]:
                raise ValueError(f"Invalid range for {key}: {value_range}. Min cannot be greater than Max.")
    
        
        # Verify df is loaded
        if self.results_df.shape[0] == 0 or self.results_df is None:
            logger.warning("No experiments loaded. Returning an empty DataFrame.")
            return pd.DataFrame()


        # Determine sorting column and order based on eval_method
        if self.eval_method == "davies_bouldin":
            sort_column =  'score_w/o_penalty'
            ascending_order = True  # Lower is better for davies_bouldin
        elif self.eval_method == "davies_noise":
            sort_column = 'score_w_penalty'
            ascending_order = True  # Lower is better for davies_bouldin
        elif self.eval_method == "silhouette":
            sort_column =  'score_w/o_penalty'
            ascending_order = False  # Higher is better for silhouette
        elif self.eval_method == "silhouette_noise":
            sort_column =  'score_w_penalty'
            ascending_order = False  # Higher is better for silhouette
        else:
            raise ValueError("Eval method not supported")


        # Filter dataframe based on cluster
        filtered_df = self.results_df[
            (self.results_df['n_clusters'] >= min_n_cluster) & 
            (self.results_df['n_clusters'] <= max_n_cluster) 
        ]

        # Check if df empty and filter based on reduction params
        if not filtered_df.empty:
            # Filter by reduction params
            for param, value_range in self.reduction_params.items():
                min_val, max_val = value_range
                filtered_df = filtered_df[
                    filtered_df['reduction_params'].apply(
                        lambda params: param in params and min_val <= params[param] <= max_val
                    )
                ]
        else:
             logger.warning("Column 'reduction_params' not found or DataFrame is empty. Skipping reduction parameter filtering.")


        if not filtered_df.empty:
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending_order)
        else:
            logger.warning("Filtered DataFrame is empty. Skipping sorting step.")

        if filtered_df.empty:
            logger.warning("Filtered DataFrame is empty after applying filters. Returning top_k experiments from the entire dataset.")
            filtered_df = self.results_df.sort_values(by=sort_column, ascending=ascending_order)

        # Select the top_k experiments
        top_k_df = filtered_df.head(top_k)
        
        return top_k_df


    def get_best_experiment_data(self, filtered_df):
        """
        Helper function to get the best experiment based on `experiment_type`.

        Parameters
        ----------
        experiment_type : str
            "best" for best silhouette or "silhouette_noise_ratio" for best silhouette-to-noise ratio.

        Returns
        -------
        pd.Series
            The row in the DataFrame corresponding to the best experiment.
        """
        # First of all, filter those with cluster number less than 10 for example
        # This should be an input parameter 
        
        if filtered_df.empty:
            raise ValueError("No experiments found.")


        # Determine sorting column and order based on eval_method
        if self.eval_method == "davies_bouldin":
            df = filtered_df.loc[filtered_df["score_w/o_penalty"].idxmin()]
        elif self.eval_method == "davies_noise":
            df = filtered_df.loc[filtered_df["score_w_penalty"].idxmin()]
        elif self.eval_method == "silhouette":
            df = filtered_df.loc[filtered_df["score_w/o_penalty"].idxmax()]
        elif self.eval_method == "silhouette_noise":
            df = filtered_df.loc[filtered_df["score_w_penalty"].idxmax()]
        else:
            raise ValueError("Eval method not supported")

        logger.info(f"Selected experiment with score: {df['score_w/o_penalty']:.3f}")
            
        return df





    def get_cluster_images_dict(self, images, experiment, knn=None, save_result=True):
        """
        Finds the k-nearest neighbors for each centroid of clusters among points that belong to the same cluster.
        Returns knn points for each cluster in dict format in case knn is not None

        Parameters
        ----------
        knn : int
            Number of nearest neighbors to find for each centroid

        Returns
        -------
        sorted_cluster_images_dict : dictionary with images per cluster (as key)
        """

        cluster_images_dict = {}
        labels = experiment['labels']

        if knn is not None:
            used_metric = "euclidean"
            
            for idx, centroid in enumerate(tqdm(experiment['centers'], desc="Processing cluster dirs (knn images selected)")):
                # Filter points based on label mask over embeddings
                cluster_points = experiment['embeddings'].values[labels == idx]
                cluster_images = [images[i] for i in range(len(images)) if labels[i] == idx]
                # Adjust neighbors, just in case
                n_neighbors_cluster = min(knn, len(cluster_points))
                
                nbrs = NearestNeighbors(n_neighbors=n_neighbors_cluster, metric=used_metric, algorithm='auto').fit(cluster_points)
                distances, indices = nbrs.kneighbors([centroid])
                closest_indices = indices.flatten()
                
                # Get images for each cluster
                cluster_images_dict[idx] = [cluster_images[i] for i in closest_indices]

            # Get noise (-1)
            cluster_images_dict[-1] = [images[i] for i in range(len(images)) if labels[i] == -1]
            
        else:
            for i, label in enumerate(tqdm(labels, desc="Processing cluster dirs")):
                if label not in cluster_images_dict:
                    cluster_images_dict[label] = []
                cluster_images_dict[label].append(images[i])
        
        # Sort dictionary
        if save_result:
            self.cluster_images_dict = dict(sorted(cluster_images_dict.items()))
        return self.cluster_images_dict




    def get_cluster_exp_path(self, experiment):
        return os.path.join(self.cluster_dir, f"experiment_{experiment['id']}/index_{experiment['original_index']}_{self.eval_method}_{experiment['score_w/o_penalty']:.3f}")






    def create_cluster_dirs(self, images, experiment, knn=None):
        """
        Create a dir for every cluster given in dictionary of images. 
        This is how we are gonna send that folder to ugr gpus
        """
        # logger.info("Copying images from Data path to cluster dirs")
        # For every key (cluster index)
        images_dict_format = self.get_cluster_images_dict(images, experiment)
        path_cluster = os.path.join(self.get_cluster_exp_path(experiment), "clusters")
        try:
            for k,v in images_dict_format.items():
                # Create folder if it doesnt exists
                cluster_dir = os.path.join(path_cluster, str(k)) 
                os.makedirs(cluster_dir, exist_ok=True)
                # For every path image, copy that image from its path to cluster folder
                for path in v:
                    shutil.copy(path, cluster_dir)
        except (os.error) as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)









    # ###############################################################################3
    # VISUALIZATION METHODS
    # ###############################################################################3



    def plot_all(self, experiment):
        
        if "silhouette" in self.eval_method:
            self.show_best_silhouette(experiment)
            self.show_best_scatter(experiment)
            self.show_best_scatter(experiment, keep_original_embeddings = False)
            self.show_best_scatter_with_centers(experiment)
            self.show_best_clusters_counters_comparision(experiment)
            #self.show_best_experiments_silhouette(experiment)
        elif "davies" in self.eval_method:
            self.show_best_scatter(experiment)
            self.show_best_scatter(experiment, keep_original_embeddings = False)
            self.show_best_scatter_with_centers(experiment)
            self.show_best_clusters_counters_comparision(experiment)
        else:
            raise ValueError("Eval Method not support for plotting")





    def show_best_silhouette(self, experiment):
        """
        Displays the top `top_n` clusters with the highest silhouette average and the 
        `top_n` clusters with the lowest silhouette average, only if the total cluster 
        count exceeds `min_clusters`. If there are `min_clusters` or fewer clusters, 
        it displays all clusters without filtering.
        """
        # Extract information from the experiment
        best_experiment = experiment
        best_id = best_experiment['id']
        best_labels = best_experiment['labels']
        clustering = best_experiment['clustering']
        dim_red = best_experiment['dim_red']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        optimizer = best_experiment['optimization']
        original_score = best_experiment['score_w/o_penalty']
        embeddings_used = best_experiment['embeddings']


        min_clusters = self.n_cluster_range[0]
        top_n = int(self.n_cluster_range[0]/2)


        # Exclude noise points (label -1)
        non_noise_mask = best_labels != -1
        non_noise_labels = best_labels[non_noise_mask]
        non_noise_data = embeddings_used[non_noise_mask]

        # Calculate silhouette values for non-noise data
        silhouette_values = silhouette_samples(non_noise_data, non_noise_labels)

        # Calculate average silhouette per cluster
        unique_labels = np.unique(non_noise_labels)
        cluster_count = len(unique_labels)

        # Determine top and bottom clusters
        top_clusters = []
        bottom_clusters = []
        if cluster_count <= min_clusters:
            selected_clusters = unique_labels
        else:
            cluster_silhouette_means = {
                label: silhouette_values[non_noise_labels == label].mean() for label in unique_labels
            }
            top_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get, reverse=True)[:top_n]
            bottom_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get)[:top_n]
            selected_clusters = sorted(set(top_clusters + bottom_clusters), key=lambda label: cluster_silhouette_means[label])

        # Setup the figure with GridSpec
        fig = plt.figure(figsize=(10, 12))
        spec = gridspec.GridSpec(2, 1, height_ratios=[7, 3])  # 70% for plot, 30% for legend

        # Generate a unique color palette for the selected clusters
        colors = sns.color_palette("tab20", len(selected_clusters))
        cluster_color_map = {label: colors[i] for i, label in enumerate(selected_clusters)}

        # Create the plot area (top 70%)
        ax_plot = fig.add_subplot(spec[0])
        y_lower = 10
        yticks = []  # To store Y-axis positions for cluster labels
        for i, label in enumerate(selected_clusters):
            ith_cluster_silhouette_values = silhouette_values[non_noise_labels == label]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Fill the silhouette for each cluster with a unique color
            ax_plot.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=cluster_color_map[label],
                alpha=0.7,
                label=f"Cluster {label}"
            )
            yticks.append(y_lower + 0.5 * size_cluster_i)  # Position for the cluster label on the Y-axis
            y_lower = y_upper + 10

            if i == len(top_clusters) - 1:
                ax_plot.axhline(y=y_lower, color='black', linestyle='--', linewidth=1.5)

        # Add a vertical line for the original silhouette score
        ax_plot.axvline(x=original_score, color="red", linestyle="--", label=f"Original Score: {original_score:.3f}")
        ax_plot.set_xlabel("Silhouette Coefficient", fontsize=16)
        ax_plot.set_ylabel("Cluster Index", fontsize=16)
        ax_plot.set_title(f"Silhouette Plot for Exp. {best_id} - {optimizer}\n"
                        f"Clustering: {clustering} | Dim Reduction: {dim_red} | Dimensions: {dimensions}\n"
                        f"Silhouette: {original_score:.3f}", fontsize=18)
        ax_plot.set_yticks(yticks)  # Set Y-axis ticks
        ax_plot.set_yticklabels(selected_clusters)  # Label the Y-axis ticks with cluster indices

        # Create the legend area (bottom 30%)
        ax_legend = fig.add_subplot(spec[1])
        ax_legend.axis("off")  # Hide the axes for the legend area
        handles, labels = ax_plot.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
        ax_legend.legend(
            handles[:len(selected_clusters)],  # Only include handles for the selected clusters
            labels[:len(selected_clusters)],  # Corresponding labels
            loc="center",
            title="Most Representative Clusters Top/Bottom Clusters",
            fontsize='small',
            title_fontsize='small',
            ncol=4
        )

        # Save and optionally show the plot
        file_suffix = "best_silhouette"
        file_path = os.path.join(self.get_cluster_exp_path(experiment),f"{file_suffix}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, bbox_inches="tight")

        logger.info(f"Silhouette plot saved to {file_path}.")







    def show_best_scatter(self, experiment, keep_original_embeddings=True):
        """
        Plots a 2D scatter plot for the best experiment configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """

        best_experiment = experiment
        best_id = best_experiment['id']
        best_index = best_experiment['original_index']
        best_labels = np.array(best_experiment['labels'])
        optimizer = best_experiment['optimization']
        clustering = best_experiment['clustering']
        eval_method = best_experiment['eval_method']
        scaler = best_experiment['scaler']
        dim_red = best_experiment['dim_red']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        embeddings = best_experiment['embeddings']
        score = best_experiment['score_w/o_penalty'] if eval_method in ("silhouette","davies_bouldin") else best_experiment['score_w_penalty']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count

        # Get original embeddings (avoind reduction over reduction embeddings)
        if keep_original_embeddings:
            data_df = pd.DataFrame(self.original_embeddings)
            data = data_df.values
        else:
            data = embeddings.values

        # Check if reduction is needed
        if data.shape[1] > 2:
            # If shape > 1, we cannot use selected reduction params, cause it doesnt make sense
            if dim_red == "umap":
                reducer = umap.UMAP(random_state=42, n_components=2, min_dist=0.2, n_neighbors=15)
                reduced_data = reducer.fit_transform(data)
            elif dim_red == "tsne":
                reducer = TSNE(random_state=42, n_components=2)
                reduced_data = reducer.fit_transform(data)
            else:
                pca = PCA(n_components=2, random_state=42)
                reduced_data = pca.fit_transform(data)
        else:
            # Use the data directly if already 2D
            reduced_data = data

        # Define colormap for clusters and manually assign red for noise
        colors = sns.color_palette("viridis", cluster_count)
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(12, 9))
        
        # Plot noise points (label -1) in red
        noise_points = reduced_data[best_labels == -1]
        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=10, alpha=0.6, label="Noise (-1)")
        
        # Plot cluster points
        cluster_points = reduced_data[best_labels != -1]
        cluster_labels = best_labels[best_labels != -1]
        scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_labels, cmap=cmap, s=10, alpha=0.6)

        # Add colorbar if useful to distinguish clusters
        plt.colorbar(scatter, spacing="proportional", ticks=np.linspace(0, cluster_count, num=10))
        
        plt.title(f"Scatter Plot for Exp. {best_id} - {optimizer} (Noise in Red, Clusters in 2D) \n\n"
                  f"Clustering: {clustering} | Dim Reduction: {dim_red} | Dimensions: {dimensions}\n"
                  f"{eval_method}: {score:.3f}", fontsize=18)


        plt.xlabel("Component 1", fontsize=16)
        plt.ylabel("Component 2", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right',fontsize=14)

        # Save and show plot
        file_suffix = "best_scatter_original_embeddings" if keep_original_embeddings else "best_scatter_reduced_embeddings"
        file_path = os.path.join(self.get_cluster_exp_path(experiment),f"{file_suffix}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight')
        logger.info(f"Scatter plot generated for the selected experiment saved to {file_path}.")




    def show_best_scatter_with_centers(self, experiment):
        """
        Plots a 2D scatter plot for the best experiment configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """
        
        # Get the experiment data based on the specified `experiment` type
        best_experiment = experiment

        best_labels = np.array(best_experiment['labels'])
        best_id = best_experiment['id']
        eval_method = best_experiment['eval_method']
        best_index = best_experiment['original_index']
        best_centers = best_experiment['centers'].values if isinstance(best_experiment['centers'], pd.DataFrame) else np.array(best_experiment['centers'])
        best_labels = best_experiment['labels']
        clustering = best_experiment['clustering']
        scaler = best_experiment['scaler']
        dim_red = best_experiment['dim_red']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        optimizer = best_experiment['optimization']
        score = best_experiment['score_w/o_penalty'] if eval_method in ("silhouette","davies_bouldin") else best_experiment['score_w_penalty']
        embeddings_used = best_experiment['embeddings']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count

        # Get data reduced from eda object
        data = embeddings_used.values

        # Check if reduction is needed
        if data.shape[1] > 2:
            # If shape > 1, we cannot use selected reduction params, cause it doesnt make sense
            if dim_red == "umap":
                reducer = umap.UMAP(random_state=42, n_components=2, min_dist=0.2, n_neighbors=15)
                reduced_data = reducer.fit_transform(data)
                pca_centers = reducer.transform(best_centers)
            elif dim_red == "tsne":
                reducer = TSNE(random_state=42, n_components=2)
                reduced_data = reducer.fit_transform(data)
                pca_centers = reducer.transform(best_centers)
            else:
                reducer = PCA(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(data)
                pca_centers = reducer.transform(best_centers)
        else:
            # Use the data directly if already 2D
            reduced_data = data
            pca_centers = best_centers


        # Color mapping for clusters and plot setup
        colors = ['#00FF00', '#FFFF00', '#0000FF', '#FF9D0A', '#00B6FF', '#F200FF', '#FF6100']
        cmap_bold = ListedColormap(colors)
        plt.figure(figsize=(12,9))
        
        # Plot noise points (label -1) in red
        noise_points = reduced_data[best_labels == -1]

        print(type(best_labels), type(reduced_data))
        print(best_labels.shape, reduced_data.shape)

        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=10, alpha=0.6, label="Noise (-1)")
        
        # Plot cluster points
        cluster_points = reduced_data[best_labels != -1]
        cluster_labels = best_labels[best_labels != -1]
        scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_labels, cmap=cmap_bold, s=15, alpha=0.6)

        # Plot cluster centers
        if pca_centers is not None:
            plt.scatter(pca_centers[:, 0], pca_centers[:, 1], marker='D', c='black', s=10, label="Cluster Centers", edgecolors='black')
        
        # Add colorbar to distinguish clusters
        plt.colorbar(scatter, spacing="proportional", ticks=np.arange(0, cluster_count + 1, max(1, cluster_count // 10)))

        
        plt.title(f"Scatter Plot for Exp. {best_id} - {optimizer} (Noise in Red) \n\n"
                f"Clustering: {clustering} | Dim Reduction: {dim_red} | Dimensions: {dimensions}\n"
                f"{eval_method}: {score:.3f}", fontsize=18)
        plt.xlabel("Component 1", fontsize=16)
        plt.ylabel("Component 2", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right',fontsize=14)


        # Save and show plot
        file_suffix = "best_scatter_with_centers_reduced"
        file_path = os.path.join(self.get_cluster_exp_path(experiment),f"{file_suffix}.png")
        plt.savefig(file_path, bbox_inches='tight')

        logger.info(f"Scatter plot generated for the selected experiment saved to {file_path}.")




    def show_best_clusters_counters_comparision(self,  experiment):
        """
        Displays a bar chart comparing the number of points in each cluster for the best configuration.
        
        The method retrieves the cluster sizes (number of points per cluster) from `label_counter`
        for the best experiment configuration and displays a bar chart to compare cluster sizes.

        Parameters
        ----------
        show_plots : bool, optional
            If True, displays the plot. Default is False.
        """
         # Get the experiment data based on the specified `experiment` type
        best_experiment = experiment

        best_labels = np.array(best_experiment['labels'])
        best_id = best_experiment['id']
        best_index = best_experiment['original_index']
        best_centers = best_experiment['centers'].values if isinstance(best_experiment['centers'], pd.DataFrame) else np.array(best_experiment['centers'])
        best_labels = best_experiment['labels']
        clustering = best_experiment['clustering']
        scaler = best_experiment['scaler']
        dim_red = best_experiment['dim_red']
        eval_method = best_experiment['eval_method']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        optimizer = best_experiment['optimization']
        embeddings_used = best_experiment['embeddings']
        score = best_experiment['score_w/o_penalty'] if eval_method in ("silhouette","davies_bouldin") else best_experiment['score_w_penalty']
        label_counter = best_experiment['label_counter']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count
        
        label_counter_filtered = {k: v for k, v in label_counter.items() if k != -1}

        # Extract cluster indices and their respective counts from label_counter
        cluster_indices = list(label_counter_filtered.keys())
        cluster_sizes = list(label_counter_filtered.values())

        # Count total with noise and without noise
        total_minus_one = label_counter.get(-1, 0)
        total_rest = sum(v for k, v in label_counter.items() if k != -1)
        
        # Plot the bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(x=cluster_indices, y=cluster_sizes, palette="viridis")
        plt.xlabel("Cluster Index")
        plt.ylabel("Number of Points")
        plt.title(f"Comparison of Cluster Sizes for Exp. {best_id}\n\n" \
                  f"Total cluster points: {total_rest}\n"   \
                  f"Total noise points: {total_minus_one}\n" \
                  f"{eval_method}: {score:.3f}", fontsize=18)
        step = 10
        cluster_indices.sort()
        plt.xticks(ticks=range(0, len(cluster_indices), step), labels=[cluster_indices[i] for i in range(0, len(cluster_indices), step)], rotation=90)
        
        # Save the plot with a name based on the `experiment` type
        file_suffix = "clusters_counter_comparison"
        file_path = os.path.join(self.get_cluster_exp_path(experiment),f"{file_suffix}.png")
        plt.savefig(file_path, bbox_inches='tight')

        logger.info(f"Scatter plot generated for the selected experiment saved to {file_path}.")








if __name__ == "__main__":
    pass
