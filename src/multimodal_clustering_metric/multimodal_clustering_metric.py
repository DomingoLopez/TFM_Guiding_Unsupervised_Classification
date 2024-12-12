from pathlib import Path
import pickle
import shutil
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.patches as mpatches


class MultiModalClusteringMetric():
    """
    MultiModalClusteringMetric that allows to create stats and metrics from 
    Dinov2 clustering and Llava inference comparision
    """
    def __init__(self, 
                 classification_lvl: int,
                 categories: list,
                 model:str,
                 n_prompt:int,
                 experiment: pd.DataFrame,
                 images_cluster_dict: dict,
                 llava_results_df: pd.DataFrame,
                 cache: bool = True, 
                 verbose: bool = False):
        """
        Loads cluster-images dict and llava inference results.

        """

        self.classification_lvl = classification_lvl
        self.categories = sorted(categories)
        self.model = model
        self.n_prompt = n_prompt
        self.experiment = experiment
        self.images_cluster_dict = images_cluster_dict
        self.llava_results_df= llava_results_df
        self.cache = cache
        self.verbose = verbose

        distinct_colors= [
            "#1f77b4",  # Azul oscuro
            "#ff7f0e",  # Naranja intenso
            "#2ca02c",  # Verde brillante
            "#9467bd",  # Morado oscuro
            "#8c564b",  # Marrón oscuro
            "#e377c2",  # Rosa oscuro
            "#7f7f7f",  # Gris neutro
            "#17becf",  # Azul cian brillante
            "#bcbd22",  # Amarillo oliva oscuro
            "#aec7e8",  # Azul claro
            "#ffbb78",  # Naranja claro
            "#c5b0d5",  # Lavanda
            "#c49c94",  # Marrón claro
            "#98df8a",  # Verde claro
            "#dbdb8d",  # Amarillo pastel
            "#9edae5",  # Cian claro
            "#393b79",  # Azul marino
            "#5254a3",  # Púrpura azulado
            "#6b6ecf",  # Violeta claro
            "#9c9ede",  # Azul lavanda claro
            "#8c6d31",  # Mostaza oscuro
            "#f7b6d2",  # Rosa pastel
            "#d9d9d9",  # Gris claro
            "#3182bd",  # Azul vibrante
            "#6baed6",  # Azul cielo
            "#31a354",  # Verde bosque
        ]



        # Map colors to categories in alphabetical order
        self.category_colors = {
            cat: distinct_colors[i % len(distinct_colors)]
            for i, cat in enumerate(self.categories)
        }

        self.category_colors["NOT VALID"] = "red"  # Reserved for NOT VALID
        self.unknown_category_color = "black"  # Reserved for unknowns


        # Base dirs
        self.results_dir = Path(__file__).resolve().parent / f"results/classification_lvl_{self.classification_lvl}/{self.model}/prompt_{self.n_prompt}/experiment_{experiment['id']}"
        self.results_csv = self.results_dir / f"cluster_vs_llava_stats.csv"
        self.quality_stats_csv = self.results_dir / f"quality.csv"
        self.quality_stats_noise_csv = self.results_dir / f"quality_w_noise.csv"
        self.category_distribution_plot = self.results_dir / f"category_distribution.png"
        self.noise_distribution_plot = self.results_dir / f"noise_distribution.png"

        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Data frame of result stats
        self.result_stats_df = None
        # Get Distinct Categories from inference
        self.categories = sorted(self.llava_results_df['category_llava'].unique())




    def add_cluster_to_llava_inference(self):
        """
        Adding column to Llava Inference csv result in order to add 
        cluster number. 
        """
        # I got images in clusters dict
        # Need to reverse that dict. 
        images_cluster_dict_reverse = {}
        for cluster, images in self.images_cluster_dict.items():
            for image in images:
                image_name = image.resolve().name.split("/")[-1]
                images_cluster_dict_reverse[image_name] = cluster
        # Done, now match image name, and add a column y llava_inference_results
        result_df = self.llava_results_df.copy()
        result_df['cluster'] = result_df['img'].map(images_cluster_dict_reverse)

        return result_df




    def generate_stats(self):
        """
        Generate statistics for each cluster's category distribution.
        Calculate the percentage of images in each cluster belonging to the same category.
        Also, calculate homogeneity and entropy for each cluster.
        """
        # Add column with cluster
        result_df = self.add_cluster_to_llava_inference()

        # Count distinct categories per cluster
        category_counts = result_df.groupby(['cluster', 'category_llava']).size().reset_index(name='count')

        # Get stats from cluster
        cluster_stats = []
        for cluster_name, group in category_counts.groupby('cluster'):
            total_images = group['count'].sum()
            predominant_category = group.loc[group['count'].idxmax(), 'category_llava']
            predominant_count = group['count'].max()
            success_percent = (predominant_count / total_images) * 100

            # Get homegeneity
            homogeneity_k = predominant_count / total_images

            # Calculate entropy
            label_counts = group['count'].values
            probabilities = label_counts / label_counts.sum()
            entropy_k = entropy(probabilities, base=2)

            # Save statistics per cluster
            cluster_stats.append({
                'cluster': cluster_name,
                'total_img': total_images,
                'predominant_category': predominant_category,
                'success_percent': success_percent,
                'homogeneity_k': homogeneity_k,
                'entropy_k': entropy_k
            })

        # Create df with statistics
        self.result_stats_df = pd.DataFrame(cluster_stats)
        self.result_stats_df = self.result_stats_df.merge(category_counts, on='cluster', how='left')
        # Save statistics to csv
        self.result_stats_df.to_csv(self.results_csv, index=False, sep=";")






    def calculate_clustering_quality(self, use_noise, alpha=1.0):
        """
        Calculate the quality metric for clustering based on homogeneity and entropy.

        Args:
            result_stats_df (pd.DataFrame): DataFrame with statistics for each cluster. 
                Expected columns:
                    - 'cluster': Cluster ID.
                    - 'total_img': Total images in the cluster.
                    - 'homogeneity_k': Proportion of images in the dominant category.
                    - 'entropy_k': Entropy of label distribution in the cluster.
            alpha (float): Weight for the entropy penalty in the quality metric.

        Returns:
            dict: A dictionary with the calculated metrics:
                - 'homogeneity_global': Weighted average of cluster homogeneities.
                - 'penalization_global': Weighted average of cluster entropies.
                - 'quality_metric': Final clustering quality score.
        """

        # epsilon in case entropy is 0
        epsilon=1e-6

        # Ensure the required columns are present
        required_columns = ['total_img', 'homogeneity_k', 'entropy_k']
        for col in required_columns:
            if col not in self.result_stats_df.columns:
                raise ValueError(f"Missing required column: {col} in result_stats_df")

        # Remove duplicate entries for clusters (consider only one row per cluster)
        unique_clusters = self.result_stats_df.drop_duplicates(subset=['cluster'])

        if not use_noise:
            unique_clusters = unique_clusters[unique_clusters['cluster'] != -1]

        # Calculate the total number of images across all clusters
        total_images = unique_clusters['total_img'].sum()

        # Calculate global homogeneity (weighted average of homogeneity_k)
        homogeneity_global = (unique_clusters['total_img'] * unique_clusters['homogeneity_k']).sum() / total_images

        # Calculate global penalty (weighted average of entropy_k)
        entropy_global = (unique_clusters['total_img'] * unique_clusters['entropy_k']).sum() / total_images

        # Combine metrics into the quality score
        quality_metric = homogeneity_global / (entropy_global + epsilon)


        # Determine the column suffix
        suffix = "_w_noise" if use_noise else ""

        # Prepare the results with the column suffix
        quality_results = {
            f'homogeneity_global{suffix}': homogeneity_global,
            f'entropy_global{suffix}': entropy_global,
            f'quality_metric{suffix}': quality_metric
        }

        # Convert results to a DataFrame for saving, but return a dict for scalar values
        path_to_save = self.quality_stats_noise_csv if use_noise else self.quality_stats_csv
        pd.DataFrame([quality_results]).to_csv(path_to_save, sep=";", index=False)

        # Return results as scalars
        return quality_results



    def plot_cluster_categories(self, threshold=0.75):
        """
        Plot four stacked bar charts showing the category distribution within each cluster,
        excluding noise (cluster -1). Additionally, create a pie chart for the noise cluster.
        """
        plot_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) != '-1'].copy()
        plot_data['cluster_int'] = pd.to_numeric(plot_data['cluster'], errors='coerce')
        plot_data['cluster_int'] = plot_data['cluster_int'].astype(int)
        plot_data = plot_data.sort_values(by='cluster_int').reset_index(drop=True)
        plot_data['cluster'] = plot_data['cluster_int'].astype(str)

 
        total_images = self.result_stats_df['count'].sum()
        total_noise_images = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']['count'].sum()

       # Define a consistent and alphabetically ordered color map for categories
        unique_categories = sorted(self.result_stats_df['category_llava'].unique())  # Sort alphabetically
        colors = plt.cm.tab20c.colors  # Use a predefined colormap
        category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}  # Assign colors

        # Divide clusters into 4 groups for 4 charts
        n_clusters = plot_data['cluster'].nunique()
        clusters_per_plot = n_clusters // 4 + (1 if n_clusters % 4 != 0 else 0)
        clusters = plot_data['cluster'].unique()  # Ensures clusters are ordered

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(25, 15))
        fig.suptitle("Category Distribution by Cluster (Excluding Noise)\n" \
                     f"Total images: {total_images}", 
                     fontsize=16)

        for i in range(4):
            start_idx = i * clusters_per_plot
            end_idx = min((i + 1) * clusters_per_plot, n_clusters)
            subset_clusters = clusters[start_idx:end_idx]
            plot_subset = plot_data[plot_data['cluster'].isin(subset_clusters)]
            
            # Asegurarse de que los datos estén agrupados y únicos
            plot_subset = plot_subset.groupby(['cluster', 'category_llava'], as_index=False).sum()
            
            # Asegurarse de que los datos estén ordenados por cluster_int
            plot_subset = plot_subset.sort_values(by='cluster_int')

            # Crear tabla pivote
            pivot_data = plot_subset.pivot_table(
                index='cluster', 
                columns='category_llava', 
                values='count', 
                aggfunc='sum',  # Agrega los valores duplicados
                fill_value=0
            )
            
            # Reindexar para garantizar el orden correcto
            pivot_data = pivot_data.reindex(index=plot_subset['cluster'].unique())

            # Crear gráfico de barras apiladas
            ax = axes[i // 2, i % 2]
            pivot_data.plot(kind='bar', stacked=True, ax=ax, color=[category_colors[cat] for cat in pivot_data.columns])
            
            # Colorear etiquetas del eje x según el porcentaje de éxito
            for label in ax.get_xticklabels():
                cluster_id = label.get_text()
                predominant_row = self.result_stats_df[(self.result_stats_df['cluster'].astype(str) == cluster_id) &
                                                    (self.result_stats_df['category_llava'] == 
                                                        self.result_stats_df.loc[self.result_stats_df['cluster'].astype(str) == cluster_id, 'predominant_category'].values[0])]
                if not predominant_row.empty:
                    success_percent = predominant_row['success_percent'].iloc[0]
                    label.set_color('green' if success_percent > threshold * 100 else 'black')

            # Etiquetas y título del gráfico
            ax.set_title(f"Clusters {subset_clusters[0]} to {subset_clusters[-1]}")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Image Count")
            ax.legend().set_visible(False)

        # Create the legend outside the loop
        fig.legend([plt.Line2D([0], [0], color=category_colors[cat], lw=4) for cat in unique_categories],
                unique_categories, title="Category", bbox_to_anchor=(0.93, 0.5), loc='center')
        plt.tight_layout(rect=[0, 0, 0.84, 0.95])
        fig.savefig(self.category_distribution_plot)
        plt.close(fig)

       

        # Create pie chart for the noise cluster (-1)
        noise_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']
        if not noise_data.empty:
            noise_counts = noise_data.groupby('category_llava')['count'].sum()
            noise_counts = noise_counts.sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 15))  # Adjust height for better legend placement
            wedges, texts = ax.pie(noise_counts, startangle=90, colors=[category_colors[cat] for cat in noise_counts.index])
            
            # Add a detailed legend below the chart
            plt.legend(wedges, [f"{label}: {value:.1f}%" for label, value in zip(noise_counts.index, 
                    (noise_counts / noise_counts.sum() * 100).round(1))], 
                    title="Categories", loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=2)
            
            ax.set_title("Category Distribution in Noise Cluster (-1)" \
                        f"Total noise images: {total_noise_images}")
            fig.savefig(self.noise_distribution_plot)
            plt.close(fig)



    def plot_cluster_categories_2(self, threshold=0.75):
        """
        Plot four stacked bar charts showing the category distribution within each cluster,
        excluding noise (cluster -1). Additionally, create a pie chart for the noise cluster.
        """
        # Extraer datos del experimento
        experiment_id = self.experiment["id"]
        classification_lvl = self.classification_lvl
        n_prompt = self.n_prompt
        model_llava = self.model
        model_clustering = self.experiment["clustering"]
        eval_method = self.experiment["eval_method"]
        score_best = self.experiment["score_w/o_penalty"]

        # Preparar datos para los gráficos
        plot_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) != '-1'].copy()

        plot_data['cluster_int'] = pd.to_numeric(plot_data['cluster'], errors='coerce')
        plot_data['cluster_int'] = plot_data['cluster_int'].astype(int)
        plot_data = plot_data.sort_values(by='cluster_int').reset_index(drop=True)
        plot_data['cluster'] = plot_data['cluster_int'].astype(str)

        total_images = self.result_stats_df['count'].sum()
        total_noise_images = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']['count'].sum()

        # Dividir clusters en 4 grupos para 4 gráficos
        n_clusters = plot_data['cluster'].nunique()
        clusters_per_plot = n_clusters // 4 + (1 if n_clusters % 4 != 0 else 0)
        clusters = plot_data['cluster'].unique()  # Garantizar que los clusters estén ordenados

        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(25, 15))
        fig.suptitle(
            f"Category Distribution by Cluster (Excluding Noise)\n"
            f"Experiment ID: {experiment_id}, Classification Level: {classification_lvl}, "
            f"Prompt: {n_prompt}, Llava Model: {model_llava}, Clustering Model: {model_clustering}\n"
            f"Evaluation Method: {eval_method}, Score (w/o Penalty): {score_best:.3f}, Total Images: {total_images}, Noise Images: {total_noise_images}", 
            fontsize=16
        )

        for i in range(4):
            start_idx = i * clusters_per_plot
            end_idx = min((i + 1) * clusters_per_plot, n_clusters)
            subset_clusters = clusters[start_idx:end_idx]
            plot_subset = plot_data[plot_data['cluster'].isin(subset_clusters)]

            # Agrupar y ordenar los datos
            plot_subset = plot_subset.groupby(['cluster', 'category_llava'], as_index=False).sum()
            plot_subset = plot_subset.sort_values(by='cluster_int')

            # Crear tabla pivote
            pivot_data = plot_subset.pivot_table(
                index='cluster', 
                columns='category_llava', 
                values='count', 
                aggfunc='sum',  # Agregar valores duplicados
                fill_value=0
            )
            pivot_data = pivot_data.reindex(index=plot_subset['cluster'].unique())

            ax = axes[i // 2, i % 2]
            pivot_data.plot(kind='bar', stacked=True, ax=ax, 
                        color=[self.category_colors.get(cat, self.unknown_category_color) for cat in pivot_data.columns])

            # Colorear etiquetas del eje x según el porcentaje de éxito
            for label in ax.get_xticklabels():
                cluster_id = label.get_text()
                predominant_row = self.result_stats_df[
                    (self.result_stats_df['cluster'].astype(str) == cluster_id) &
                    (self.result_stats_df['category_llava'] ==
                    self.result_stats_df.loc[self.result_stats_df['cluster'].astype(str) == cluster_id, 'predominant_category'].values[0])
                ]
                if not predominant_row.empty:
                    success_percent = predominant_row['success_percent'].iloc[0]
                    label.set_color('green' if success_percent > threshold * 100 else 'black')

            # Etiquetas y título del gráfico
            ax.set_title(f"Clusters {subset_clusters[0]} to {subset_clusters[-1]}")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Image Count")
            ax.legend().set_visible(False)

        # Crear la leyenda fuera del bucle
        fig.legend(
            [plt.Line2D([0], [0], color=self.category_colors.get(cat, self.unknown_category_color), lw=4) for cat in self.categories],
            [cat[:38] + "..." if len(cat) > 38 else cat for cat in self.categories],  # Truncar nombres largos
            title="Category", bbox_to_anchor=(0.93, 0.5), loc='center'
        )

        plt.tight_layout(rect=[0, 0, 0.84, 0.95])
        fig.savefig(self.category_distribution_plot)
        plt.close(fig)

        # Crear gráfico de pastel para el cluster de ruido (-1)
        noise_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']
        if not noise_data.empty:
            noise_counts = noise_data.groupby('category_llava')['count'].sum()
            noise_counts = noise_counts.sort_index()

            fig, ax = plt.subplots(figsize=(8, 15))  # Ajustar altura para mejorar la colocación de la leyenda
            wedges, texts = ax.pie(
                noise_counts,
                startangle=90,
                colors=[self.category_colors.get(cat, self.unknown_category_color) for cat in noise_counts.index]
            )

            # Añadir leyenda detallada debajo del gráfico
            plt.legend(
                wedges,
                [
                    f"{label[:38]}...: {value:.1f}%" if len(label) > 38 else f"{label}: {value:.1f}%"
                    for label, value in zip(noise_counts.index, (noise_counts / noise_counts.sum() * 100).round(1))
                ],
                title="Categories",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.1),
                ncol=2
            )

            ax.set_title(
                f"Category Distribution in Noise Cluster (-1)\n"
                f"Experiment ID: {self.experiment['id']}, Classification Level: {self.classification_lvl}, "
                f"Prompt: {self.n_prompt}, Llava Model: {self.model}, Clustering Model: {self.experiment['clustering']}\n"
                f"Evaluation Method: {self.experiment['eval_method']}, Score (w/o Penalty): {self.experiment['score_w/o_penalty']:.4f}, "
                f"Total Noise Images: {noise_data['count'].sum()}"
            )
            plt.tight_layout()
            plt.savefig(self.noise_distribution_plot)
            plt.close(fig)


    def plot_cluster_categories_3(self, threshold=0.75):
        """
        Plot two stacked bar charts showing the category distribution within each cluster,
        excluding noise (cluster -1). Additionally, create a pie chart for the noise cluster.
        """
        # Extraer datos del experimento
        experiment_id = self.experiment["id"]
        classification_lvl = self.classification_lvl
        n_prompt = self.n_prompt
        model_llava = self.model
        model_clustering = self.experiment["clustering"]
        eval_method = self.experiment["eval_method"]
        score_best = self.experiment["score_w/o_penalty"]

        # Preparar datos para los gráficos
        plot_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) != '-1'].copy()
        plot_data['cluster_int'] = pd.to_numeric(plot_data['cluster'], errors='coerce')
        plot_data['cluster_int'] = plot_data['cluster_int'].astype(int)
        plot_data = plot_data.sort_values(by='cluster_int').reset_index(drop=True)
        plot_data['cluster'] = plot_data['cluster_int'].astype(str)

        total_images = self.result_stats_df['count'].sum()
        total_noise_images = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']['count'].sum()

        # Dividir clusters en 2 grupos para 2 gráficos
        n_clusters = plot_data['cluster'].nunique()
        clusters_per_plot = n_clusters // 2 + (1 if n_clusters % 2 != 0 else 0)
        clusters = plot_data['cluster'].unique()

        # Crear subplots para 2 gráficos en 2 filas y 1 columna
        fig, axes = plt.subplots(2, 1, figsize=(35, 35))
        fig.suptitle(
            f"Category Distribution by Cluster (Excluding Noise)\n"
            f"Exp.: {experiment_id}, Class Level: {classification_lvl}, "
            f"Prompt: {n_prompt}, Llava Model: {model_llava}, Clustering Model: {model_clustering}\n"
            f"Eval. Method: {eval_method}, Score (w/o Penalty): {score_best:.3f}, Total Images: {total_images}, Noise Images: {total_noise_images}", 
            fontsize=45
        )

        for i in range(2):
            start_idx = i * clusters_per_plot
            end_idx = min((i + 1) * clusters_per_plot, n_clusters)
            subset_clusters = clusters[start_idx:end_idx]
            plot_subset = plot_data[plot_data['cluster'].isin(subset_clusters)]

            # Agrupar y ordenar los datos
            plot_subset = plot_subset.groupby(['cluster', 'category_llava'], as_index=False).sum()
            plot_subset = plot_subset.sort_values(by='cluster_int')

            # Crear tabla pivote
            pivot_data = plot_subset.pivot_table(
                index='cluster', 
                columns='category_llava', 
                values='count', 
                aggfunc='sum', 
                fill_value=0
            )
            pivot_data = pivot_data.reindex(index=plot_subset['cluster'].unique())

            ax = axes[i]
            pivot_data.plot(kind='bar', stacked=True, ax=ax, 
                            color=[self.category_colors.get(cat, self.unknown_category_color) for cat in pivot_data.columns])

            # Colorear etiquetas del eje x según el porcentaje de éxito
            for label in ax.get_xticklabels():
                cluster_id = label.get_text()
                predominant_row = self.result_stats_df[
                    (self.result_stats_df['cluster'].astype(str) == cluster_id) &
                    (self.result_stats_df['category_llava'] ==
                    self.result_stats_df.loc[self.result_stats_df['cluster'].astype(str) == cluster_id, 'predominant_category'].values[0])
                ]
                if not predominant_row.empty:
                    success_percent = predominant_row['success_percent'].iloc[0]
                    label.set_color('green' if success_percent > threshold * 100 else 'black')

            # Etiquetas y título del gráfico
            ax.set_title(f"Clusters {subset_clusters[0]} to {subset_clusters[-1]}",fontsize=24)
            ax.set_ylabel("Image Count", fontsize=24)
            ax.set_xlabel("", fontsize=18)
            ax.tick_params(axis='x', labelsize=28) 
            ax.tick_params(axis='y', labelsize=28) 
            ax.legend().set_visible(False)

        # # Crear la leyenda fuera del bucle
        # fig.legend(
        #     [plt.Line2D([0], [0], color=self.category_colors.get(cat, self.unknown_category_color), lw=4) for cat in self.categories],
        #     [cat[:38] + "..." if len(cat) > 38 else cat for cat in self.categories],
        #     title="Category", bbox_to_anchor=(0.93, 0.5), loc='center', fontsize=22,title_fontsize=22
        # )

        # plt.tight_layout(rect=[0, 0, 0.84, 0.95])
        # fig.savefig(self.category_distribution_plot)
        # plt.close(fig)

        # Crear la leyenda debajo del gráfico principal
        legend_patches = [
            mpatches.Patch(color=self.category_colors.get(cat, self.unknown_category_color), label=cat[:38] + "..." if len(cat) > 38 else cat)
            for cat in self.categories
        ]

        fig.legend(
            handles=legend_patches,    # Usar parches personalizados
            loc="lower center",        # Mueve la leyenda debajo del gráfico
            fontsize=24,               # Tamaño de los valores de la leyenda
            title_fontsize=24,         # Tamaño del título de la leyenda
            ncol=4,                    # Número de columnas en la leyenda
            handleheight=1.2,          # Altura de los cuadros de color
            handlelength=3.0           # Longitud de los cuadros de color
        )

        # Ajuste del diseño para acomodar la leyenda y evitar superposición
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Ajusta el espacio para la leyenda
        fig.savefig(self.category_distribution_plot)
        plt.close(fig)

        # Crear gráfico de pastel para el cluster de ruido (-1)
        noise_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']
        if not noise_data.empty:
            noise_counts = noise_data.groupby('category_llava')['count'].sum()
            noise_counts = noise_counts.sort_index()

            fig, ax = plt.subplots(figsize=(8, 15))
            wedges, texts = ax.pie(
                noise_counts,
                startangle=90,
                colors=[self.category_colors.get(cat, self.unknown_category_color) for cat in noise_counts.index]
            )

            plt.legend(
                wedges,
                [
                    f"{label[:38]}...: {value:.1f}%" if len(label) > 38 else f"{label}: {value:.1f}%"
                    for label, value in zip(noise_counts.index, (noise_counts / noise_counts.sum() * 100).round(1))
                ],
                title="Categories",
                loc="upper center",
                fontsize=18,
                bbox_to_anchor=(0.5, 0.1),
                ncol=2
            )

            ax.set_title(
                f"Category Distribution in Noise Cluster (-1)\n"
                f"Experiment ID: {self.experiment['id']}, Classification Level: {self.classification_lvl}, "
                f"Prompt: {self.n_prompt}, Llava Model: {self.model}, Clustering Model: {self.experiment['clustering']}\n"
                f"Evaluation Method: {self.experiment['eval_method']}, Score (w/o Penalty): {self.experiment['score_w/o_penalty']:.4f}, "
                f"Total Noise Images: {noise_data['count'].sum()}"
            )
            plt.tight_layout()
            plt.savefig(self.noise_distribution_plot)
            plt.close(fig)



if __name__ == "__main__":
    pass
