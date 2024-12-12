
from itertools import product
import json
from loguru import logger
import numpy as np
import sys
from pathlib import Path
import os
from PIL import Image
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, normalize
import umap



class Preprocess:
    """
    Preprocess class to make scale, normalization, dim reduction, etc.
    """
    def __init__(self, 
                 embeddings=None,
                 dino_model="small",
                 scaler=None,
                 normalization=None,
                 dim_red=None,
                 reduction_params=None,
                 verbose=False,
                 cache = True
                 ):
        
        self.scaler = scaler
        self.dino_model = dino_model
        self.normalization = normalization
        self.dim_red = dim_red
        self.reduction_params = reduction_params 
        self.cache = cache

        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Take embeddings from cache if they dont exists
        if embeddings == None:
            cache_root_folder = Path(__file__).resolve().parent.parent / "dinov2_inference/cache"  
            if Path(cache_root_folder).is_dir():
                logger.info("Accessing cache to recover latest embeddings generated.")
                files = [f for f in Path(cache_root_folder).glob('*') if f.is_file()]
    
            if not files:
                raise FileNotFoundError(f"El directorio de caché no contiene embeddings generados. No es posible recuperarlos. Indique embeddings a analizar.")
            
            # Obtener último archivo generado de embeddings
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            self.embeddings = pickle.load(
                open(str(latest_file), "rb")
            )
        else:
            self.embeddings = embeddings
            
        self.embeddings_df = pd.DataFrame(self.embeddings)

        # Dirs and files
        self.cache_dir = Path(__file__).resolve().parent / "cache" 
        os.makedirs(self.cache_dir, exist_ok=True)




    def run_preprocess(self):
        """
        Run preprocess, including normalization, scaling, dim reduction if needed.
        """
        # Get preprocess embeddings if exist in cache
        emb_d_df = self.check_preprocess_embeddings_on_cache()
        if emb_d_df is not None:
            return emb_d_df
        else:
            emb_n_df = self.do_normalization_l2(self.embeddings_df) if self.normalization else self.embeddings_df
            emb_s_df = self.do_scaler(emb_n_df) if self.scaler else emb_n_df
            emb_d_df = self.do_dim_red(emb_s_df) if self.dim_red else emb_s_df
            # save preprocess
            self.save_preprocess_embeddings(emb_d_df)

            return emb_d_df



    def get_preprocess_embeddings_cache_path(self):
        """
        Get path to store or recover embeddings in cache
        """
        return os.path.join(self.cache_dir, f'dino_model_{self.dino_model}--norm_{self.normalization}--scaler_{self.scaler}--dimred_{self.dim_red}--' + \
                            'reduction_params_'+('_'.join([f'{key}={value}' for key, value in self.reduction_params.items()]) if self.reduction_params is not None else "None") + '.pkl')


    def check_preprocess_embeddings_on_cache(self):
        """
        Check if reduced/normalized/scaled embeddings are available in cache and load them if they are.
        Returns
        -------
        preprocess_embeddings : pd.DataFrame or None
            The cached reduced/normalized/scaled embeddings as a DataFrame if available, else None.
        """
        # Define the path based on different params
        path = self.get_preprocess_embeddings_cache_path()
        # Check if the file exists and load it if available
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    preprocess_embeddings = pickle.load(f)
                return preprocess_embeddings
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                # Maneja errores específicos (archivo no encontrado o carga fallida)
                logger.error(f"Error al cargar las embeddings procesadas desde la caché: {e}")
                return None
        else:
            return None  # Return None if the file doesn't exist


             
    def save_preprocess_embeddings(self, embeddings_df):
        """
        Save reduced embeddings to a cache file.

        Parameters
        ----------
        embeddings_df : pd.DataFrame
            The DataFrame containing the reduced embeddings.
        """
        # Define the path based on scaler and dimensionality reduction technique
        path = self.get_preprocess_embeddings_cache_path()
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save reduced embeddings to cache
        with open(path, "wb") as f:
            pickle.dump(embeddings_df, f)
        logger.info(f"Preprocess embeddings saved to {path}")

            



    def do_normalization_l2(self, embeddings_df):
        """
        Normalize embeddings (vector) in order to improve dim reduction and clustering
        """
        embeddings = embeddings_df.values
        # Get L2 norm
        l2_norms = np.linalg.norm(embeddings, axis=1)
        # Check if they are close to 1
        are_normalized = np.allclose(l2_norms, 1, atol=1e-6)
        if not are_normalized:
            logger.info(f"Applying l2 normalization")
            embeddings_normalized = normalize(embeddings, norm='l2')
        else:
            embeddings_normalized = embeddings

        return pd.DataFrame(embeddings_normalized)


    def do_scaler(self, embeddings_df):
        """
        Apply a specified scaler to the embeddings DataFrame.

        Parameters
        ----------
        type : str
            The type of scaler to apply. Options are "standard", "minmax", "robust", or "maxabs".
        
        Returns
        -------
        embeddings_scaled : pd.DataFrame
            The scaled embeddings as a DataFrame.
        """
        # Scaler options
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler()
        }
        
        logger.info(f"Applying {self.scaler} scaler to embeddings")
        scaler = scalers.get(self.scaler, StandardScaler())
        # Apply scaler
        embeddings = scaler.fit_transform(embeddings_df.values)
        # return embeddings_scaled
        embeddings_scaled = pd.DataFrame(embeddings, columns=embeddings_df.columns)

        return embeddings_scaled


    def do_dim_red(self, embeddings_df):
        """
        Execute  dim reduction
        """
        if self.dim_red == "umap":
            embeddings_dim_red = self.__do_UMAP(embeddings_df)
        elif self.dim_red == "pca":
            embeddings_dim_red = self.__do_PCA(embeddings_df)
        elif self.dim_red == "tsne":
            embeddings_dim_red = self.__do_TSNE(embeddings_df)
        else:
            embeddings_dim_red = self.__do_UMAP(embeddings_df)

        return embeddings_dim_red

    
        
    def __do_PCA(self, embeddings_df):
        """
        PCA Dim reduction. 
        """
        if self.reduction_params is None:
            raise ValueError("No reduction params provided")
        
        logger.info(f"Using PCA Dim. reduction. Params: {', '.join([f'{key}={value}' for key, value in self.reduction_params.items()])}")
        pca = PCA(random_state=42, **self.reduction_params)
        pca_result = pca.fit_transform(embeddings_df.values)
        pca_df = pd.DataFrame(data=pca_result)
        # Eigenvectors
        eigenvectors = pca.components_
        print("Principal components (Eigenvectors):")
        print(eigenvectors)
        # Eigenvalues
        eigenvalues = pca.explained_variance_ratio_ 
        print("Explained variance ratio (Eigenvalues):")
        print(eigenvalues)
        return pca_df


    
    def __do_UMAP(self, embeddings_df):
        """
        UMAP Dim reduction. 
        More info in https://umap-learn.readthedocs.io/en/latest/
        """
        if self.reduction_params is None:
            raise ValueError("No reduction params provided")
        
        logger.info(f"Using UMAP Dim. reduction. Params: {', '.join([f'{key}={value}' for key, value in self.reduction_params.items()])}")
        reducer = umap.UMAP(random_state=42, **self.reduction_params)
        umap_result = reducer.fit_transform(embeddings_df.values)
        umap_df = pd.DataFrame(data=umap_result)
        return umap_df
    

    def __do_TSNE(self, embeddings_df):
        """
        t-SNE Dimensionality Reduction.
        More info at https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        """
        if self.reduction_params is None:
            raise ValueError("No reduction params provided")
        
        logger.info(f"Using t-SNE Dim. reduction. Params: {', '.join([f'{key}={value}' for key, value in self.reduction_params.items()])}")
        reducer = TSNE(random_state=42, **self.reduction_params)
        tsne_result = reducer.fit_transform(embeddings_df.values)
        tsne_df = pd.DataFrame(data=tsne_result)
        return tsne_df




if __name__ == "__main__":
    
    # FALTA GENERAR, SIN REDUCIR, SIN ESCALAR, PERO NORMALIZANDO, 
    # SIN REDUCIR, SIN NORMALIZAR , PERO ESCALANDO
    
    
    experiment = {
        "id": 1,
        "optimizer" : "optuna",
        "normalization": True,
        "scaler" : "standard",
        "dim_red" : None,
        "reduction_parameters" : {
            "metric": ["euclidean","cosine"],
            "n_components": [2,5,7,9,11,13,15],
            "n_neighbors": [2, 5, 10, 15, 20, 50, 100, 200],
            "min_dist": [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
        },
        "clustering" : "hdbscan",
        "eval_method" : "silhouette",
        "penalty" : "",
        "penalty_range" : "",
        "cache" : True
    }
    
    normalization=experiment.get("normalization",None)
    scaler=experiment.get("scaler",None)
    dim_red=experiment.get("dim_red",None)
    reduction_params=experiment.get("reduction_parameters",None)

        
    # For every single combination of params to apply to dim reduction technique
    if reduction_params is not None:
        param_names = list(reduction_params.keys())
        param_values = list(reduction_params.values())
        param_combinations = product(*param_values)
        
        for combination in param_combinations:
            reduction_params = dict(zip(param_names, combination))
            preprocces_obj = Preprocess(embeddings=None, 
                                scaler=scaler, 
                                normalization=normalization,
                                dim_red=dim_red,
                                reduction_params=reduction_params)
            preprocces_obj.run_preprocess()
    else:
        preprocces_obj = Preprocess(embeddings=None, 
                    scaler=scaler, 
                    normalization=normalization,
                    dim_red=dim_red,
                    reduction_params=reduction_params)
        preprocces_obj.run_preprocess()
    
    # Objeto EDA.
    # eda = Preprocess(embeddings=None, verbose=False)
    # scalers = ["no_scaler","standard","minmax","robust","maxabs"]
    # reduction_parameters = {
    #         "metric": ["euclidean","cosine"],
    #         "n_components": [2,5,7,9,11,13,15],
    #         "n_neighbors": [2, 5, 10, 20, 50, 100, 200],
    #         "min_dist": [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
    # }
    # for scaler in scalers:
    #     embeddings_scaled = eda.run_scaler(scaler)
    #     param_names = list(reduction_parameters.keys())
    #     param_values = list(reduction_parameters.values())
    #     param_combinations = product(*param_values)

    #     for combination in param_combinations:
    #         reduction_params = dict(zip(param_names, combination))
    #         dimension = reduction_params.get("n_components", None)
    #         embeddings = eda.run_dim_red(
    #             embeddings_scaled, dim_reduction="umap", scaler=scaler, 
    #             show_plots=False, reduction_params = reduction_params)
    



