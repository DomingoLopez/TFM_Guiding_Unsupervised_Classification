import pandas as pd
from src.clustering.clust_agglomerative import AgglomerativeClusteringModel
from src.clustering.clust_dbscan import DBSCANClustering
from src.clustering.clustering_model import ClusteringModel
from src.clustering.clust_hdbscan import HDBSCANClustering
from src.clustering.clust_kmeans import KMeansClustering


class ClusteringFactory:
    """
    Factory class to create instances of clustering models.

    This factory provides a simple interface to create different types of clustering 
    models (e.g., KMeans, DBSCAN) based on the specified method. It abstracts the 
    instantiation process, making it easy to switch between clustering algorithms.
    """

    @staticmethod
    def create_clustering_model(method: str, data: pd.DataFrame, **kwargs) -> ClusteringModel:
        """
        Creates an instance of a clustering model based on the specified method.

        Parameters
        ----------
        method : str
            The clustering algorithm to use. Supported values are:
            - "kmeans": K-means clustering algorithm.
            - "hdbscan": HDBSCAN clustering algorithm.
        data : pd.DataFrame
            The dataset on which clustering will be performed.
        **kwargs : dict
            Additional parameters to be passed to the clustering model.

        Returns
        -------
        ClusteringModel
            An instance of a clustering model, such as KMeansClustering, ready for use.

        Raises
        ------
        ValueError
            If the specified method is not recognized or supported by the factory.
        
        Examples
        --------
        >>> data = pd.DataFrame([...])
        >>> model = ClusteringFactory.create_clustering_model("kmeans", data, n_clusters=3)
        >>> model.run()
        """
        if method == "kmeans":
            return KMeansClustering(data, **kwargs)
        elif method == "hdbscan":
            return HDBSCANClustering(data, **kwargs)
        elif method == "dbscan":
            return DBSCANClustering(data, **kwargs)
        elif method == "agglomerative":
            return AgglomerativeClusteringModel(data, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")


if __name__ == "__main__":
    pass
    # Example usage (uncomment to run):
    # data = pd.DataFrame([...])
    # kmeans_model = ClusteringFactory.create_clustering_model("kmeans", data, n_clusters=2)
    # kmeans_model.run()
    # print("KMeans clustering completed.")
