import numpy as np
from scipy.spatial import cKDTree


class SpatialLocator:
    """
    Handles spatial queries using a KDTree for fast nearest-neighbor lookup.
    """

    def __init__(self, x_coords: np.ndarray, y_coords: np.ndarray):
        """
        Initialize the spatial index using model mesh node coordinates.

        Args:
            x_coords (np.ndarray): Longitudes of the model mesh nodes.
            y_coords (np.ndarray): Latitudes of the model mesh nodes.
        """
        self.tree = cKDTree(np.column_stack((x_coords, y_coords)))

    def query(self, lon: np.ndarray, lat: np.ndarray, k: int = 3):
        """
        Query the KDTree for the nearest model nodes to the satellite points.

        Args:
            lon (np.ndarray): Satellite longitudes.
            lat (np.ndarray): Satellite latitudes.
            k (int): Number of nearest neighbors to return.

        Returns:
            tuple: (distances, indices) of shape (N, k)
        """
        points = np.column_stack((lon, lat))
        distances, indices = self.tree.query(points, k=k)
        return distances, indices


def inverse_distance_weights(distances: np.ndarray,
                             power: float = 1.0) -> np.ndarray:
    """
    Compute inverse distance weights (IDW) with optional exponent.

    Args:
        distances (np.ndarray): Array of distances to nearest neighbors (N, k).
        power (float): Exponent used in IDW. Default is 1.0.

    Returns:
        np.ndarray: Normalized weights of shape (N, k).
    
    Notes:
        power parameter to control the steepness of the weighting (e.g., 1 = linear, 2 = quadratic).
    """
    safe_distances = np.maximum(distances, 1e-6) #to avoid division by zero
    weights = 1.0 / np.power(safe_distances, power)
    return weights / weights.sum(axis=1, keepdims=True)