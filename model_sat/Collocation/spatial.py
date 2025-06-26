import numpy as np
from scipy.spatial import cKDTree


def inverse_distance_weights(distances: np.ndarray,
                             power: float = 1.0) -> np.ndarray:
    """
    Compute inverse distance weights (IDW) with configurable exponent.

    Parameters
    ----------
    distances : np.ndarray
        Distance array to nearest neighbors, shape (N, k)
    power : float, optional
        Power exponent for distance weighting (default is 1.0).
        Use 1.0 for linear, 2.0 for quadratic, etc.

    Returns
    -------
    np.ndarray
        Normalized inverse distance weights of shape (N, k)

    Notes
    -----
    A small epsilon (1e-6) is used to avoid division by zero.
    """
    safe_distances = np.maximum(distances, 1e-6) #to avoid division by zero
    weights = 1.0 / np.power(safe_distances, power)
    return weights / weights.sum(axis=1, keepdims=True)

class SpatialLocator:
    """KDTree-based spatial query engine

    Handles nearest-neighbor lookups between satellite points and
    model grid nodes using a fast cKDTree.

    Methods
    -------
    query(lon, lat, k=3) -> Tuple[np.ndarray, np.ndarray]
        Query for the `k` nearest model nodes to each satellite point.

    Notes
    -----
    Coordinates are assumed to be in the same projected or geodetic
    system (e.g., lon/lat or UTM).
    """

    def __init__(self,
                 x_coords: np.ndarray,
                 y_coords: np.ndarray) -> None:
        """
        Parameters
        ----------
        x_coords : np.ndarray
            X-coordinates (e.g., longitude) of model mesh nodes
        y_coords : np.ndarray
            Y-coordinates (e.g., latitude) of model mesh nodes
        """
        self.tree = cKDTree(np.column_stack((x_coords, y_coords)))

    def query(self,
              lon: np.ndarray,
              lat: np.ndarray,
              k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        lon : np.ndarray
            Longitudes of satellite observations
        lat : np.ndarray
            Latitudes of satellite observations
        k : int, optional
            Number of nearest model neighbors to return (default is 3)

        Returns
        -------
        tuple of np.ndarray
            Distances and indices of nearest model nodes, both of shape (N, k)
        """
        points = np.column_stack((lon, lat))
        distances, indices = self.tree.query(points, k=k)
        return distances, indices
