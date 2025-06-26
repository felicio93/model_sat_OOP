import logging
import os
import re
from typing import List, Union

import numpy as np
import xarray as xr

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
_logger = logging.getLogger()


def natural_sort_key(filename):
    """
    Generates a sorting key that handles numbers correctly.
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', filename)]

def _parse_gr3_mesh(filepath: str):
    """
    Parse a SCHISM hgrid.gr3 mesh file to extract lon, lat, and depth arrays.

    Parameters:
        filepath (str): Path to hgrid.gr3 file.

    Returns:
        Tuple of np.ndarray: (lon, lat, depth)
    """
    with open(filepath, 'r') as f:
        _ = f.readline()  # mesh name
        ne_np_line = f.readline()
        n_elements, n_nodes = map(int, ne_np_line.strip().split())

        lons = np.empty(n_nodes)
        lats = np.empty(n_nodes)
        depths = np.empty(n_nodes)

        for i in range(n_nodes):
            parts = f.readline().strip().split()
            lons[i] = float(parts[1])
            lats[i] = float(parts[2])
            depths[i] = float(parts[3])

    return lons, lats, depths

class SCHISM:
    """
    Encapsulates selecting and loading model files.
    """
    def __init__(self, rundir: str,
                 model_dict: dict,
                 start_date: np.datetime64,
                 end_date: np.datetime64,
                 output_subdir: str = "outputs"):

        self.rundir = rundir
        self.model_dict = model_dict
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.output_dir = os.path.join(self.rundir, output_subdir)

        self._validate_model_dict()
        self._files = self._select_model_files()

        self._mesh_path = os.path.join(self.rundir, 'hgrid.gr3')
        self._mesh_x, self._mesh_y, self._mesh_depth = _parse_gr3_mesh(self._mesh_path)

    def _validate_model_dict(self):
        """
        Ensure the model_dict has the necessary keys.
        """
        required_keys = ['startswith', 'var', 'var_type']
        missing = [k for k in required_keys if k not in self.model_dict]
        if missing:
            raise ValueError(f"Missing keys in model_dict: {missing}")

    def _select_model_files(self) -> List[str]:
        """
        Select model output NetCDF files filtered by filename pattern.
        """
        if not os.path.isdir(self.output_dir):
            _logger.warning(f"Output directory {self.output_dir} does not exist.")
            return []

        all_files = [f for f in os.listdir(self.output_dir)
                     if os.path.isfile(os.path.join(self.output_dir, f))]
        all_files.sort(key=natural_sort_key)

        selected = []
        for fname in all_files:
            if not fname.startswith(self.model_dict['startswith']) or not fname.endswith(".nc"):
                continue

            fpath = os.path.join(self.output_dir, fname)
            try:
                with xr.open_dataset(fpath, decode_times=False) as ds:
                    if 'time' not in ds.variables:
                        continue
                    times = ds['time'].values
                    times = xr.decode_cf(ds[['time']])['time'].values  # decode only time
    
                    if times[-1] >= self.start_date and times[0] <= self.end_date:
                        selected.append(fpath)
            except Exception as e:
                _logger.warning(f"Error reading {fpath}: {e}")
                continue
            # selected.append(os.path.join(self.output_dir, fname))
        if not selected:
            _logger.warning(f"No files matched pattern in {self.output_dir}.\n"
                            f"Make sure the model files fall within {self.start_date} and {self.end_date} ")
        return selected

    def load_variable(self, path: str) -> xr.DataArray:
        """
        Load the variable from a NetCDF file, slicing 3D data if needed.
        """
        _logger.info("Opening model file: %s", path)
        with xr.open_dataset(path) as ds:
            var = ds[self.model_dict['var']]
            if self.model_dict['var_type'] == '3D':
                var = var.isel(nSCHISM_vgrid_layers=-1)
        return var

   
    @property
    def mesh_x(self) -> np.ndarray:
        return self._mesh_x
    @mesh_x.setter
    def mesh_x(self, new_mesh_x: Union[np.ndarray, list]):
        if len(new_mesh_x) != len(self.mesh_x):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_x = new_mesh_x

    @property
    def mesh_y(self) -> np.ndarray:
        return self._mesh_y
    @mesh_y.setter
    def mesh_y(self, new_mesh_y: Union[np.ndarray, list]):
        if len(new_mesh_y) != len(self.mesh_y):
            raise ValueError("New longitude array must match existing size.")
        self._mesh_y = new_mesh_y

    @property
    def mesh_depth(self) -> np.ndarray:
        return self._mesh_depth

    @property
    def files(self) -> List[str]:
        return self._files
