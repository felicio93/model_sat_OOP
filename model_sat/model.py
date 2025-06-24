import logging
import os
import re
from typing import List, Union

import numpy as np
import xarray as xr
from ocsmesh import Mesh

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
_logger = logging.getLogger()


def convert_longitude(lon: Union[float, np.ndarray],
                      mode: int = 1) -> np.ndarray:
    """
    Convert longitudes between common geographic conventions.

    Args:
        lon: array-like of longitudes
        mode: conversion mode:
            - 1: Convert from [-180, 180] to [0, 360] (Greenwich at 0°)
            - 2: Convert from [0, 360] to [-180, 180] (Greenwich at 0°)
            - 3: Convert from [-180, 180] to [0, 360] (Greenwich at 180°)

    Returns:
        np.ndarray of converted longitudes
    """
    _logger.debug("Converting longitude with mode %d", mode)
    lon = np.asarray(lon)
    if mode == 1:
        return lon % 360
    elif mode == 2:
        return np.where(lon > 180, lon - 360, lon)
    elif mode == 3:
        return lon + 180
    return lon

def natural_sort_key(filename):
    """
    Generates a sorting key that handles numbers correctly.
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', filename)]


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
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.output_dir = os.path.join(self.rundir, output_subdir)

        self._validate_model_dict()
        self._files = self._select_model_files()

        self._mesh_path = os.path.join(self.rundir, 'hgrid.gr3')
        self._mesh = Mesh.open(self._mesh_path, crs=4326)

        self._mesh_x = self._mesh.vert2['coord'][:, 0]
        self._mesh_y = self._mesh.vert2['coord'][:, 1]
        self._mesh_depth = self._mesh.value.ravel()

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
            _logger.warning(f"No files matched pattern in {self.output_dir}")
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

    @property
    def mesh_y(self) -> np.ndarray:
        return self._mesh_y

    @property
    def mesh_depth(self) -> np.ndarray:
        return self._mesh_depth

    @property
    def files(self) -> List[str]:
        return self._files

    @property
    def mesh(self) -> Mesh:
        return self._mesh