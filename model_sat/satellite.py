import xarray as xr
import numpy as np

from typing import Union

class SatelliteData:
    def __init__(self, filepath: str):
        self.ds = xr.open_dataset(filepath)
        
        # Check essential variables exist
        required_vars = ['time', 'lon', 'lat', 'swh', 'sla', 'source']
        missing = [v for v in required_vars if v not in self.ds.variables]
        if missing:
            raise ValueError(f"Missing required variables in dataset: {missing}")

    @property
    def time(self):
        return self.ds.time.values

    @property
    def lon(self):
        return self.ds.lon.values

    @property
    def lat(self):
        return self.ds.lat.values

    @property
    def swh(self):
        return self.ds.swh.values

    @property
    def sla(self):
        return self.ds.sla.values

    @property
    def source(self):
        return self.ds.source.values

    def filter_by_time(self, start_date: Union[str, np.datetime64], end_date: Union[str, np.datetime64]) -> xr.Dataset:
        """
        Filter dataset by time slice.
        Returns filtered dataset (and updates self.ds).
        """
        start_date = np.datetime64(start_date)
        end_date = np.datetime64(end_date)
        self.ds = self.ds.sel(time=slice(start_date, end_date))
        return self.ds