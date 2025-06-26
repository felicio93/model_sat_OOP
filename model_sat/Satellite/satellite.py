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
    @lon.setter
    def lon(self, new_lon: Union[np.ndarray, list]):
        if len(new_lon) != len(self.ds.lon):
            raise ValueError("New longitude array must match existing size.")
        self.ds['lon'] = ('time', np.array(new_lon))

    @property
    def lat(self):
        return self.ds.lat.values
    @lat.setter
    def lat(self, new_lat: Union[np.ndarray, list]):
        if len(new_lat) != len(self.ds.lat):
            raise ValueError("New latitude array must match existing size.")
        self.ds['lat'] = ('time', np.array(new_lat))
        
    @property
    def swh(self):
        return self.ds.swh.values

    @property
    def sla(self):
        return self.ds.sla.values

    @property
    def source(self):
        return self.ds.source.values

    def filter_by_time(self, start_date: str, end_date: str):
        # Convert to datetime for safety
        start = np.datetime64(start_date)
        end = np.datetime64(end_date)

        # Ensure time is datetime64 and sorted
        if not np.issubdtype(self.ds['time'].dtype, np.datetime64):
            self.ds['time'] = xr.decode_cf(self.ds).time

        self.ds = self.ds.sortby('time')  # Ensure sorted time axis
        self.ds = self.ds.sel(time=slice(start, end))