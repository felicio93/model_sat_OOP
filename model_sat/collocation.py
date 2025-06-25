import logging
from typing import Optional, List, Union

import numpy as np
import xarray as xr
import scipy
from tqdm import tqdm

from model import SCHISM
from satellite import SatelliteData


# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
_logger = logging.getLogger()


def make_collocated_nc(results: dict, n_nearest: int) -> xr.Dataset:
    """
    Converts the results dict to a CF 1.7 compliant xarray Dataset.
    """
    ds = xr.Dataset(
        {
            "lon": (["time"], np.concatenate(results["lon_sat"])),
            "lat": (["time"], np.concatenate(results["lat_sat"])),
            "sat_swh": (["time"], np.concatenate(results["sat_swh"])),
            "sat_sla": (["time"], np.concatenate(results["sat_sla"])),
            "model_swh": (["time", "nearest_nodes"], np.vstack(results["model_swh"])),
            "model_swh_weighted": (["time"], np.concatenate(results["model_swh_weighted"])),
            "model_dpt": (["time", "nearest_nodes"], np.vstack(results["model_dpt"])),
            "dist_deltas": (["time", "nearest_nodes"], np.vstack(results["dist_deltas"])),
            "node_ids": (["time", "nearest_nodes"], np.vstack(results["node_ids"])),
            "time_deltas": (["time"], np.concatenate(results["time_deltas"])),
            "bias_raw": (["time"], np.concatenate(results["bias_raw"])),
            "bias_weighted": (["time"], np.concatenate(results["bias_weighted"])),
            "dist_coast": (["time"], np.concatenate(results["dist_coast"])),
            "source_sat": (["time"], np.concatenate(results["source_sat"])),
        },
        coords={
            "time": np.concatenate(results["time_sat"]),
            "nearest_nodes": np.arange(n_nearest),
        },
    )
    # Add similar attribute setting here as in your previous function...
    ds.attrs["Conventions"] = "CF-1.7"
    ds.attrs["title"] = "CF-compliant Satellite vs Model SWH Dataset"
    return ds


class Collocate:
    """
    Handles collocating satellite and model data spatially and temporally.
    """

    def __init__(
        self,
        model_run: SCHISM,
        satellite: SatelliteData,
        dist_coast: xr.Dataset,
        n_nearest: int = 3,
        time_buffer: np.timedelta64 = np.timedelta64(30, "m"),
        weight_power: float = 1.0,
        temporal_interp: bool = False,
    ):
        self.model = model_run
        self.sat = satellite
        self.dist_coast = dist_coast["distcoast"]
        self.n_nearest = n_nearest
        self.time_buffer = time_buffer
        self.weight_power = weight_power
        self.temporal_interp = temporal_interp

        # Build KDTree
        coords = np.column_stack((self.model.mesh_x, self.model.mesh_y))
        self.tree = scipy.spatial.cKDTree(coords)


    def temporal_nearest(self, ds_sat: xr.Dataset, model_times: np.ndarray):
        # Define extended time range to allow nearest search near time boundaries
        start_date = model_times.min() - self.time_buffer
        end_date = model_times.max() + self.time_buffer

        # Filter satellite data using the time buffer
        sat = ds_sat.sortby("time").sel(time=slice(start_date, end_date))
        sat_times = sat["time"].values

        # Find nearest model time for each satellite time
        nearest_inds = np.abs(sat_times[:, None] - model_times[None, :]).argmin(axis=1)
        nearest_model_times = model_times[nearest_inds]

        time_deltas = (sat_times - nearest_model_times).astype("timedelta64[s]").astype(int)

        return sat, nearest_inds, time_deltas

    def temporal_interpolated(self, ds_sat: xr.Dataset, model_times: np.ndarray):
        # Define extended time range to allow interpolation near time boundaries
        start_date = model_times.min() - self.time_buffer
        end_date = model_times.max() + self.time_buffer

        # Filter and sort satellite data
        sat = ds_sat.sortby("time").sel(time=slice(start_date, end_date))
        sat_times = sat["time"].values
        model_times_s = model_times.astype("datetime64[s]")

        ib, ia, weights, valid_time_indices = [], [], [], []

        for i, t in enumerate(sat_times):
            idx = np.searchsorted(model_times_s, t)
            i0 = max(0, idx - 1)
            i1 = min(len(model_times_s) - 1, idx)

            if i0 == i1:
                continue  # Skip if no valid interval for interpolation

            dt = model_times_s[i1] - model_times_s[i0]
            w = (t - model_times_s[i0]) / dt

            ib.append(i0)
            ia.append(i1)
            weights.append(w)
            valid_time_indices.append(i)
            

        sat_sub = sat.isel(time=valid_time_indices)
        ib = np.array(ib, dtype=int)
        ia = np.array(ia, dtype=int)
        weights = np.array(weights)

        # Find the nearest model time to each satellite timestamp:
        t0 = model_times_s[ib]
        t1 = model_times_s[ia]

        dt0 = np.abs(sat_sub["time"].values - t0)
        dt1 = np.abs(sat_sub["time"].values - t1)
        nearest_model_times = np.where(dt0 <= dt1, t0, t1)
        t_deltas = (sat_sub["time"].values - nearest_model_times).astype("timedelta64[s]").astype(int)

        return sat_sub, ib, ia, weights, t_deltas

    def spatial(self, ds_sat_sub: xr.Dataset):
        pts = np.column_stack((ds_sat_sub["lon"].values, ds_sat_sub["lat"].values))
        return self.tree.query(pts, k=self.n_nearest)

    def extract_model(self, m_var, times_or_inds, nodes):
        model_data = m_var.values
        depths = self.model.mesh_depth

        vals, deps = [], []
        if self.temporal_interp:
            ib, ia, wts = times_or_inds
            for i, nd in enumerate(nodes):
                v0 = model_data[ib[i], nd]
                v1 = model_data[ia[i], nd]
                vals.append(v0 * (1 - wts[i]) + v1 * wts[i])
                deps.append(depths[nd])
        else:
            # Convert indices to actual model times
            model_times = m_var["time"].values
            for i, (t_idx, nd) in enumerate(zip(times_or_inds, nodes)):
                t = model_times[t_idx]
                vals.append(m_var.sel(time=t, nSCHISM_hgrid_node=nd).values)
                deps.append(depths[nd])
        return np.array(vals), np.array(deps)

    def idw_weights(self, dists: np.ndarray):
        w = 1.0 / np.power(np.maximum(dists, 1e-6), self.weight_power)
        return w / w.sum(axis=1, keepdims=True)

    def coast_dist(self, lats: np.ndarray, lons: np.ndarray):
        # lons = convert_longitude(lons, mode=1)
        return self.dist_coast.sel(
            latitude=xr.DataArray(lats, dims="points"),
            longitude=xr.DataArray(lons, dims="points"),
            method="nearest",
        ).values

    def run(self, output_path: Optional[str] = None) -> xr.Dataset:
        results = {k: [] for k in [
            "time_sat", "lat_sat", "lon_sat", "source_sat",
            "sat_swh", "sat_sla", "model_swh", "model_dpt",
            "dist_deltas", "node_ids", "time_deltas",
            "model_swh_weighted", "bias_raw", "bias_weighted", "dist_coast"
        ]}

        for path in tqdm(self.model.files, desc="Collocating..."):
            var = self.model.load_variable(path)

            m_times = var["time"].values
            if self.temporal_interp:
                sat_sub, ib, ia, wts, tdel = self.temporal_interpolated(self.sat.ds, m_times)
                time_args = (ib, ia, wts)
            else:
                sat_sub, idx, tdel = self.temporal_nearest(self.sat.ds, m_times)
                time_args = idx

            dists, nodes = self.spatial(sat_sub)
            model_vals, model_deps = self.extract_model(var, time_args, nodes)
            w_sp = self.idw_weights(dists)
            weighted = (model_vals * w_sp).sum(axis=1)
            coast_d = self.coast_dist(sat_sub["lat"].values, sat_sub["lon"].values)

            results["time_sat"].append(sat_sub["time"].values)
            results["lat_sat"].append(sat_sub["lat"].values)
            results["lon_sat"].append(sat_sub["lon"].values)
            results["source_sat"].append(sat_sub["source"].values)
            results["sat_swh"].append(sat_sub["swh"].values)
            results["sat_sla"].append(sat_sub["sla"].values)
            results["model_swh"].append(model_vals)
            results["model_dpt"].append(model_deps)
            results["dist_deltas"].append(dists)
            results["node_ids"].append(nodes)
            results["time_deltas"].append(tdel)
            results["model_swh_weighted"].append(weighted)
            results["bias_raw"].append(model_vals.mean(axis=1) - sat_sub["swh"].values)
            results["bias_weighted"].append(weighted - sat_sub["swh"].values)
            results["dist_coast"].append(coast_d)

            # ds.close()

        ds_out = make_collocated_nc(results, self.n_nearest)
        if output_path:
            ds_out.to_netcdf(output_path)
        return ds_out