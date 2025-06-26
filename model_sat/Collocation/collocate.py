from typing import Optional
import logging
import numpy as np
import xarray as xr
from tqdm import tqdm

from Model.model import SCHISM
from Satellite.satellite import SatelliteData
from Collocation.temporal import temporal_nearest, temporal_interpolated
from Collocation.spatial import SpatialLocator, inverse_distance_weights
from Collocation.output import make_collocated_nc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
_logger = logging.getLogger(__name__)


class Collocate:
    def __init__(
        self,
        model_run: SCHISM,
        satellite: SatelliteData,
        dist_coast: Optional[xr.Dataset] = None,
        n_nearest: int = 3,
        time_buffer: Optional[np.timedelta64] = None,
        weight_power: float = 1.0,
        temporal_interp: bool = False,
    ):
        self.model = model_run
        self.sat = satellite
        self.dist_coast = dist_coast["distcoast"] if dist_coast is not None else None
        self.n_nearest = n_nearest
        self.weight_power = weight_power
        self.temporal_interp = temporal_interp
        # Automatically estimate time buffer if not provided
        if time_buffer is None:
            example_file = self.model.files[0]
            times = self.model.load_variable(example_file)["time"].values

            if len(times) < 2:
                raise ValueError("Cannot infer time_buffer: less than two model timesteps.")

            # Calculate timestep and use half of it as buffer
            timestep = times[1] - times[0]  # Assumes constant step
            self.time_buffer = timestep / 2
            _logger.info(f"Inferred time_buffer as half timestep: {self.time_buffer}")
        else:
            self.time_buffer = time_buffer

        self.locator = SpatialLocator(self.model.mesh_x, self.model.mesh_y)

    def _extract_model_values(self, m_var, times_or_inds, nodes):
        model_data = m_var.values
        depths = self.model.mesh_depth

        values, dpts = [], []

        if self.temporal_interp:
            ib, ia, wts = times_or_inds
            for i, nd in enumerate(nodes):
                v0 = model_data[ib[i], nd]
                v1 = model_data[ia[i], nd]
                values.append(v0 * (1 - wts[i]) + v1 * wts[i])
                dpts.append(depths[nd])
        else:
            for i, (t_idx, nd) in enumerate(zip(times_or_inds, nodes)):
                t = m_var["time"].values[t_idx]
                values.append(m_var.sel(time=t, nSCHISM_hgrid_node=nd).values)
                dpts.append(depths[nd])

        return np.array(values), np.array(dpts)

    def _coast_distance(self, lats, lons):
        if self.dist_coast is None:
            return np.full_like(lats, fill_value=np.nan, dtype=float)
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
            "model_swh_weighted", "bias_raw", "bias_weighted"
        ]}

        include_coast = self.dist_coast is not None
        if include_coast:
            results["dist_coast"] = []

        for path in tqdm(self.model.files, desc="Collocating..."):
            m_var = self.model.load_variable(path)
            m_times = m_var["time"].values

            if self.temporal_interp:
                sat_sub, ib, ia, wts, tdel = temporal_interpolated(self.sat.ds, m_times, self.time_buffer)
                time_args = (ib, ia, wts)
            else:
                sat_sub, idx, tdel = temporal_nearest(self.sat.ds, m_times, self.time_buffer)
                time_args = idx

            dists, nodes = self.locator.query(sat_sub["lon"].values, sat_sub["lat"].values, self.n_nearest)
            m_vals, m_dpts = self._extract_model_values(m_var, time_args, nodes)
            w_sp = inverse_distance_weights(dists, self.weight_power)
            weighted = (m_vals * w_sp).sum(axis=1)

            results["time_sat"].append(sat_sub["time"].values)
            results["lat_sat"].append(sat_sub["lat"].values)
            results["lon_sat"].append(sat_sub["lon"].values)
            results["source_sat"].append(sat_sub["source"].values)
            results["sat_swh"].append(sat_sub["swh"].values)
            results["sat_sla"].append(sat_sub["sla"].values)
            results["model_swh"].append(m_vals)
            results["model_dpt"].append(m_dpts)
            results["dist_deltas"].append(dists)
            results["node_ids"].append(nodes)
            results["time_deltas"].append(tdel)
            results["model_swh_weighted"].append(weighted)
            results["bias_raw"].append(m_vals.mean(axis=1) - sat_sub["swh"].values)
            results["bias_weighted"].append(weighted - sat_sub["swh"].values)

            if include_coast:
                coast_d = self._coast_distance(sat_sub["lat"].values, sat_sub["lon"].values)
                results["dist_coast"].append(coast_d)

        ds_out = make_collocated_nc(results, self.n_nearest)
        if output_path:
            ds_out.to_netcdf(output_path)
        return ds_out
