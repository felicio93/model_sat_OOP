import xarray as xr
import numpy as np


def make_collocated_nc(results: dict, n_nearest: int) -> xr.Dataset:
    data_vars = {
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
        "source_sat": (["time"], np.concatenate(results["source_sat"])),
    }

    if "dist_coast" in results:
        data_vars["dist_coast"] = (["time"], np.concatenate(results["dist_coast"]))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.concatenate(results["time_sat"]),
            "nearest_nodes": np.arange(n_nearest),
        },
        attrs={
            "Conventions": "CF-1.7",
            "title": "CF-compliant Satellite vs Model SWH Dataset",
        }
    )
    return ds