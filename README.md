# model_sat_OOP

Download, crop, and collocate satellite wave height data with model outputs (e.g. SCHISM).

## Install

```
pip install git+https://github.com/felicio93/model_sat_OOP
```
## Usage
```
import xarray as xr
from Model.model import SCHISM
from Satellite.satellite import SatelliteData
import numpy as np
from utils import convert_longitude
from Collocation.collocate import Collocate

# Paths
sat_path = r"Path"
model_path = r"Path"
dist_coast_path = r'Path'
output_path = "Path/collocated.nc"
s_time,e_time = "2019-08-01", "2019-08-03"
# Load data
sat_data = SatelliteData(sat_path)
sat_data.lon = convert_longitude(sat_data.lon,mode=1)

model_run = SCHISM(
    rundir=model_path,
    model_dict={'var': 'sigWaveHeight',
                'startswith': 'out2d_',
                'var_type': '2D',
                'model': 'SCHISM'},
    start_date=np.datetime64(s_time),
    end_date=np.datetime64(e_time)
)

dist_coast = xr.open_dataset(dist_coast_path)

# Collocate
coll = Collocate(
    model_run=model_run,
    satellite=sat_data,
    dist_coast=dist_coast,
    n_nearest=3,
    # time_buffer=np.timedelta64(30, "m"),
    weight_power=1.0,
    temporal_interp=True  # or True if you want interpolated matching
)

ds_out = coll.run(output_path=output_path)
print("Collocation completed. Output saved to:", output_path)
```