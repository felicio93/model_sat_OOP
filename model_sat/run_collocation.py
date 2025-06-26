import xarray as xr
from Model.model import SCHISM
from Satellite.satellite import SatelliteData
from Satellite import get_sat
import numpy as np
from utils import convert_longitude
from Collocation.collocate import Collocate


# Download the Satellite data
get_sat.get_multi_sat(start_date="2019-07-30",
                      end_date="2019-08-04",
                      sat_list=['sentinel3a','sentinel3b','jason2','jason3','cryosat2','saral'],#swot, sentinel6a,
                      output_dir=r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\OOP_Model_Sat/sat/",
                      lat_min=49.109,
                      lat_max=66.304309,
                      lon_min=156.6854,
                      lon_max=-156.864,
                      ) 


# Paths
sat_path = r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\OOP_Model_Sat/sat/multisat_cropped_2019-07-30_2019-08-04.nc"
model_path = r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\WaveCu_paper\R09b"
dist_coast_path = r'C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\WaveCu_paper\WaveTools\gridinfo/distFromCoast.nc'
output_path = "R09b_collocated_53.nc"
s_time,e_time = "2019-08-01", "2019-08-03"

# Load data (in case you already had downloaded it)
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