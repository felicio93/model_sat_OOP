
from model import SCHISM
from satellite import SatelliteData
import xarray as xr
import numpy as np
import collocation
from utils import convert_longitude

s_time,e_time = "2019-08-01", "2019-08-03"
sat = SatelliteData(r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\WaveCu_paper\allsat.nc")
sat.lon = convert_longitude(sat.lon,mode=1)
# sat.filter_by_time(s_time,e_time)

model = SCHISM(
    rundir=r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\WaveCu_paper\R09b",
    model_dict={'var': 'sigWaveHeight', 'startswith': 'out2d_', 'var_type': '2D', 'model': 'SCHISM'},
    start_date=np.datetime64(s_time),
    end_date=np.datetime64(e_time)
)

dist_coast = xr.open_dataset(r'C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\WaveCu_paper\WaveTools\gridinfo/distFromCoast.nc')

coll = collocation.Collocate(model, sat, dist_coast, temporal_interp=False)
result_ds = coll.run(output_path="R09b_collocated_temp_interp9.nc")