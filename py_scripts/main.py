'''
Module to estimate induced voltagesin the transmission line during a geomagnetic storm.
The module levrerages the bezos package to estimate the induced voltages in the transmission line.
the module is developed by Greg  Lucas and is based on the work of Lucas et al.
'''

# %%
from pathlib import Path
import random
from estimate_voltages import PredictBEFields
from viz import viz_voltages 
from load_data.data_manager import DataManager
from datetime import datetime
import pytz
import pandas as pd


# Use default paths to the grid regions file (path and regions)
current_dir = Path(__file__).resolve().parent

# Construct the path to the data directory
data_dir = current_dir / 'data'

emtf_data = data_dir / 'EMTF'
tl_data = data_dir / "US_transmission_lines.geojson"
mag_data = data_dir / 'magnetic_data' 
grid_region = None;
grid_regions_path = data_dir / 'ferc_regions' / 'FERC_grid_regions.geojson'

# %%
# if __name__ == "__main__":
# Data manager
data_manager = DataManager(ferc_path=grid_regions_path, region=grid_region)
data_manager.load_magnetic_data(mag_data_folder=mag_data, year_of_data='2024') # Use default path to magnetic observatories data
data_manager.load_emtf_data(emtf_data) # Use default path to EMTF data
data_manager.load_tl_data(tl_path=tl_data) # Use default path to TL data
# %%
# Predict the induced voltages
e_b_fields_predictor = PredictBEFields(data_manager)

transmision_lines_v.head(5)
# %%
