
# %%
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np


data_loc = Path('data')
grid_mapping_loc = data_loc / 'grid_mapping.csv'


# Read the file
grid_mapping = pd.read_csv(grid_mapping_loc)

# Eval attributes column
grid_mapping['Attributes'] = grid_mapping['Attributes'].apply(eval)
# %%