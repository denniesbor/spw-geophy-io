# %%

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from pathlib import Path
import pandas as pd
import pickle
import seaborn as sns

# Compare greg lucas's results with what we have
# Load the data
data_path = Path("__file__").resolve().parent / "data"

geomag = data_path / "geomagnetic_data.h5"
greg_lucas = data_path / "gl_geoelectric_100year.csv"

# Load the data our data
# Load the data from storm maxes
with h5py.File(geomag, "r") as f:
    # Read MT site information
    mt_names = f["sites/mt_sites/names"][:]
    mt_coords = f["sites/mt_sites/coordinates"][:]

    # Read transmission line IDs
    line_ids = f["sites/transmission_lines/line_ids"][:]

    # Read Halloween storm data
    halloween_e = f["events/halloween/E"][:]
    halloween_b = f["events/halloween/B"][:]
    halloween_v = f["events/halloween/V"][:]

    # Read st_patricks storm data
    st_patricks_e = f["events/st_patricks/E"][:]
    st_patricks_b = f["events/st_patricks/B"][:]
    st_patricks_v = f["events/st_patricks/V"][:]

    # Read the Gannon storm data
    gannon_e = f["events/gannon/E"][:] / 1000
    gannon_b = f["events/gannon/B"][:] / 1000
    gannon_v = f["events/gannon/V"][:] / 1000

    # Read 50-year predictions for all fields
    # e_field_50 = f["predictions/E/250_year"][:]
    # b_field_50 = f["predictions/B/250_year"][:]
    # v_50 = f["predictions/V/250_year"][:]

    # Read 100-year predictions for all fields
    e_field_100 = f["predictions/E/100_year"][:]
    b_field_100 = f["predictions/B/100_year"][:]
    v_100 = f["predictions/V/100_year"][:]

    # Read 500-year predictions for all fields
    e_field_500 = f["predictions/E/500_year"][:]
    b_field_500 = f["predictions/B/500_year"][:]
    v_500 = f["predictions/V/500_year"][:]

    # Read 1000-year predictions for all fields
    e_field_1000 = f["predictions/E/1000_year"][:]
    b_field_1000 = f["predictions/B/1000_year"][:]
    v_1000 = f["predictions/V/1000_year"][:]


# Load the data from Greg Lucas
greg_df = pd.read_csv(
    "/home/dkbor/google-earth-leaflet/data-col-notebooks/data/gl_geoelectric_100year.csv"
)
# %%
# New df
# Convert byte strings to regular strings for mt_names
mt_names = [name.decode("utf-8") for name in mt_names]

# Separate longitude and latitude
mt_latitudes = mt_coords[:, 0]
mt_longitudes = mt_coords[:, 1]

# Create DataFrame
df = pd.DataFrame(
    {
        "Site": mt_names,
        "Latitude": mt_latitudes,
        "Longitude": mt_longitudes,
        "B 100-year": b_field_100,
        "E 100-year": e_field_100,
    }
)

# %%
merged_df = pd.merge(df, greg_df, on="Site", suffixes=("_ours", "_greg"))

# %%
# load mt pickle and max array vals
mt_pickle = data_path / "EMTF" / "mt_pickle.pkl"
max_array_script = data_path / "maxE_arr_testing.npy"
max_array_nb = data_path / "maxes" / "maxE_arr.npy"

with open(mt_pickle, "rb") as f:
    mt_df = pickle.load(f)

max_arr_1 = np.load(max_array_script)
max_arr_2 = np.load(max_array_nb)

our_df = pd.read_csv("geoelectric_100year.csv")

# %%
# max voltages
max_voltages_1 = np.load(data_path / "maxes" / "maxV_arr.npy")
max_voltages_2 = np.load(data_path / "maxV_arr_testing.npy")

# %%

# Merge the two dataframes
merged_df_ = pd.merge(our_df, greg_df, on="Site", suffixes=("_ours", "_greg"))
# %%
# Calculate the differences
merged_df["B100_diff"] = merged_df["B 100-year_ours"] - merged_df["B 100-year_greg"]
merged_df["E100_diff"] = merged_df["E 100-year_ours"] - merged_df["E 100-year_greg"]

# Plot the differences for B100-year and E100-year
plt.figure(figsize=(10, 6))

# Bar plot for B100-year differences
plt.subplot(2, 1, 1)
sns.barplot(x="Site", y="B100_diff", data=merged_df)
plt.title("Difference in B100-year (Ours - Greg's)")
plt.ylabel("B100-year Difference")
plt.xticks(rotation=90)

# Bar plot for E100-year differences
plt.subplot(2, 1, 2)
sns.barplot(x="Site", y="E100_diff", data=merged_df)
plt.title("Difference in E100-year (Ours - Greg's)")
plt.ylabel("E100-year Difference")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# %%
# Plot E100-year comparison
plt.subplot(2, 1, 2)
plt.plot(
    merged_df["Site"], merged_df["E 100-year_ours"], label="E100-year (Ours)", marker="o"
)
plt.plot(
    merged_df["Site"],
    merged_df["E 100-year_greg"],
    label="E100-year (Greg's)",
    marker="x",
)
plt.title("E100-year Comparison (Ours vs Greg's)")
plt.ylabel("E100-year")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Previously calc storm maxes
e_field_array_1 = np.load("maxE_arr.npy")

e_field_array_2 = np.load(data_path / "maxE_arr.npy")

# %%
