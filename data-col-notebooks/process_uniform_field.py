# ...............................................................................
# Description: Levereages the admittance matrix to calculate the GICs along the
# transmission lines and transformers in the network.
# Dependency: Output from build_admittance_matrix.py and data from data_preprocessing.ipynb
# ...............................................................................

# %%
import warnings
import sys
import os
import pickle
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from shapely import wkt
from scipy import stats
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from memory_profiler import profile
from shapely.ops import transform
from scipy.interpolate import griddata
import pyproj
from shapely.geometry import LineString
from shapely.geometry import Point  # Point class
import geopandas as gpd


# Custom logger
# Set up logging, a reusable function
def setup_logging():
    """
    Set up logging to both file and console.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler("final_data_analysis.log")
    file_handler.setLevel(logging.INFO)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# Data loc
data_loc = Path.cwd() / "data"
# %%

#  A bool that allow user to specify data collected by grid mapping team
consider_real_transformer_data = True
if consider_real_transformer_data:
    # Read the admittance matrix
    Y = np.load(data_loc / "admittance_matrix_real.npy")

else:
    # Read the admittance matrix
    Y = np.load(data_loc / "admittance_matrix.npy")

# Read the transmission line data
df_lines = pd.read_csv(data_loc / "transmission_lines.csv")
df_lines.drop(columns=["geometry"], inplace=True)
df_lines["name"] = df_lines["name"].astype(np.int32)

transmission_line_path = (
    data_loc / "Electric__Power_Transmission_Lines" / "trans_lines_pickle.pkl"
)

with open(transmission_line_path, "rb") as p:
    trans_lines_gdf = pickle.load(p)

trans_lines_gdf["line_id"] = trans_lines_gdf["line_id"].astype(np.int32)
# Merge trnaslines geometry with the df_lines
df_lines = df_lines.merge(
    trans_lines_gdf[["line_id", "geometry"]], right_on="line_id", left_on="name"
)

# Substions info
df_substations_info = pd.read_csv(data_loc / "substation_info.csv")
df_substations_info.buses = df_substations_info.buses.apply(lambda x: eval(x))
# Transformer data
df_transformers = pd.read_csv(data_loc / "transformers.csv")

# Transformer counts dictionary
with open(data_loc / "transformer_counts.pkl", "rb") as f:
    transformer_counts_dict = pickle.load(f)

# %%
# Load the data from storm maxes
with h5py.File(data_loc / "geomagnetic_data.h5", "r") as f:

    logger.info("Reading geomagnetic data from geomagnetic_data.h5")
    # Read MT site information
    mt_names = f["sites/mt_sites/names"][:]
    mt_coords = f["sites/mt_sites/coordinates"][:]

    # Read transmission line IDs
    line_ids = f["sites/transmission_lines/line_ids"][:]

    # Read Halloween storm data
    halloween_e = f["events/halloween/E"][:] / 1000
    halloween_b = f["events/halloween/B"][:]
    halloween_v = f["events/halloween/V"][:]

    # Read st_patricks storm data
    st_patricks_e = f["events/st_patricks/E"][:] / 1000
    st_patricks_b = f["events/st_patricks/B"][:]
    st_patricks_v = f["events/st_patricks/V"][:]

    # Read the Gannon storm data
    gannon_e = f["events/gannon/E"][:] / 1000
    gannon_b = f["events/gannon/B"][:]
    gannon_v = f["events/gannon/V"][:]

    # Read 50-year predictions for all fields
    e_field_50 = f["predictions/E/250_year"][:]
    b_field_50 = f["predictions/B/250_year"][:]
    v_50 = f["predictions/V/250_year"][:]

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

# Voltage cols for all events
v_cols = ["V_halloween", "V_st_patricks", "V_gannon", "V_100", "V_500", "V_1000"]

# Create a resuable mask for querying the data
id_to_index = {id: i for i, id in enumerate(line_ids)}

# Create an array of indices
indices = np.array([id_to_index.get(name, -1) for name in df_lines["name"]])

# Use boolean indexing to handle missing values
mask = indices != -1
df_lines.loc[mask, "V_halloween"] = halloween_v[indices[mask]]
df_lines.loc[mask, "V_st_patricks"] = st_patricks_v[indices[mask]]
df_lines.loc[mask, "V_gannon"] = gannon_v[indices[mask]]
df_lines.loc[mask, "V_50"] = v_50[indices[mask]]
df_lines.loc[mask, "V_100"] = v_100[indices[mask]]
df_lines.loc[mask, "V_500"] = v_500[indices[mask]]
df_lines.loc[mask, "V_1000"] = v_1000[indices[mask]]

# Set a default value for all missing values
df_lines[v_cols] = df_lines[v_cols].fillna(0)

# # Zip the MT sites coordinates with interpolated E fields
# mt_coords_e = list(zip(mt_coords, e_field_100, e_field_500, e_field_1000))

# %%%
# ...............................................................................
# Custom functions used in solving for nodal currents, voltages and line currents
# Derived from the functions in estimating a uniform geolectric field
# ...............................................................................


# Function to find the substation name for a given bus
def find_substation_name(bus, sub_ref):
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name

    # If not found, return None
    logger.error(f"Bus {bus} not found in any substation.")
    return None


# %%
# Create a dictionary for quick substation lookup
sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))


# %%
def adjust_index(bus, test=False):
    if test:
        return bus - 1 if bus < 9 else bus - 3
    else:
        return bus - 1


def safe_inverse(value, min_denominator=1e-10):
    if pd.isnull(value) or abs(value) < min_denominator:
        return 0
    return 1 / value


# %%
# Calculate the injection currents for the network
def calculate_injection_currents(df, n_nodes, col):
    """
    Calculate the injection currents for each node in a network based on the given dataframe.
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the network data.
    - n_nodes (int): The number of nodes in the network.
    - col (str): The column name representing the source currents in the dataframe.
    Returns:
    - injection_currents (numpy.ndarray): An array of injection currents for each node in the network.
    """
    logger.info(f"Calculating injection currents for {col}...")

    # Initialize injection currents for x and y components
    injection_currents = np.zeros(n_nodes)

    # Calculate line currents and injection currents
    for i, line in df.iterrows():
        # Get the source currents
        I_eff = line[col] / line["R"]

        if np.isnan(I_eff):
            logger.error(f"NaN current encountered for line {line['name']}")
            continue

        # Get i and j
        i, j = adjust_index(line["from_bus"]), adjust_index(line["to_bus"])

        # Currents into and out of the nodes
        injection_currents[i] -= I_eff
        injection_currents[j] += I_eff

    return injection_currents


# Solve for nodal voltages
def solve_nodal_voltages(Y, injection_currents):
    # Solve for nodal voltages
    # Due to size of the matrix add regularization term
    # Skews the results slightly but is fairly faste
    Y_reg = Y.T @ Y + 1e-6 * eye(Y.shape[1])
    V_nodal = spsolve(Y_reg, Y.T @ injection_currents)

    # Return nodal voltages
    return V_nodal


# %%
# Jnk is the current between the bus n and k
# Vn and Vk are the nodal voltages
# Y nk is a system of admittances btwn n and k
# Jnk = Ynk * (Vn - Vk)
def calculate_GIC(df, V_nodal, col):
    """
    Calculate the Ground Induced Current (GIC) for a given dataframe.
    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the transmission line data.
    - V_nodal (numpy.ndarray): The nodal voltages.
    - col (str): The column name representing the GIC values.
    Returns:
    - df (pandas.DataFrame): The input dataframe with an additional column representing the calculated GIC values.
    """

    logger.info(f"Calculating GIC for {col}...")

    # Get the nodal voltages bus ids
    bus_n = df["from_bus"].values - 1
    bus_k = df["to_bus"].values - 1

    y_nk = 1 / df["R"].values
    j_nk = df[col].values * y_nk

    # Get the nodal voltages
    vn = V_nodal[bus_n]
    vk = V_nodal[bus_k]

    # Solving for transmission lines GIC
    i_nk = j_nk + (vn - vk) * y_nk

    df[f"{col.split('_')[1]}_i_nk"] = i_nk

    return df


# %%
def calc_trfmr_gic(df, V_nodal, sub_ref=sub_ref):
    """
    Calculate the GIC (Geomagnetically Induced Current) for each transformer in the given dataframe.
    Parameters:
    - df: pandas DataFrame
        The dataframe containing transformer information.
    - V_nodal: list
        The list of nodal voltages.
    - sub_ref: str, optional
        The reference for the substation.
    Returns:
    - df: pandas DataFrame
        The dataframe with the added GIC information for each transformer.
    Raises:
    - Warning
        If no substation is found for a transformer.
    - Warning
        If no grounding resistance is found for a substation.
    - Warning
        If invalid W1 or W2 values are found for a Tee network transformer.
    - Warning
        If invalid Y_series or Y_common values are found for an Auto transformer.
    - Warning
        If invalid Y_primary or Y_secondary values are found for a transformer.
    - Warning
        If an unknown transformer type is encountered.
    """

    logger.info("Calculating transformer GIC...")

    # Get the transformers GIC
    # Add transformer admittances
    df["gic"] = None
    # gic
    for i, row in df.iterrows():
        if consider_real_transformer_data:
            # Get the tr count
            tr_count = transformer_counts_dict.get(row["sub_id"], 1)

        else:
            tr_count = 1

        sub = find_substation_name(row["bus1"], sub_ref)
        tranformer_gic = {}
        if sub is None:
            logger.error(
                f"No substation found for bus {row['bus1']} in transformer {row['name']}"
            )
            continue

        Rg = df_substations_info[df_substations_info.name == sub][
            "grounding_resistance"
        ]
        if Rg.empty:
            logger.error(f"No grounding resistance found for substation {sub}")
            Rg = 0
        else:
            Rg = Rg.iloc[0]

        if row["type"] in ["GSU", "GY-D"]:
            Y_gsu = safe_inverse(pd.to_numeric(row["W1"], errors="coerce") + Rg)
            Y_gsu = Y_gsu * tr_count

            bus_n = row["bus1"]
            i_k = V_nodal[bus_n - 1] * Y_gsu
            tranformer_gic["i_k"] = i_k

        elif row["type"] == "Tee":
            W1 = pd.to_numeric(row["W1"], errors="coerce")
            W2 = pd.to_numeric(row["W2"], errors="coerce")
            bus_n = row["bus1"]
            bus_k = row["bus2"]

            if pd.notnull(W1) and pd.notnull(W2):
                # Current through the t -network is impedance * voltage diff between buses
                y_k = safe_inverse(W1 + W2)
                y_k = y_k * tr_count
                i_nk = (V_nodal[bus_n - 1] - V_nodal[bus_k - 1]) * y_k
                tranformer_gic["i_nk"] = i_nk
            else:
                logger.error(
                    f"Invalid W1 or W2 for Tee network in transformer {row['name']}"
                )

        elif row["type"] == "GSU w/ GIC BD":
            R_blocking = 1e6
            Y_gsu = safe_inverse(pd.to_numeric(row["W1"], errors="coerce") + R_blocking)
            Y_gsu = Y_gsu * tr_count
            # Current flows only through the secondarty (high voltage side)
            bus_n = row["bus1"]
            i_k = V_nodal[bus_n - 1] * Y_gsu
            tranformer_gic["i_k"] = i_k

        elif row["type"] == "Auto":
            Y_series = safe_inverse(pd.to_numeric(row["W1"], errors="coerce"))
            Y_common = safe_inverse(pd.to_numeric(row["W2"], errors="coerce") + Rg)
            Y_series = Y_series * tr_count
            Y_common = Y_common * tr_count

            bus_n = row["bus1"]
            bus_k = row["bus2"]
            if pd.notnull(Y_series) and pd.notnull(Y_common):
                # In this scenario, there are two possibilities (current through the serial and current throuigh common to ground)
                I_s = (V_nodal[bus_n - 1] - V_nodal[bus_k - 1]) * Y_series
                I_c = (V_nodal[bus_k - 1]) * Y_common
                tranformer_gic["i_s"] = I_s
                tranformer_gic["i_c"] = I_c

            else:
                logger.error(
                    f"Invalid Y_series or Y_common for Auto transformer {row['name']}"
                )

        elif row["type"] in ["GY-GY-D", "GY-GY"]:
            Y_primary = safe_inverse(pd.to_numeric(row["W1"], errors="coerce") + Rg)
            Y_secondary = safe_inverse(pd.to_numeric(row["W2"], errors="coerce") + Rg)
            Y_primary = Y_primary * tr_count
            Y_secondary = Y_secondary * tr_count

            bus_n = row["bus1"]
            bus_k = row["bus2"]
            if pd.notnull(Y_primary) and pd.notnull(Y_secondary):
                I_w1 = (V_nodal[bus_n - 1]) * Y_primary
                I_w2 = (V_nodal[bus_k - 1]) * Y_secondary
                tranformer_gic["i_w1"] = I_w1
                tranformer_gic["i_w2"] = I_w2
            else:
                logger.error(
                    f"Invalid Y_primary or Y_secondary for transformer {row['name']}"
                )
        else:
            logger.error(
                f"Unknown transformer type {row['type']} for transformer {row['name']}"
            )

        df.at[i, "gic"] = tranformer_gic

        logger.info(f"Transformer GIC calculated for {row['name']}")

    return df


# %%

n_nodes = len(
    set([bus for bus_sublist in df_substations_info.buses for bus in bus_sublist])
)

injections_halloween = calculate_injection_currents(df_lines, n_nodes, "V_halloween")
injections_st_patricks = calculate_injection_currents(
    df_lines, n_nodes, "V_st_patricks"
)
injections_gannon = calculate_injection_currents(df_lines, n_nodes, "V_gannon")
injections_50 = calculate_injection_currents(df_lines, n_nodes, "V_50")
injections_100 = calculate_injection_currents(df_lines, n_nodes, "V_100")
injections_500 = calculate_injection_currents(df_lines, n_nodes, "V_500")
injections_1000 = calculate_injection_currents(df_lines, n_nodes, "V_1000")

# Solve for nodal voltages
V_nodal_halloween = solve_nodal_voltages(Y, injections_halloween)
V_nodal_st_patricks = solve_nodal_voltages(Y, injections_st_patricks)
V_nodal_gannon = solve_nodal_voltages(Y, injections_gannon)
V_nodal_50 = solve_nodal_voltages(Y, injections_50)
V_nodal_100 = solve_nodal_voltages(Y, injections_100)
V_nodal_500 = solve_nodal_voltages(Y, injections_500)
V_nodal_1000 = solve_nodal_voltages(Y, injections_1000)


df_gic_halloween = calc_trfmr_gic(
    df_transformers, V_nodal_halloween, sub_ref=sub_ref
)  # GIC Halloweeeen
df_gic_halloween.to_csv(data_loc / "gic_halloween.csv", index=False)  # Export
df_gic_st_patricks = calc_trfmr_gic(
    df_transformers, V_nodal_st_patricks, sub_ref=sub_ref
)  # GIC St. Patricks
df_gic_st_patricks.to_csv(data_loc / "gic_st_patricks.csv", index=False)  # Export
df_gic_gannon = calc_trfmr_gic(
    df_transformers, V_nodal_gannon, sub_ref=sub_ref
)  # GIC Gannon
df_gic_gannon.to_csv(data_loc / "gic_gannon.csv", index=False)  # Export
df_gic_100 = calc_trfmr_gic(
    df_transformers, V_nodal_100, sub_ref=sub_ref
)  # 100 year hazards
df_gic_100.to_csv(data_loc / "gic_100.csv", index=False)  # Export
df_gic_500 = calc_trfmr_gic(
    df_transformers, V_nodal_500, sub_ref=sub_ref
)  # 500 year gic
df_gic_500.to_csv(data_loc / "gic_500.csv", index=False)
df_gic_1000 = calc_trfmr_gic(
    df_transformers, V_nodal_1000, sub_ref=sub_ref
)  # 1000 year gic
df_gic_1000.to_csv(data_loc / "gic_1000.csv", index=False)  # Export

# %%   

# PLot gridded countour map
@profile
def get_conus_polygon():
    """
    Retrieves the polygon representing the continental United States (CONUS).

    Returns:
        A polygon object representing the boundary of the continental United States.

    Raises:
        None.
    """
    logger.info("Retrieving CONUS polygon...")

    try:
        shapename = "admin_1_states_provinces_lakes"
        us_states = shpreader.natural_earth(
            resolution="110m", category="cultural", name=shapename
        )

        conus_states = []
        for state in shpreader.Reader(us_states).records():
            if state.attributes["admin"] == "United States of America":
                # Exclude Alaska and Hawaii
                if state.attributes["name"] not in ["Alaska", "Hawaii"]:
                    conus_states.append(state.geometry)

        conus_polygon = unary_union(conus_states)
        logger.info("CONUS polygon retrieved.")
        return conus_polygon
    except Exception as e:
        logger.error(f"Failed to retrieve CONUS polygon: {e}")
        return None


# For efficieny, we can preprocess tghe grid x, y, and gridz and pickle them
# Then load them in the carto function
def generate_grid_and_mask(
    e_fields,
    mt_coordinates,
    resolution=(500, 1000),
    filename=data_loc / "grid_mask.pkl",
):

    if mt_coordinates.shape[0] != e_fields.shape[0]:
        print("Warning: Number of points and values still don't match!")

    logging.info("Generating grid and mask...")
    lon_min, lon_max = np.min(mt_coordinates[:, 1]), np.max(mt_coordinates[:, 1])
    lat_min, lat_max = np.min(mt_coordinates[:, 0]), np.max(mt_coordinates[:, 0])

    grid_x, grid_y = np.mgrid[
        lon_min : lon_max : complex(0, resolution[0]),
        lat_min : lat_max : complex(0, resolution[1]),
    ]

    conus_polygon = get_conus_polygon()

    if conus_polygon is not None:
        mask = np.array(
            [
                conus_polygon.contains(Point(x, y))
                for x, y in zip(grid_x.ravel(), grid_y.ravel())
            ]
        ).reshape(grid_x.shape)
    else:
        mask = (
            (grid_x >= lon_min)
            & (grid_x <= lon_max)
            & (grid_y >= lat_min)
            & (grid_y <= lat_max)
        )

    # Interpolate the E-field data onto the grid
    grid_z = griddata(
        mt_coordinates[:, [1, 0]], e_fields, (grid_x, grid_y), method="linear"
    )

    # Apply mask to grid_z only
    grid_z = np.ma.array(grid_z, mask=~mask)

    with open(filename, "wb") as f:
        pickle.dump((grid_x, grid_y, grid_z, e_fields), f)
        logging.info("Grid and mask saved to file.")


grid_e_100_path = data_loc / "grid_e_100.pkl"
grid_e_500_path = data_loc / "grid_e_500.pkl"
grid_e_1000_path = data_loc / "grid_e_1000.pkl"
grid_e_gannon_path = data_loc / "grid_e_gannon.pkl"

grid_file_paths = [
    grid_e_100_path,
    grid_e_500_path,
    grid_e_1000_path,
    grid_e_gannon_path,
]

e_fields = [e_field_100, e_field_500, e_field_1000, gannon_e]


# Prepare the transmission lines data for plotting
for grid_filename, e_field in zip(grid_file_paths, e_fields):
    if not os.path.exists(grid_filename):
        # Generate and save the grid and mask
        generate_grid_and_mask(
            e_field, mt_coords, resolution=(500, 1000), filename=grid_filename
        )

logging.info("Grid and mask generated and saved.")

# %%

# Prepare transmission lines data for plotting
def extract_line_coordinates(
    df, geometry_col="geometry", source_crs=None, target_crs="EPSG:4326", filename=None
):
    """
    Extract line coordinates from a DataFrame with a geometry column, optionally transforming coordinates.

    Parameters:
    - df: DataFrame with geometry column containing LineString objects
    - geometry_col: name of the geometry column (default: 'geometry')
    - source_crs: The source CRS of the geometries (e.g., 'EPSG:4326' for WGS84)
    - target_crs: The target CRS (default: 'EPSG:4326' for WGS84)

    Returns:
    - line_coordinates: list of numpy arrays containing line coordinates
    - valid_indices: list of indices of valid LineStrings
    """

    logger.info("Extracting line coordinates...")
    line_coordinates = []
    valid_indices = []

    # Ensure df is a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=geometry_col)

    # Ensure the GeoDataFrame has a CRS
    if df.crs is None:
        if source_crs is None:
            raise ValueError("GeoDataFrame has no CRS and source_crs is not provided")
        df = df.set_crs(source_crs, allow_override=True)

    # Transform to target CRS if needed
    if df.crs.to_string() != target_crs:
        df = df.to_crs(target_crs)

    for idx, geometry in enumerate(df[geometry_col]):
        if isinstance(geometry, LineString):
            coords = np.array(geometry.coords)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                line_coordinates.append(coords[:, :2])
                valid_indices.append(idx)
            else:
                logger.error(
                    f"Skipping linestring at index {idx} with unexpected shape: {coords.shape}"
                )
        else:
            logger.error(f"Invalid LineString at index {idx}: {geometry}")

    logger.info("Line coordinates extracted.")

    with open(filename, "wb") as f:
        pickle.dump((line_coordinates, valid_indices), f)
        logger.info("Line coordinates saved to file.")

    return line_coordinates, valid_indices


line_coords_file = data_loc / "line_coords.pkl"
source_crs = "EPSG:4326"
if not os.path.exists(line_coords_file):
    line_coordinates, valid_indices = extract_line_coordinates(
        df_lines, filename=line_coords_file
    )

# Append df lines with potential differrences
df_lines = calculate_GIC(df_lines, V_nodal_halloween, "V_halloween")
df_lines = calculate_GIC(df_lines, V_nodal_st_patricks, "V_st_patricks")
df_lines = calculate_GIC(df_lines, V_nodal_gannon, "V_gannon")
df_lines = calculate_GIC(df_lines, V_nodal_100, "V_100")
df_lines = calculate_GIC(df_lines, V_nodal_500, "V_500")
df_lines = calculate_GIC(df_lines, V_nodal_1000, "V_1000")

# Pickle df_lines, mt_coords, and mt_names
with open(data_loc / "final_tl_data.pkl", "wb") as f:
    pickle.dump((df_lines, mt_coords, mt_names), f)

# %%
