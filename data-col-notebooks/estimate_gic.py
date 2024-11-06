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
from datetime import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve, cholesky
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

# Import custom functions
from build_admittance_matrix import process_substation_buses, random_admittance_matrix


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
data_loc = Path.cwd() / "data"
# Load the data from storm maxes
with h5py.File(data_loc / "geomagnetic_data_2.h5", "r") as f:

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

# %%
# Function to find the substation name for a given bus
def find_substation_name(bus, sub_ref):
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name

    # If not found, return None
    return None


def load_and_process_gic_data(data_loc, df_lines):
    """
    Load and process geomagnetically induced current (GIC) data.

    Parameters
    ----------
    data_loc : Path
        Path to the data directory.
    consider_real_transformer_data : bool, optional
        Whether to use real transformer data (default is False).

    Returns
    -------
    dict
        A dictionary containing processed data:
        - Y_total: Total admittance matrix
        - df_lines: DataFrame with transmission line data
        - df_substations_info: DataFrame with substation information
        - df_transformers: DataFrame with transformer data
        - transformer_counts_dict: Dictionary of transformer counts
        - sub_look_up: Dictionary for substation lookup
        - sub_ref: Dictionary for quick substation reference
    """

    logger.info("Loading and processing GIC data...")

    # Load the data from storm maxes
    with h5py.File(data_loc / "geomagnetic_data_2.h5", "r") as f:

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

    logger.info("GIC data loaded and processed successfully.")

    return (
        df_lines,
        mt_coords,
        mt_names,
        e_field_50,
        e_field_100,
        e_field_500,
        e_field_1000,
        gannon_e,
    )


# %%
# Calculate the injection currents for the network
def calculate_injection_currents(df, n_nodes, col, non_zero_indices, sub_look_up):
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
            # logger.error(f"NaN current encountered for line {line['name']}")
            continue

        # Get i and j
        j, i = sub_look_up.get(line["to_bus"]), sub_look_up.get(line["from_bus"])

        # Currents into and out of the nodes
        injection_currents[i] -= I_eff
        injection_currents[j] += I_eff

    # Eliminate the zero rows and columns
    injection_currents = injection_currents[non_zero_indices]

    return injection_currents


# Solve for nodal voltages
# Using sparse to solve for np gics
def solve_eqn(Y, injection_currents):
    # Solve for nodal voltages
    # Due to size of the matrix add regularization term
    # Skews the results slightly but is fairly faste
    logger.info("Solving for nodal voltages...")
    Y_reg = Y.T @ Y + 1e-20 * eye(Y.shape[1])
    V_nodal = spsolve(Y_reg, Y.T @ injection_currents)
    logger.info("Nodal voltages solved successfully.")
    # Return nodal voltages
    return V_nodal


# Using lPm to solve for nodal voltages
def get_and_solve_cholesky(Y, Je):
    regularization = 1e-6
    Y_reg = Y + np.eye(Y.shape[0]) * regularization
    # Cholesky decomposition of Y_n
    L = cholesky(Y_reg, lower=True)

    # Step 1: Solve L * P = Je
    P = solve(L, Je)

    # Step 2: Solve L^T * V_n = P to get nodal voltages V_n
    V_n = solve(L.T, P)

    return V_n


# %%
# Calculate the GIC for the transmission lines
def calculate_GIC(df, V_nodal, col, non_zero_indices, n_nodes):
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

    V_all = np.zeros(n_nodes)
    V_all[non_zero_indices] = V_nodal

    bus_n = df["from_bus"].values
    bus_k = df["to_bus"].values

    y_nk = 1 / df["R"].values

    j_nk = (df[col].values) * y_nk

    # Get the nodal voltages
    vn = V_all[bus_n]
    vk = V_all[bus_k]

    # Solving for transmission lines GIC
    i_nk = np.round(j_nk + (vn - vk) * y_nk, 2)

    df[f"{col.split('_')[1]}_i_nk"] = i_nk

    logger.info(f"GIC calculation for {col} completed.")

    return df


# %%
def calc_trafo_gic(
    sub_look_up, df_transformers, V_nodal, sub_ref, n_nodes, non_zero_indices, title=""
):

    logger.info(f"Calculating GIC for transformers {title}...")
    gic = {}

    V_all = np.zeros(n_nodes)
    V_all[non_zero_indices] = V_nodal

    # Process transformers and build admittance matrix
    for bus, bus_idx in sub_look_up.items():
        sub = find_substation_name(bus, sub_ref)

        # Filter transformers for current bus
        trafos = df_transformers[df_transformers["bus1"] == bus]

        if len(trafos) == 0 or sub == "Substation 7":
            continue

        # Process each transformer
        for _, trafo in trafos.iterrows():
            # Extract parameters
            bus1 = trafo["bus1"]
            bus2 = trafo["bus2"]
            neutral_point = trafo["sub"]  # Neutral point node (for auto-transformers)
            W1 = trafo["W1"]  # Impedance for Winding 1 (Primary, Series)
            W2 = trafo["W2"]  # Impedance for Winding 2 (Secondary, if available)

            trafo_name = trafo["name"]
            trafo_type = trafo["type"]
            bus1_idx = sub_look_up[bus1]
            neutral_idx = sub_look_up.get(neutral_point)
            bus2_idx = sub_look_up[bus2]

            if trafo_type == "GSU" or trafo_type == "GSU w/ GIC BD":
                Y_w1 = 1 / W1  # Primary winding admittance
                i_k = (V_all[bus1_idx] - V_all[neutral_idx]) * Y_w1
                gic[trafo_name] = {"HV": i_k}

            elif trafo_type == "Tee":
                # Commented out code, consider removing if not needed
                continue

            elif trafo_type == "Auto":
                Y_series = 1 / W1
                Y_common = 1 / W2
                I_s = (V_all[bus1_idx] - V_all[bus2_idx]) * Y_series
                I_c = (V_all[bus2_idx] - V_all[neutral_idx]) * Y_common
                gic[trafo_name] = {"Series": I_s, "Common": I_c}

            elif trafo_type in ["GY-GY-D", "GY-GY"]:
                Y_primary = 1 / W1
                Y_secondary = 1 / W2
                I_w1 = (V_all[bus1_idx] - V_all[neutral_idx]) * Y_primary
                I_w2 = (V_all[bus2_idx] - V_all[neutral_idx]) * Y_secondary
                gic[trafo_name] = {"HV": I_w1, "LV": I_w2}

    logger.info(f"GIC calculation for transformers {title} completed.")
    return gic


# %%
# Solve for the currents to the ground through the neutral points
def solve_total_nodal_gic(Y_e, Vn, title=""):

    logger.info(f"Solving for total NP GIC {title}...")

    # Current flowing to the ground due to eastward/northward geolectric field
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_e = 1 / Y_e
        Z_e[np.isinf(Z_e)] = 0  # Replace inf with 0

    i_g = solve_eqn(Z_e, Vn)  # Per phase
    return i_g * 3  # Total


# %%
# PLot gridded countour map
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
    filename="grid.pkl",
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


# Get injections data
def get_injection_currents(df_lines, n_nodes, non_zero_indices, sub_look_up, data_loc):

    # # Path to the pickled file
    # pkl_file = data_loc / "injections.pkl"

    # # Check if pickle file exists
    # if os.path.exists(pkl_file):
    #     with open(pkl_file, "rb") as f:
    #         injections_data = pickle.load(f)
    # else:
    # Calculate injections if pickle does not exist
    injections_data = {
        "halloween": calculate_injection_currents(
            df_lines, n_nodes, "V_halloween", non_zero_indices, sub_look_up
        ),
        "st_patricks": calculate_injection_currents(
            df_lines, n_nodes, "V_st_patricks", non_zero_indices, sub_look_up
        ),
        "gannon": calculate_injection_currents(
            df_lines, n_nodes, "V_gannon", non_zero_indices, sub_look_up
        ),
        "V_50": calculate_injection_currents(
            df_lines, n_nodes, "V_50", non_zero_indices, sub_look_up
        ),
        "V_100": calculate_injection_currents(
            df_lines, n_nodes, "V_100", non_zero_indices, sub_look_up
        ),
        "V_500": calculate_injection_currents(
            df_lines, n_nodes, "V_500", non_zero_indices, sub_look_up
        ),
        "V_1000": calculate_injection_currents(
            df_lines, n_nodes, "V_1000", non_zero_indices, sub_look_up
        ),
    }

    # # Save to pickle file
    # with open(pkl_file, "wb") as f:
    #     pickle.dump(injections_data, f)

    return injections_data


def calculate_GIC_multiple(
    df_lines_copy, sub_look_up, V_nodals, non_zero_indices, n_nodes
):
    df_list = []
    scenarios = ["Gannon", "V_100", "V_500", "V_1000"]

    for V_nodal, scenario in zip(V_nodals, scenarios):
        df = calculate_GIC(
            df_lines_copy.copy(), V_nodal, scenario, non_zero_indices, n_nodes
        )
        df_list.append(df)

    return pd.concat(df_list, axis=1).loc[
        :, ~pd.concat(df_list, axis=1).columns.duplicated()
    ]


def get_and_solve_cholesky_wrapper(args):
    Y_total, injections = args
    return get_and_solve_cholesky(Y_total, injections)

def parallel_nodal_voltage_calculation(Y_total, injections_data):
    """
    Perform parallel nodal voltage calculations for multiple injection datasets.
    
    Parameters:
    - Y_total: The admittance matrix (Y matrix) for the system.
    - injections_data: Dictionary containing various injection data.
    
    Returns:
    - results: Dictionary of calculated voltages for each dataset.
    """
    logger.info("Preparing tasks for parallel nodal voltage calculation...")
    
    # List of tasks for parallel execution (filter out None tasks)
    tasks = [(name, injections_data.get(name)) for name in ["gannon", "V_100", "V_500", "V_1000"]]
    tasks = [(name, data) for name, data in tasks if data is not None]
    
    if not tasks:
        logger.error("No valid injection data provided.")
        return {}

    # Prepare the data for multiprocessing
    task_data = [(Y_total, t[1]) for t in tasks]
    
    logger.info("Solving for nodal voltages...")
    
    # Use Pool for parallel computation
    with Pool(processes=min(cpu_count(), len(tasks))) as pool:
        try:
            # Solve the system for each injection dataset in parallel
            results = {name: result for name, result in zip([t[0] for t in tasks],
                                                            pool.map(get_and_solve_cholesky_wrapper, task_data))}
            logger.info("Nodal voltages solved successfully.")
            return results
        except Exception as e:
            logger.error(f"Error during parallel execution: {str(e)}")
            return {}

def parallel_gic_calculation_and_processing(
    Y_e, nodal_voltages, non_zero_indices, n_nodes, data_loc, filename
):
    # Create a partial function with Y_e fixed
    solve_func = partial(solve_total_nodal_gic, Y_e)

    # List of tasks to be executed
    tasks = [
        ("Gannon", nodal_voltages["gannon"], "Gannon"),
        ("100-year-hazard", nodal_voltages["V_100"], "100-year-hazard"),
        ("500-year-hazard", nodal_voltages["V_500"], "500-year-hazard"),
        ("1000-year-hazard", nodal_voltages["V_1000"], "1000-year-hazard"),
    ]

    # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(solve_func, V_nodal, scenario): name
            for name, V_nodal, scenario in tasks
        }

        # Collect results as they complete
        results = {}
        for future in as_completed(future_to_task):
            name = future_to_task[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"{name} generated an exception: {exc}")

    # Process the results
    data = np.stack([results[name] for name, _, _ in tasks], axis=1)

    data_zeros = np.zeros((n_nodes, 4))
    data_zeros[non_zero_indices, :] = data
    df_ig = pd.DataFrame(data_zeros, columns=[name for name, _, _ in tasks])

    # Save the dataframe
    df_ig.to_csv(data_loc / f"np_gic_rand_{filename}.csv", index=False)

    return df_ig


def main(generate_grid=False):

    # Load the data
    # Data loc
    data_loc = Path.cwd() / "data"

    # Get substation buses data
    substation_buses, bus_ids_map, sub_look_up, df_lines, df_substations_info = (
        process_substation_buses(data_loc)
    )

    # Load and process transmission line data
    df_lines.drop(columns=["geometry"], inplace=True)
    df_lines["name"] = df_lines["name"].astype(np.int32)

    transmission_line_path = (
        data_loc / "Electric__Power_Transmission_Lines" / "trans_lines_pickle.pkl"
    )
    with open(transmission_line_path, "rb") as p:
        trans_lines_gdf = pickle.load(p)

    trans_lines_gdf["line_id"] = trans_lines_gdf["line_id"].astype(np.int32)
    df_lines = df_lines.merge(
        trans_lines_gdf[["line_id", "geometry"]], right_on="line_id", left_on="name"
    )

    # Create a dictionary for quick substation lookup
    sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))

    # Load and process GIC data
    (
        df_lines,
        mt_coords,
        mt_names,
        e_field_50,
        e_field_100,
        e_field_500,
        e_field_1000,
        gannon_e,
    ) = load_and_process_gic_data(data_loc, df_lines)

    def filename_generator():
        # Uses dateto generate a unique filename
        current_time = datetime.now()
        filename_timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        filename_timestamp

        return filename_timestamp

    n_nodes = len(sub_look_up)  # Number of nodes in the network

    # Get 1000 dfs of winding GICs and np gics
    for i in range(0, 500):
        logger.info(f"Processing iteration {i}...")
        # Generate a random admittance matrix
        print("Script started")

        Y_n, Y_e, df_transformers = random_admittance_matrix(
            substation_buses, bus_ids_map, sub_look_up, df_lines, df_substations_info
        )
        
        # Find indices of rows/columns where all elements are zero in the admittance mat
        zero_row_indices = np.where(np.all(Y_n == 0, axis=1))[0]  # Zero rows
        zero_col_indices = np.where(np.all(Y_n == 0, axis=0))[0]  # Zero columns

        # Get the non-zero row/col indices
        non_zero_indices = np.setdiff1d(np.arange(Y_n.shape[0]), zero_row_indices)

        # Reduce the Y_n and Y_e matrices
        Y_n = Y_n[np.ix_(non_zero_indices, non_zero_indices)]
        Y_e = Y_e[np.ix_(non_zero_indices, non_zero_indices)]

        # Y total is summ of earthing and network impedances
        Y_total = Y_n + Y_e

        # Get the injections data
        injections_data = get_injection_currents(
            df_lines, n_nodes, non_zero_indices, sub_look_up, data_loc
        )

        # Solve for nodal voltages
        # V_nodal_halloween = get_and_solve_cholesky(Y_total, injections_data["halloween"])
        # V_nodal_st_patricks = get_and_solve_cholesky(Y_total, injections_data["st_patricks"])
        # V_nodal_50 = get_and_solve_cholesky(Y_total, injections_data["V_50"])

        # V_nodal_gannon = get_and_solve_cholesky(Y_total, injections_data["gannon"])
        # V_nodal_100 = get_and_solve_cholesky(Y_total, injections_data["V_100"])
        # V_nodal_500 = get_and_solve_cholesky(Y_total, injections_data["V_500"])
        # V_nodal_1000 = get_and_solve_cholesky(Y_total, injections_data["V_1000"])
        
        # Parallelize nodal voltage calculation
        nodal_voltages = parallel_nodal_voltage_calculation(Y_total, injections_data)
        V_nodal_gannon = nodal_voltages["gannon"]
        V_nodal_100 = nodal_voltages["V_100"]
        V_nodal_500 = nodal_voltages["V_500"]
        V_nodal_1000 = nodal_voltages["V_1000"]

        df_lines_copy = df_lines.copy()

        # Get the nodal voltages bus ids
        df_lines_copy["from_bus"] = df_lines_copy["from_bus"].apply(
            lambda x: sub_look_up.get(x)
        )
        df_lines_copy["to_bus"] = df_lines_copy["to_bus"].apply(
            lambda x: sub_look_up.get(x)
        )

        # # Append df lines with potential differrences
        # df = calculate_GIC(
        #     df_lines_copy, V_nodal_halloween, "V_halloween", non_zero_indices, n_nodes
        # )
        # df = calculate_GIC(
        #     df_lines_copy, V_nodal_st_patricks, "V_st_patricks", non_zero_indices, n_nodes
        # )
        df = calculate_GIC(
            df_lines_copy, V_nodal_gannon, "V_gannon", non_zero_indices, n_nodes
        )
        df = calculate_GIC(
            df_lines_copy, V_nodal_100, "V_100", non_zero_indices, n_nodes
        )
        df = calculate_GIC(
            df_lines_copy, V_nodal_500, "V_500", non_zero_indices, n_nodes
        )
        df = calculate_GIC(
            df_lines_copy, V_nodal_1000, "V_1000", non_zero_indices, n_nodes
        )

        # # Pickle df_lines, mt_coords, and mt_names
        # with open(data_loc / "final_tl_data.pkl", "wb") as f:
        #     pickle.dump((df, mt_coords, mt_names), f)

        # Estimate the GICs for the transformers
        df_transformers_copy = df_transformers.copy()

        # GIC
        # gic_halloween = calc_trafo_gic(
        #     sub_look_up,
        #     df_transformers_copy,
        #     V_nodal_halloween,
        #     sub_ref,
        #     n_nodes,
        #     non_zero_indices,
        #     "Halloween",
        # )
        # gic_st_patricks = calc_trafo_gic(
        #     sub_look_up,
        #     df_transformers_copy,
        #     V_nodal_st_patricks,
        #     sub_ref,
        #     n_nodes,
        #     non_zero_indices,
        #     "St. Patricks",
        # )

        gic_gannon = calc_trafo_gic(
            sub_look_up,
            df_transformers_copy,
            V_nodal_gannon,
            sub_ref,
            n_nodes,
            non_zero_indices,
            "Gannon",
        )
        gic_100 = calc_trafo_gic(
            sub_look_up,
            df_transformers_copy,
            V_nodal_100,
            sub_ref,
            n_nodes,
            non_zero_indices,
            "100-year-hazard",
        )
        gic_500 = calc_trafo_gic(
            sub_look_up,
            df_transformers_copy,
            V_nodal_500,
            sub_ref,
            n_nodes,
            non_zero_indices,
            "500-year-hazard",
        )
        gic_1000 = calc_trafo_gic(
            sub_look_up,
            df_transformers_copy,
            V_nodal_1000,
            sub_ref,
            n_nodes,
            non_zero_indices,
            "1000-year-hazard",
        )

        # exploded dicts
        # hash_gic_halloween = [
        #     (trafo, winding, gic)
        #     for trafo, windings in gic_halloween.items()
        #     for winding, gic in windings.items()
        # ]
        # hash_gic_st_patricks = [
        #     (trafo, winding, gic)
        #     for trafo, windings in gic_st_patricks.items()
        #     for winding, gic in windings.items()
        # ]

        hash_gic_gannon = [
            (trafo, winding, gic)
            for trafo, windings in gic_gannon.items()
            for winding, gic in windings.items()
        ]
        hash_gic_100 = [
            (trafo, winding, gic)
            for trafo, windings in gic_100.items()
            for winding, gic in windings.items()
        ]
        hash_gic_500 = [
            (trafo, winding, gic)
            for trafo, windings in gic_500.items()
            for winding, gic in windings.items()
        ]
        hash_gic_1000 = [
            (trafo, winding, gic)
            for trafo, windings in gic_1000.items()
            for winding, gic in windings.items()
        ]

        # # Make as df
        # winding_halloween_df = pd.DataFrame(
        #     hash_gic_halloween, columns=["Transformer", "Winding", "Hallloween A/ph"]
        # )
        # winding_st_patricks_df = pd.DataFrame(
        #     hash_gic_st_patricks, columns=["Transformer", "Winding", "St. Patricks A/ph"]
        # )
        winding_gannon_df = pd.DataFrame(
            hash_gic_gannon, columns=["Transformer", "Winding", "Gannon A/ph"]
        )
        winding_100_df = pd.DataFrame(
            hash_gic_100, columns=["Transformer", "Winding", "100-year-hazard A/ph"]
        )
        winding_500_df = pd.DataFrame(
            hash_gic_500, columns=["Transformer", "Winding", "500-year-hazard A/ph"]
        )
        winding_1000_df = pd.DataFrame(
            hash_gic_1000, columns=["Transformer", "Winding", "1000-year-hazard A/ph"]
        )

        # Merge all into a single df for export
        winding_gic_df = pd.concat(
            [
                # winding_halloween_df,
                # winding_st_patricks_df,
                winding_gannon_df,
                winding_100_df,
                winding_500_df,
                winding_1000_df,
            ],
            axis=1,
        )

        # Remove duplicate columns
        winding_gic_df = winding_gic_df.loc[:, ~winding_gic_df.columns.duplicated()]

        df_transformers["Transformer"] = df_transformers["name"]
        winding_gic_df = winding_gic_df.merge(
            df_transformers[["sub_id", "Transformer", "latitude", "longitude"]],
            on="Transformer",
            how="inner",
        )

        filename = filename_generator()

        # Save the tf winding gic df
        winding_gic_df.to_csv(
            data_loc / f"winding_gic_rand_{filename}.csv", index=False
        )

        # ig_halloween = solve_total_nodal_gic(Y_e, V_nodal_halloween, title="Halloween")
        # ig_st_patricks = solve_total_nodal_gic(Y_e, V_nodal_st_patricks, "St. Patricks")

        # ig_gannon = solve_total_nodal_gic(Y_e, V_nodal_gannon, "Gannon")
        # ig_100 = solve_total_nodal_gic(Y_e, V_nodal_100, "100-year-hazard")
        # ig_500 = solve_total_nodal_gic(Y_e, V_nodal_500, "500-year-hazard")
        # ig_1000 = solve_total_nodal_gic(Y_e, V_nodal_1000, "1000-year-hazard")

        # # %%
        # # Save as a dataframe - stack the data
        # data = np.stack(
        #     [ig_gannon, ig_100, ig_500, ig_1000], axis=1
        # )

        # data_zeros = np.zeros((n_nodes, 4))
        # data_zeros[non_zero_indices, :] = data
        # df_ig = pd.DataFrame(
        #     data_zeros,
        #     columns=[
        #         # "Halloween",
        #         "Gannon",
        #         # "St. Patricks",
        #         "100-year-hazard",
        #         "500-year-hazard",
        #         "1000-year-hazard",
        #     ],
        # )

        # # Save the df
        # # df_ig.to_csv(data_loc / f"np_gic_rand_{i}.csv", index=False)
        # df_ig = parallel_gic_calculation_and_processing(
        #     Y_e, nodal_voltages, non_zero_indices, n_nodes, data_loc, filename
        # )

        # Prepare the grid and mask for plotting
        if generate_grid:
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
                        e_field,
                        mt_coords,
                        resolution=(500, 1000),
                        filename=grid_filename,
                    )

            logging.info("Grid and mask generated and saved.")

        line_coords_file = data_loc / "line_coords.pkl"
        source_crs = "EPSG:4326"
        if not os.path.exists(line_coords_file):
            line_coordinates, valid_indices = extract_line_coordinates(
                df_lines, filename=line_coords_file
            )

if __name__ == "__main__":
    main(generate_grid=False)

# %%
