# ...............................................................
# Description: This script is used to build the US EHV admittance matrix
# Dependecy scripts: Requires data from data_preprocessing.ipynb notebook
# Output: Admittance matrix
# Author: Dennies Bor
# ...............................................................

# %%
import pickle
import sys
import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve


# Set up logging, a reusable function
def setup_logging():
    """
    Set up logging to both file and console.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler("admittance_matrix.log")
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

# Data path
data_loc = Path(__file__).resolve().parent / "data"

# %%
# Unpickle substations pairs and EHVs
unique_sub_voltage_pairs = pickle.load(
    open(data_loc / "unique_sub_voltage_pairs.pkl", "rb")
)

# Load filtered EHV transmission lines > 200 kv
df_lines_EHV = pickle.load(open(data_loc / "df_lines_EHV.pkl", "rb"))

# Load the cleaned tranmission lines with integrable geometries. if uniform geolectric field skip
transmission_line_path = (
    data_loc / "Electric__Power_Transmission_Lines" / "trans_lines_pickle.pkl"
)
with open(transmission_line_path, "rb") as p:
    trans_lines_gdf = pickle.load(p)

# Filter on name in trans_lines_gdf
df_lines_EHV = df_lines_EHV[df_lines_EHV["line_id"].isin(trans_lines_gdf.line_id)]
# Load the transmission lines within FERC regions
substation_to_line_voltages = pickle.load(
    open(data_loc / "substation_to_line_voltages.pkl", "rb")
)
ss_df = pickle.load(open(data_loc / "ss_df.pkl", "rb"))
trans_lines_within_FERC_filtered_ = pickle.load(
    open(data_loc / "trans_lines_within_FERC_filtered.pkl", "rb")
)

# Create a dictionary of SS_ID with type of substation
ss_type_dict = dict(zip(ss_df["SS_ID"], ss_df["SS_TYPE"]))


# %%
def build_substation_buses(unique_sub_voltage_pairs, df_lines_EHV):
    """
    Build substation buses based on unique substation-voltage pairs and a dataframe of EHV lines.
    Parameters:
    unique_sub_voltage_pairs (DataFrame): A DataFrame containing unique substation-voltage pairs.
    df_lines_EHV (DataFrame): A DataFrame containing EHV lines.
    Returns:
    tuple: A tuple containing two elements:
        - substation_buses (dict): A dictionary mapping substation IDs to substation information.
        - transformers_data (list): A list of transformer data.
    Raises:
    None
    Notes:
    - This function approximates substation buses based on spatially intersected transmission lines and substations.
    - The function estimates injection currents for the designed low voltage and high voltage buses.
    - The function makes assumptions about transformer characteristics based on the voltage ratings of the connected lines.
    - The function assigns transformer types such as "GY-D", "GY-GY", "GY-GY-D", "Auto", or "Unknown" based on the number and ratios of voltage ratings.
    - The function identifies external buses connected to the low voltage bus
    """

    logging.info("Building substation buses...")
    # Approximate substation buses
    substation_buses = {}

    # Tranformer generator number
    transformer_gen_num = 0
    transformers_data = []
    count = 0
    # .........................................................................
    # Design the low voltage and high voltage buses using spatially intersected tls and substations
    # This approach allows for the estimation of the injection currents
    # .........................................................................
    for i, row in unique_sub_voltage_pairs.iterrows():
        substation = row.substation
        line_voltage = row.voltage
        ss_type = ss_type_dict[substation]

        if substation not in substation_buses:

            # We will focus now on distribution and transmission substations
            if ss_type not in ["Distribution", "Transmission"]:
                continue

            # Voltages in substation
            sub_connected_lines = np.array(
                [float(v) for v in substation_to_line_voltages[substation]]
            )
            substation_buses_unique = np.unique(np.sort(sub_connected_lines))[::-1]

            # Max voltage in substation
            max_voltage_substation = np.max(substation_buses_unique)

            # Get the bus in the substation with the highest voltage
            sub_a_maxV_bus_series = unique_sub_voltage_pairs.query(
                "substation == @substation and voltage == @max_voltage_substation"
            )["bus_id"]

            external_bus_to_hv_bus = []
            # If we have multiple lines intersecting in a substation
            if len(sub_a_maxV_bus_series.values) >= 1:
                sub_a_maxV_bus = sub_a_maxV_bus_series.values[0]

                # Find connected to_bus_ids and from_bus_ids
                sub_a_maxV_bus_series_to = df_lines_EHV.query(
                    "from_bus_id == @sub_a_maxV_bus"
                )["to_bus_id"].values
                sub_a_maxV_bus_series_from = df_lines_EHV.query(
                    "to_bus_id == @sub_a_maxV_bus"
                )["from_bus_id"].values

                # Combine and get unique bus ids
                all_connected_buses = np.unique(
                    np.concatenate(
                        (sub_a_maxV_bus_series_to, sub_a_maxV_bus_series_from)
                    )
                )
                external_bus_to_hv_bus = list(all_connected_buses)

            # If the max bus is not available in the filtered substation or not connected to any other bus
            else:
                sub_a_maxV_bus = f"{substation}_{int(max_voltage_substation)}"

            # ...............................................
            # Make assumptions of transformer characteristics
            # In some generating stations, two of the generators are connected to one transformer
            # In some generations, power might be exported in two voltage levels
            # ...............................................

            # If the lines have single voltage rating, the gic doesn't flow in the secondary (assign a D-Wye)
            if len(substation_buses_unique) == 1:
                low_voltage_bus = sub_a_maxV_bus + "lv"
                transformer_type = "GY-D"

            else:
                # If lines are multiple rated, could be three windings (Gy-Gy-D), Gy-Gy or Auto transformer if closely rated
                low_voltage_bus = f"{substation}_{int(substation_buses_unique[1])}"

                # If only two unique voltage ratings, check for their ratios
                # Most def an auto if they are close
                if len(substation_buses_unique) == 2:
                    transformer_type = (
                        "GY-GY"
                        if np.max(sub_connected_lines) / np.min(sub_connected_lines) > 2
                        else "Auto"
                    )

                # If three unique voltage ratings, could be a three winding transformer
                elif len(substation_buses_unique) == 3:
                    transformer_type = "GY-GY-D"
                # If multiple voltage ratings, we are interested in HV buses (ignore others)
                elif len(substation_buses_unique) > 3:
                    # Filter those with greater than 200, else assign Gy-D
                    filtered_sub_bus_unique = substation_buses_unique[
                        substation_buses_unique >= 200
                    ]

                    if len(filtered_sub_bus_unique) == 1:
                        transformer_type = "GY-GY"
                    elif len(filtered_sub_bus_unique) == 2:
                        transformer_type = (
                            "GY-GY"
                            if np.max(sub_connected_lines) / np.min(sub_connected_lines)
                            > 2
                            else "Auto"
                        )
                    elif len(filtered_sub_bus_unique) == 3:
                        transformer_type = "GY-GY-D"
                    else:
                        transformer_type = "Unknown"

                else:
                    transformer_type = "Unknown"

            transformer_gen_num += 1
            transformer_number = "T" + str(transformer_gen_num)

            transformer_data = {
                "sub_id": substation,
                "name": transformer_number,
                "type": transformer_type,
                "bus1_id": sub_a_maxV_bus,
                "bus2_id": low_voltage_bus,
            }

            # Substation information
            substation_info = {
                "SS_ID": substation,
                "transformers": [transformer_number],
                "buses": [sub_a_maxV_bus, low_voltage_bus],
                "hv_bus": sub_a_maxV_bus,
                "lv_bus": low_voltage_bus,
                "external_bus_to_hv_bus": external_bus_to_hv_bus,
                "external_bus_to_lv_bus": [],
            }

            # ......................................................
            # If the transformer type not a gsu, GY-D or Tee, get external buses connected to the low voltage bus
            # Interested in buses > 200 v. if no lines, then ignore
            # ......................................................

            # Get other buses in substations except max_buses
            if transformer_type not in ["Gy-D", "Tee", "GY-D"]:
                lv_bus_v = int(substation_buses_unique[1])
                lv_bus_id = f"{substation}_{lv_bus_v}"
                if lv_bus_v >= 200:
                    # Find sub bus connected to_bus_ids and from_bus_ids
                    sub_bus_series_to = df_lines_EHV.query("from_bus_id == @lv_bus_id")[
                        "to_bus_id"
                    ].values
                    sub_bus_series_from = df_lines_EHV.query("to_bus_id == @lv_bus_id")[
                        "from_bus_id"
                    ].values

                    # Combine and get unique bus ids
                    sub_bus_connected_buses = np.unique(
                        np.concatenate((sub_bus_series_to, sub_bus_series_from))
                    )

                    substation_info["external_bus_to_lv_bus"] = list(
                        sub_bus_connected_buses
                    )

            substation_buses[substation] = substation_info
            transformers_data.append(transformer_data)

    logging.info("Substation buses built.")

    return substation_buses, transformers_data


# %%
# Get substation_buses and bus ids
substation_buses, transformers_data = build_substation_buses(
    unique_sub_voltage_pairs, df_lines_EHV
)

# %%
# Create bus ids from buses within the substations
buses = []
for sub_info in substation_buses.values():
    buses.extend(sub_info["buses"])

buses = np.unique(buses)

# Create a dictionary to map buses to integers
bus_ids_map = {bus: i + 1 for i, bus in enumerate(buses)}
print(f"Number of buses: {len(buses)}")

# %%

# Typically substation grounding resistances range from 0.1 -> 0.2 (D.Boteler, et al., 2013)
grounding_resistances = [0.1, 0.2]

# Use from dict to load substation_buses
df_substations_info = pd.DataFrame(substation_buses).T.reset_index()
df_substations_info = df_substations_info[["SS_ID", "transformers", "buses"]]

# Apply bus_id map to buses
df_substations_info["buses"] = df_substations_info["buses"].apply(
    lambda x: [bus_ids_map[bus] for bus in x]
)

# Merge with substations info to get latitude and longitude
df_substations_info = df_substations_info.merge(
    ss_df[["SS_ID", "lat", "lon"]], on="SS_ID", how="left"
)
df_substations_info["grounding_resistance"] = grounding_resistances[1]

# Rename SSID, lat, and lon columns
df_substations_info.rename(
    columns={"SS_ID": "name", "lat": "latitude", "lon": "longitude"}, inplace=True
)

df_substations_info.tail(5)

# %%
transformer_make = ["GY-GY-D", "GY-GY", "Auto", "GSU", "GY-D"]

# ..................................................
# Create a dictionary to store the resistance values
# Only HV distribution and transmission substations are taken into consideration
# We consider a Delta-Wye architecture for substations with a single bus
# Although, a dummy LV bus is introduced where GIC flow is zero
# ..................................................
transformer_winding_resistances = {
    "GY-GY-D": {"pri": 0.2, "sec": 0.1},
    "GY-GY": {"pri": 0.04, "sec": 0.06},
    "Auto": {"pri": 0.04, "sec": 0.06},
    "GSU": {"pri": 0.15, "sec": float("inf")},
    "Tee": {"pri": 0.01, "sec": 0.01},
    "GY-D": {"pri": 0.05, "sec": 0.1},
}

# Create a transformer df
df_transformers = pd.DataFrame(transformers_data)

# Transformer W1 and W2
df_transformers["W1"] = df_transformers["type"].apply(
    lambda x: transformer_winding_resistances[x]["pri"]
)
df_transformers["W2"] = df_transformers["type"].apply(
    lambda x: transformer_winding_resistances[x]["sec"]
)

# Apply bus_id map
df_transformers["bus1"] = df_transformers["bus1_id"].map(bus_ids_map)
df_transformers["bus2"] = df_transformers["bus2_id"].map(bus_ids_map)

# Merge transformers with lat and lon data
df_transformers = df_transformers.merge(
    df_substations_info[["name", "latitude", "longitude"]],
    left_on="sub_id",
    right_on="name",
    how="left",
)

# Rename name_x to name
df_transformers.rename(columns={"name_x": "name"}, inplace=True)

# Drop name_y
df_transformers.drop("name_y", axis=1, inplace=True)

# Useful cols
df_transformers = df_transformers[
    ["sub_id", "name", "type", "bus1", "bus2", "W1", "W2", "latitude", "longitude"]
]

df_transformers.tail(5)


# %%
def flatten_substation_dict(data):
    """
    Flatten the substation dictionary and generate records for building an admittance matrix.
    Parameters:
        data (dict): A dictionary containing substation details.
    Returns:
        list: A list of records for building an admittance matrix.
    """

    # Format lines into a format to build an admittance matrix
    records = []

    logging.info("Flattening substation dictionary...")

    for substation, details in data.items():
        hv_bus = details["hv_bus"]
        lv_bus = details["lv_bus"]

        # Connections from HV bus to external buses
        for ext_bus in details["external_bus_to_hv_bus"]:
            line_connections = df_lines_EHV[
                (df_lines_EHV["from_bus_id"] == hv_bus)
                & (df_lines_EHV["to_bus_id"] == ext_bus)
                | (df_lines_EHV["from_bus_id"] == ext_bus)
                & (df_lines_EHV["to_bus_id"] == hv_bus)
            ][["line_id", "from_bus_id", "to_bus_id"]].values.tolist()
            for line in line_connections:
                line_id = line[0]
                sub1 = line[1]
                sub2 = line[2]

                if sub1 in buses and sub2 in buses:
                    records.append((hv_bus, ext_bus, "line", line_id))

        # Connections from LV buses to external buses
        for ext_bus_2_lv_bus in details["external_bus_to_lv_bus"]:
            line_connections = df_lines_EHV[
                (df_lines_EHV["from_bus_id"] == lv_bus)
                & (df_lines_EHV["to_bus_id"] == ext_bus_2_lv_bus)
                | (df_lines_EHV["from_bus_id"] == ext_bus_2_lv_bus)
                & (df_lines_EHV["to_bus_id"] == lv_bus)
            ][["line_id", "from_bus_id", "to_bus_id"]].values.tolist()

            for line in line_connections:
                line_id = line[0]
                sub1 = line[1]
                sub2 = line[2]

                if sub1 in buses and sub2 in buses:
                    records.append((lv_bus, ext_bus_2_lv_bus, "line", line_id))
    return records


# Flatten the dictionary
flattened_data = flatten_substation_dict(substation_buses)

# Create DataFrame
df_lines_final = pd.DataFrame(
    flattened_data, columns=["from_bus_id", "to_bus_id", "utility", "name"]
)

# %%
df_lines_EHV[~(df_lines_EHV.VOLTAGE.isin([345, 230, 765, 500]))].shape
# %%
# Approximate line resistances
line_resistance = {
    765: 0.01,  # Ω/km (converted from 0.0227 Ω/mi)
    500: 0.0141,  # Ω/km (converted from 0.0227 Ω/mi)
    345: 0.0283,  # Ω/km (converted from 0.0455 Ω/mi)
    232: 0.0450,  # Ω/km (estimated)
    230: 0.0500,  # Ω/km (estimated)
    220: 0.0700,  # Ω/km (estimated)
    138: 0.0800,  # Ω/km (estimated)
    69: 0.1200,  # Ω/km (estimated)
}


def calculate_line_resistances(df, df_ehv, line_resistance):
    """
    Calculate line resistances for each line in the dataframe.
    Parameters:
    - df (pandas.DataFrame): The input dataframe containing line information.
    - df_ehv (pandas.DataFrame): The input dataframe containing EHV line information.
    - line_resistance (dict): A dictionary mapping voltage levels to line resistances.
    Returns:
    - df (pandas.DataFrame): The modified dataframe with calculated line resistances.
    Raises:
    - None
    Example:
    df = pd.DataFrame(...)
    df_ehv = pd.DataFrame(...)
    line_resistance = {110: 0.1, 220: 0.2, 400: 0.3}
    result = calculate_line_resistances(df, df_ehv, line_resistance)
    """
    # Remove duplicates
    df.drop_duplicates(subset=["name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Map substation IDs to unique numbers
    df["from_bus"] = df["from_bus_id"].map(bus_ids_map)
    df["to_bus"] = df["to_bus_id"].map(bus_ids_map)

    # Merge with the lines to get the line length and Votage rating
    df = df.merge(
        df_ehv[["line_id", "length", "VOLTAGE"]],
        left_on="name",
        right_on="line_id",
        how="left",
    )

    # Merge df_lines_unique with transmission lines to get geometries for plotting
    df = df.merge(
        trans_lines_within_FERC_filtered_[["line_id", "geometry"]],
        left_on="name",
        right_on="line_id",
        how="left",
    )

    # Get the useful buses
    df = df[["name", "from_bus", "to_bus", "length", "VOLTAGE", "geometry"]]

    # Rename Voltage to V
    df.rename(columns={"VOLTAGE": "V"}, inplace=True)

    # Increase length by 3%
    df["length"] = df["length"] * 1.03

    # Apply line resistance to get R per km
    df["R_per_km"] = df["V"].map(line_resistance)

    # Get R for the entire length
    df["R"] = df["length"] * df["R_per_km"]

    df.drop_duplicates(subset=["name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# %%
# --------------------------------------------------------------
# Take the real datasets of transformer count from gridd mapping team
# For now we shall consider the transformers as connected in parallel to the LV and HV buses
# Additionally, the transformer units are identical, even though they may have different ratings and characteristics
# --------------------------------------------------------------

grid_mapping_loc = data_loc / "grid_mapping.csv"


# Read the file
grid_mapping = pd.read_csv(grid_mapping_loc)

# Eval attributes column
grid_mapping["Attributes"] = grid_mapping["Attributes"].apply(eval)
transformers_df = grid_mapping[grid_mapping["Marker Label"] == "Transformer"]
transformers_df["Role"] = transformers_df["Attributes"].apply(lambda x: x["role"])
transformers_df["type"] = transformers_df["Attributes"].apply(lambda x: x["type"])
transformer_counts = (
    transformers_df.groupby("SS_ID").size().reset_index(name="transformer_count")
)
# Zip the transformer counts to a dictionary
transformer_counts_dict = dict(
    zip(transformer_counts["SS_ID"], transformer_counts["transformer_count"])
)

# Pickle this for use in the final data analysis

with open(data_loc / "transformer_counts.pkl", "wb") as f:
    pickle.dump(transformer_counts_dict, f)

# Get unique ss_id and Role
ss_role = transformers_df[["SS_ID", "Role"]].drop_duplicates()

# Zip ss_id and role
ss_role_dict = dict(zip(ss_role["SS_ID"], ss_role["Role"]))


# %%
# ......................................................................
# Build Admittance Matrix
# ......................................................................

# Get the number of unique nodes
n_nodes = len(buses)

# Create a dictionary for quick substation lookup
sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))

# Initialize the admittance matrix
Y = np.zeros((n_nodes, n_nodes))


def adjust_index(bus, test=False):
    if test:
        return bus - 1 if bus < 9 else bus - 3
    else:
        return bus - 1


def safe_inverse(value, min_denominator=1e-10):
    if pd.isnull(value) or abs(value) < min_denominator:
        return 0
    return 1 / value


def add_admittance(Y, from_bus, to_bus, admittance):
    i, j = adjust_index(from_bus), adjust_index(to_bus)
    Y[i, i] += admittance
    if i != j:
        Y[j, j] += admittance
        Y[i, j] -= admittance
        Y[j, i] -= admittance


def add_admittance_auto(Y, from_bus, to_bus, Y_series, Y_common):
    i, j = adjust_index(to_bus), adjust_index(from_bus)
    Y[i, i] += Y_common
    Y[i, i] += Y_series
    Y[j, j] += Y_series
    Y[i, j] -= Y_series
    Y[j, i] -= Y_series


def add_tee_network(Y, bus1, bus2, R1, R2):
    i, j = adjust_index(bus1), adjust_index(bus2)
    Y1, Y2 = safe_inverse(R1), safe_inverse(R2)

    # Add admittances for the T-network
    Y[i, i] += Y1
    Y[j, j] += Y2
    Y[i, j] -= Y1
    Y[j, i] -= Y1


def safe_add_admittance(Y, from_bus, to_bus, admittance):
    if pd.notnull(admittance) and pd.notnull(from_bus) and pd.notnull(to_bus):
        add_admittance(Y, from_bus, to_bus, admittance)
    else:
        logging.error(
            f"Could not add admittance for buses {from_bus}, {to_bus} with value {admittance}"
        )


def find_substation_name(bus, sub_ref):
    for sub_name, buses in sub_ref.items():
        if bus in buses:
            return sub_name

    # If not found, return None
    print(f"Bus {bus} not found in any substation.")
    return None


# Consider real transformer data
consider_real_transformer_data = True


# Add admittance matrix
def build_admittance_matrix(df_tr, df_lines, sub_ref):
    """
    Build the admittance matrix based on the given transformer and line dataframes.
    Parameters:
        df_tr (pandas.DataFrame): Dataframe containing transformer information.
        df_lines (pandas.DataFrame): Dataframe containing transmission line information.
        sub_ref (str): Reference for the substation.
    Returns:
        numpy.ndarray: The admittance matrix.
    Raises:
        UserWarning: If no substation is found for a transformer.
        UserWarning: If no grounding resistance is found for a substation.
        UserWarning: If invalid W1 or W2 values are found for a Tee network transformer.
        UserWarning: If invalid Y_series or Y_common values are found for an Auto transformer.
        UserWarning: If an unknown transformer type is encountered.
    """

    logging.info("Building admittance matrix...")

    # Initialize the admittance matrix
    Y = np.zeros((n_nodes, n_nodes))

    # Add transformer admittances
    for i, row in df_transformers.iterrows():
        sub = find_substation_name(row["bus1"], sub_ref)

        if consider_real_transformer_data:
            # Get the tr count
            tr_count = transformer_counts_dict.get(row["sub_id"], 1)

        else:
            tr_count = 1

        if sub is None:
            logging.error(
                f"No substation found for bus {row['bus1']} in transformer {row['name']}"
            )
            continue

        # Get the grounding resistance of the substations
        Rg = df_substations_info[df_substations_info.name == sub][
            "grounding_resistance"
        ]
        if Rg.empty:
            logging.error(f"No grounding resistance found for substation {sub}")
            Rg = 0
        else:
            Rg = Rg.iloc[0]

        # -------------------------------------------------------------
        # Add admittances for the GSU or GY-D transformer
        # GIC flows in the primary winding only
        # Calculated admittance by summing Rg and W1 serial resistances
        # -------------------------------------------------------------
        if row["type"] == "GSU" or row["type"] == "GY-D":
            Y_gsu = (
                safe_inverse(pd.to_numeric(row["W1"], errors="coerce") + Rg) * tr_count
            )  # Assuming all transformers are identical and connected in parallel
            safe_add_admittance(Y, row["bus1"], row["bus1"], Y_gsu)

        # In T network, No GIC is recorded
        elif row["type"] == "Tee":
            W1 = pd.to_numeric(row["W1"], errors="coerce")
            W2 = pd.to_numeric(row["W2"], errors="coerce")
            if pd.notnull(W1) and pd.notnull(W2):
                add_tee_network(Y, row["bus1"], row["bus2"], 1 / W1, 1 / W2)
            else:
                logging.error(
                    f"Invalid W1 or W2 for Tee network in transformer {row['name']}"
                )

        # GSU with bloicking device has np.inf resistances
        elif row["type"] == "GSU w/ GIC BD":
            R_blocking = 1e6
            Y_gsu = (
                safe_inverse(pd.to_numeric(row["W1"], errors="coerce") + R_blocking)
                * tr_count
            )
            safe_add_admittance(Y, row["bus1"], row["bus1"], Y_gsu)

        # Share a common winding, therefore GIC flow in primary and secondary
        elif row["type"] == "Auto":
            Y_series = safe_inverse(pd.to_numeric(row["W1"], errors="coerce"))
            Y_common = safe_inverse(pd.to_numeric(row["W2"], errors="coerce") + Rg)

            # Assuming all transformers are identical and connected in parallel
            Y_series, Y_common = Y_series * tr_count, Y_common * tr_count
            if pd.notnull(Y_series) and pd.notnull(Y_common):
                add_admittance_auto(Y, row["bus1"], row["bus2"], Y_series, Y_common)
            else:
                logging.error(
                    f"Invalid Y_series or Y_common for Auto transformer {row['name']}"
                )

        # -------------------------------------------------------------
        # In three winding, Delta side is ignored
        # Gy-GY share a neutral point and GIC flow in all the windings
        # -------------------------------------------------------------
        elif row["type"] in ["GY-GY-D", "GY-GY"]:
            Y_primary = safe_inverse(pd.to_numeric(row["W1"], errors="coerce") + Rg)
            Y_secondary = safe_inverse(pd.to_numeric(row["W2"], errors="coerce") + Rg)

            # Assuming all transformers are identical and connected in parallel
            Y_primary, Y_secondary = Y_primary * tr_count, Y_secondary * tr_count

            safe_add_admittance(Y, row["bus1"], row["bus1"], Y_primary)
            safe_add_admittance(Y, row["bus2"], row["bus2"], Y_secondary)

        else:
            logging.error(
                f"Unknown transformer type {row['type']} for transformer {row['name']}"
            )

    # Add transmission line admittances
    for i, line in df_lines.iterrows():
        Y_line = 1 / line["R"]

        if np.isnan(Y_line):
            logging.error(f"NaN Y_line encountered for line {line['name']}")
            continue
        add_admittance(Y, line["from_bus"], line["to_bus"], Y_line)

    return Y


df_lines = calculate_line_resistances(df_lines_final, df_lines_EHV, line_resistance)

# Get the admittance matrix
Y = build_admittance_matrix(df_transformers, df_lines, sub_ref)

# Print the admittance matrix
print("Admittance Matrix:")
print(np.round(Y, 3))

if consider_real_transformer_data:
    logging.info("Considered real transformer data")

    # Save the admittance matrix as npy
    np.save(data_loc / "admittance_matrix_real.npy", Y)

else:
    logging.info("Considered uniform transformer data")

    # Save the admittance matrix as npy
    np.save(data_loc / "admittance_matrix.npy", Y)

# Save transmission lines
df_lines.to_csv(data_loc / "transmission_lines.csv", index=False)

# Df substations info
df_substations_info.to_csv(data_loc / "substation_info.csv", index=False)

# Df transformers
df_transformers.to_csv(data_loc / "transformers.csv", index=False)

# %%
