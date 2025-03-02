# --------------------------------------------------------------------------
# Script that prepares the substation and transmission line shapefiles as input for the geomagnetic data processing.
# Graphs the substations and power lines which are connected to the substations.
# Filters the data so as to extract the EHV substations and power lines.
# Author: Dennies Bor
# --------------------------------------------------------------------------
# %%
import os
import gc
import pickle
from pathlib import Path
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiPoint
import geopandas as gpd
import pandas as pd
import numpy as np
import bezpy

# Set up working directories
data_loc = Path(__file__).resolve().parent.parent / "data"

crs = "EPSG:4326"

# Transmission lines data
transmission_lines_path = (
    data_loc
    / "Electric__Power_Transmission_Lines"
    / "Electric__Power_Transmission_Lines.shp"
)
substations_data_path = data_loc / "substations" / "substations.geojson"
ferc_gdf_path = data_loc / "nerc_gdf.geojson"


# function to process the ferc gdf
def process_ferc_gdf(ferc_gdf_path):
    ferc_gdf = gpd.read_file(ferc_gdf_path)
    ferc_gdf = ferc_gdf.to_crs(crs)

    # Rename the ferc regions
    ferc_gdf.loc[ferc_gdf["REGIONS"] == "NorthernGridConnected", "REGIONS"] = "NorthGC"
    ferc_gdf.loc[ferc_gdf["REGIONS"] == "WestConnect", "REGIONS"] = "WestC"
    ferc_gdf.loc[ferc_gdf["REGIONS"] == "NorthernGridUnconnected", "REGIONS"] = (
        "NorthGUC"
    )
    ferc_gdf.loc[ferc_gdf["REGIONS"] == "WestConnectNonEnrolled", "REGIONS"] = "WestCNE"
    ferc_gdf = ferc_gdf[ferc_gdf["REGIONS"] != "NotOrder1000"]
    return ferc_gdf


# Function to process and load the transmission lines
def load_and_process_transmission_lines(transmission_lines, ferc_gdf_path):
    # Load the transmission lines data
    gdf = gpd.read_file(transmission_lines)
    gdf = gdf.to_crs(crs)

    # Rename the ID column to line_id
    gdf.rename({"ID": "line_id"}, inplace=True, axis=1)

    # Explode the gdf to fix the multilinestring
    gdf = gdf.reset_index(drop=True).reset_index(names="original_index")

    # Explode the GeoDataFrame
    exploded = gdf.explode(index_parts=True)

    # Reset index to get the part number
    exploded = exploded.reset_index(level=1)

    # Create a new unique line_id by appending a suffix to the original line_id
    exploded["new_line_id"] = exploded.apply(
        lambda row: (
            f"{row['line_id']}_{row['level_1']}"
            if row["level_1"] > 0
            else row["line_id"]
        ),
        axis=1,
    )

    # Drop unnecessary columns and rename new_line_id to line_id
    exploded = exploded.drop(columns=["level_1", "line_id", "original_index"])
    exploded = exploded.rename(columns={"new_line_id": "line_id"})

    # Filter voltages above 200 kv so as to get the EHV lines
    gdf = exploded[exploded["VOLTAGE"] >= 200]

    # Confine gdf to the ferc regions
    ferc_gdf = process_ferc_gdf(ferc_gdf_path)

    # Spatial join to get the ferc regions
    gdf = gpd.sjoin(
        gdf, ferc_gdf, how="inner", predicate="intersects"
    )  # Only those within ferc

    # Drop index righ
    gdf = gdf.drop(columns="index_right")

    # Reset the index and drop the unnecessary columnsq
    gdf = gdf.reset_index(drop=True)

    # Use Bezpy to get the transmission line length
    gdf["obj"] = gdf.apply(bezpy.tl.TransmissionLine, axis=1)
    gdf["length"] = gdf["obj"].apply(lambda x: x.length)

    # Remove from memory
    del exploded
    del ferc_gdf

    gc.collect()

    return gdf


# A dictionary that map each substation to their roles
substation_type_mapping = {
    # Transmission
    "transmission": "Transmission",
    "subtransmission": "Transmission",
    # Distribution
    "distribution": "Distribution",
    "minor_distribution": "Distribution",
    # Generation
    "generation": "Generation",
    "generator": "Generation",
    # Switching
    "switching": "Switching",
    "switch": "Switching",
    "switchyard": "Switching",
    "switch_yard": "Switching",
    "switchgear": "Switching",
    "switch_gear": "Switching",
    "kirk_switchyard": "Switching",  # Correctly categorized
    # Traction
    "traction": "Traction",
    "rail": "Traction",
    # Industrial
    "industrial": "Industrial",
    # Conversion
    "converter": "Conversion",
    # Storage
    "storage": "Storage",
    # Compensation
    "compensation": "Compensation",
    # Gas
    "gas": "Gas",
    "compression": "Gas",
    # Other/Misc
    "transition": "Other",
    "transformer": "Other",
    "collector": "Other",
    "heat-exchanger": "Other",
    "internet": "Other",
    "valve": "Other",
    # General/Unspecified
    "yes": "General",
    "substation": "General",
    "unit": "General",
    # Named substations
    "ross_substation": "General",
    "mayberry_substation": "General",
    "skinner_lake_substation": "General",
    "jacobs_road_substation": "General",
    "bop/lmf_substation": "General",
}


# Function to process the substations data
def load_and_process_substations(substations_data_path, ferc_gdf_path):
    # Load the substations data
    substations_gdf = gpd.read_file(substations_data_path)
    substations_gdf = substations_gdf.to_crs(crs)

    # make new ss_id based on osmid
    substations_gdf["ss_id"] = substations_gdf["osmid"]

    # A helper function to convert the substations shp to polygon
    def to_polygon(geom):
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom
        elif isinstance(geom, Point):
            return geom.buffer(0.00001)  # Create a small circular polygon
        elif isinstance(geom, LineString):
            return geom.buffer(0.00001)  # Create a small buffered polygon
        else:
            return None  # Return None for any other geometry type

    # Convert all geometries to polygons
    substations_gdf["geometry"] = substations_gdf["geometry"].apply(to_polygon)

    # Remove any rows where geometry conversion failed (returned None)
    substations_gdf = substations_gdf.dropna(subset=["geometry"])

    # Get those within the ferc regions
    ferc_gdf = process_ferc_gdf(ferc_gdf_path)
    substations_gdf = gpd.sjoin(
        substations_gdf, ferc_gdf, how="inner", predicate="intersects"
    )  # Only those within ferc

    # Reset the index
    substations_gdf = substations_gdf.reset_index(drop=True)

    # A helper function to get the substation type
    def map_substation_type(subtype):
        if pd.isna(subtype) or subtype is None:
            return "Unknown"
        # Convert to lowercase and remove any leading/trailing whitespace
        cleaned_subtype = str(subtype).lower().strip()
        # Check if the subtype contains 'switchyard'
        if "switchyard" in cleaned_subtype:
            return "Switching"
        # Use the mapping, defaulting to 'Other' if not found
        return substation_type_mapping.get(cleaned_subtype, "Other")

    # Apply the mapping to get the substation type
    substations_gdf["substation"] = substations_gdf["substation"].apply(
        map_substation_type
    )

    # Remove the substation types which are nan
    substations_gdf = substations_gdf[~pd.isna(substations_gdf["substation"])]

    # Drop the index right
    substations_gdf = substations_gdf.drop(columns="index_right")

    del ferc_gdf
    gc.collect()

    return substations_gdf


def get_nodes_edges(tl_gdf_subset):

    # Initialize dictionaries to store connections
    substation_to_lines = {}
    line_to_substations = {}
    substation_to_line_voltages = {}
    substation_to_substations = {}
    line_to_substations_map = {}
    sub2all = {}

    # Fill dictionaries with connections from intersection_gdf_subset DataFrame
    for idx, row in tl_gdf_subset.iterrows():
        substation_id = row["SS_ID"]
        line_id = row["LINE_ID"]
        line_voltage = row["LINE_VOLTAGE"]

        # Track which lines are connected to each substation
        if substation_id not in substation_to_lines:
            substation_to_lines[substation_id] = []
        if line_id not in substation_to_lines[substation_id]:
            substation_to_lines[substation_id].append(line_id)

        # Track which line voltages are connected to each substation
        if substation_id not in substation_to_line_voltages:
            substation_to_line_voltages[substation_id] = []
        substation_to_line_voltages[substation_id].append(line_voltage)

        # Track which substations are connected per line
        if line_id not in line_to_substations:
            line_to_substations[line_id] = []
        if substation_id not in line_to_substations[line_id]:
            line_to_substations[line_id].append(substation_id)

    # Create substation-to-substation connections and line-to-substations mapping
    for line_id, substations in line_to_substations.items():
        if line_id in line_to_substations_map:
            continue
        line_to_substations_map[line_id] = []
        for i in range(len(substations)):
            for j in range(i + 1, len(substations)):
                sub1, sub2 = substations[i], substations[j]
                if sub1 not in substation_to_substations:
                    substation_to_substations[sub1] = []
                if sub2 not in substation_to_substations[sub1]:
                    substation_to_substations[sub1].append(sub2)

                if sub2 not in substation_to_substations:
                    substation_to_substations[sub2] = []
                if sub1 not in substation_to_substations[sub2]:
                    substation_to_substations[sub2].append(sub1)

                # Track which substations are connected by each line
                line_to_substations_map[line_id].append((sub1, sub2))

    # Delete the variables
    del tl_gdf_subset
    gc.collect()

    return (
        substation_to_lines,
        line_to_substations,
        substation_to_line_voltages,
        substation_to_substations,
        line_to_substations_map,
    )


# Intersect transmission lines with substations
def graph_the_nodes_edges(substation_gdf, translines_gdf):
    # Functions that intersect the transmission lines with the substations
    # Get the nodes (substations) and edges (transmission lines)

    # Make sure both have the correct crs
    substation_gdf = substation_gdf.to_crs(crs)
    translines_gdf = translines_gdf.to_crs(crs)

    # Reprokject to NAD83
    # CRS for the contiguous US
    projected_crs = "EPSG:5070"
    substation_gdf = substation_gdf.to_crs(projected_crs)
    translines_gdf = translines_gdf.to_crs(projected_crs)

    # Add a buffer of 0.25 km to substations
    buffer_distance = 0.25
    substation_gdf["buffered"] = substation_gdf.geometry.buffer(buffer_distance)

    # Create a GeoDataFrame for the buffered substations
    buffered_ss_gdf = gpd.GeoDataFrame(substation_gdf, geometry="buffered")

    # Perform the intersection with the substation data
    intersection_gdf = gpd.sjoin(
        translines_gdf, buffered_ss_gdf, how="inner", predicate="intersects"
    )
    
    # Convert back to WGS84
    intersection_gdf = intersection_gdf.to_crs(crs)

    # # Drop the buffered column
    # intersection_gdf = intersection_gdf.drop(columns="buffered")

    # Rename the columns
    renamed_columns = {
        "name": "SS_NAME",
        "operator": "SS_OPERATOR",
        "voltage": "SS_VOLTAGE",
        "substation": "SS_TYPE",
        "geometry": "geometry",
        "ss_id": "SS_ID",
        "REGION_ID_left": "REGION_ID",
        "REGIONS_left": "REGION",
        "line_id": "LINE_ID",
        "NAICS_CODE": "LINE_NAICS_CODE",
        "VOLTAGE": "LINE_VOLTAGE",
        "VOLT_CLASS": "LINE_VOLT_CLASS",
        "INFERRED": "INFERRED",
        "SUB_1": "SUB_1",
        "SUB_2": "SUB_2",
        "length": "LINE_LEN",
    }

    # Create a GeoDataFrame for transmission lines
    tl_gdf = gpd.GeoDataFrame(intersection_gdf.copy())
    tl_gdf = tl_gdf.rename(columns={"geometry_left": "geometry"})
    tl_gdf = tl_gdf.drop(columns=["geometry_right"])
    # Set geom
    tl_gdf = tl_gdf.set_geometry("geometry")

    # Create a GeoDataFrame for substations
    substation_gdf = gpd.GeoDataFrame(intersection_gdf.copy())
    substation_gdf = substation_gdf.drop(columns=["geometry"])
    substation_gdf = substation_gdf.rename(columns={"geometry_right": "geometry"})
    # Set geom
    substation_gdf = substation_gdf.set_geometry("geometry")

    # Subset the tl_gdf to include only the columns in renamed_columns
    tl_gdf_subset = tl_gdf[list(renamed_columns.keys())]
    ss_gdf_subset = substation_gdf[list(renamed_columns.keys())]

    # Rename the columns
    tl_gdf_subset = tl_gdf_subset.rename(columns=renamed_columns)
    ss_gdf_subset = ss_gdf_subset.rename(columns=renamed_columns)

    # Ensure the geometry column is correctly set
    tl_gdf_subset = gpd.GeoDataFrame(tl_gdf_subset, geometry="geometry")
    substation_gdf = gpd.GeoDataFrame(ss_gdf_subset, geometry="geometry")

    # Calc centroid as rep point and make it geom
    substation_gdf["rep_point"] = substation_gdf.geometry.centroid
    substation_gdf = substation_gdf.set_geometry("rep_point")

    # Reproject to WGS84
    substation_gdf = substation_gdf.to_crs(crs)
    tl_gdf_subset = tl_gdf_subset.to_crs(crs)
    ss_gdf_subset = ss_gdf_subset.to_crs(crs)

    del intersection_gdf
    del buffered_ss_gdf
    del translines_gdf
    gc.collect()

    return substation_gdf, tl_gdf_subset, ss_gdf_subset


# Helper function to split a LineString at a Point (for the lines which cross multiple subs)
def split_line_at_intersections(line, buffers):
    intersection_points = []

    for buffer in buffers:
        if line.intersects(buffer):
            intersection = line.intersection(buffer)

            if isinstance(intersection, Point):
                intersection_points.append(intersection)
            elif isinstance(intersection, LineString):
                # Convert LineString intersection to Points
                intersection_points.extend(
                    [Point(coord) for coord in intersection.coords]
                )
            elif isinstance(intersection, MultiPoint):
                intersection_points.extend(list(intersection))
            else:
                # Handle other geometries like GeometryCollection, MultiLineString
                for geom in intersection.geoms:
                    if isinstance(geom, Point):
                        intersection_points.append(geom)
                    elif isinstance(geom, LineString):
                        intersection_points.extend(
                            [Point(coord) for coord in geom.coords]
                        )

    # Deduplicate and sort split points along the line
    intersection_points = sorted(
        set(intersection_points), key=lambda p: line.project(p)
    )

    if not intersection_points:
        return [line]

    # Perform the splitting
    split_segments = []
    current_line = line

    for point in intersection_points:
        # Split the current line at this point
        new_segments = split(current_line, point)
        split_segments.append(new_segments.geoms[0])
        current_line = new_segments.geoms[1]

    split_segments.append(current_line)
    return split_segments


def get_EHV_lines(substation_gdf, tl_gdf_subset, trans_lines_within_FERC):

    (
        substation_to_lines,
        line_to_substations,
        substation_to_line_voltages,
        substation_to_substations,
        line_to_substations_map,
    ) = get_nodes_edges(tl_gdf_subset)

    # Unique substations
    unique_substations_gdf = substation_gdf.drop_duplicates(subset="SS_ID")

    # To lat lon crs
    unique_substations_gdf = unique_substations_gdf.to_crs(epsg=4326)

    # Extract latitude and longitude from the centroid
    unique_substations_gdf["lat"] = unique_substations_gdf.centroid.y
    unique_substations_gdf["lon"] = unique_substations_gdf.centroid.x

    unique_substations_gdf["connected_tl_id"] = unique_substations_gdf["SS_ID"].map(
        substation_to_lines
    )

    # Assign_voltages
    unique_substations_gdf["LINE_VOLTS"] = unique_substations_gdf["SS_ID"].map(
        substation_to_line_voltages
    )

    # substation useful columns
    useful_ss_cols = [
        "SS_ID",
        "SS_NAME",
        "SS_OPERATOR",
        "SS_VOLTAGE",
        "SS_TYPE",
        "REGION_ID",
        "REGION",
        "lat",
        "lon",
        "connected_tl_id",
        "LINE_VOLTS",
    ]

    # Subset the cols and save the ss df
    ss_df = unique_substations_gdf[useful_ss_cols]

    tm_ss_df_gt_300v = ss_df.copy()
    tm_ss_df_gt_300v.reset_index(inplace=True, drop=True)

    # Get a list of all connected transmission line IDs
    connected_tl_ids = tm_ss_df_gt_300v["connected_tl_id"].explode().unique()

    # Filter the transmission lines within FERC to include only those with IDs in the connected_tl_ids list
    trans_lines_within_FERC_filtered = trans_lines_within_FERC[
        trans_lines_within_FERC["line_id"].isin(connected_tl_ids)
    ]
    trans_lines_within_FERC_filtered = trans_lines_within_FERC_filtered[
        [
            "line_id",
            "OWNER",
            "VOLTAGE",
            "VOLT_CLASS",
            "SUB_1",
            "SUB_2",
            "SHAPE__Len",
            "geometry",
            "REGION_ID",
            "REGIONS",
            "length",
        ]
    ]

    trans_lines_within_FERC_filtered.reset_index(inplace=True, drop=True)

    # Create the ss_region_dict from the given DataFrame
    ss_region_dict = dict(zip(tm_ss_df_gt_300v["SS_ID"], tm_ss_df_gt_300v["REGION"]))

    # Initialize an empty DataFrame
    df_ss_graph = pd.DataFrame()

    # Add the SS_IDs column
    df_ss_graph["SS_IDs"] = substation_to_line_voltages.keys()

    # Map connecting lines to the DataFrame
    df_ss_graph["Connecting_Lines"] = df_ss_graph["SS_IDs"].map(substation_to_lines)

    # Map connecting substations to the DataFrame
    df_ss_graph["Connecting_Substations"] = df_ss_graph["SS_IDs"].map(
        substation_to_substations
    )

    # Map regions to the DataFrame
    df_ss_graph["REGION"] = df_ss_graph["SS_IDs"].map(ss_region_dict)

    # Filter the DataFrame to include only substations in tm_ss_df_gt_300v
    df_ss_graph = df_ss_graph[
        df_ss_graph["SS_IDs"].isin(tm_ss_df_gt_300v["SS_ID"])
    ].reset_index(drop=True)

    split_lines_gdf = gpd.GeoDataFrame(
        columns=["line_id", "geometry", "sub1", "sub2"], crs=tl_gdf_subset.crs
    )
    split_lines_list = []

    for line_id, substation_pairs in line_to_substations_map.items():

        if len(substation_pairs) < 2:
            continue
        # Query the original line geometry by line_id
        original_line = tl_gdf_subset.loc[
            tl_gdf_subset["LINE_ID"] == line_id, "geometry"
        ].values[0]

        # Iterate through each pair of substations connected by the line
        for sub1, sub2 in substation_pairs:
            # Find the substation points
            sub1_geom = substation_gdf.loc[
                substation_gdf["SS_ID"] == sub1, "geometry"
            ].centroid.values[0]
            sub2_geom = substation_gdf.loc[
                substation_gdf["SS_ID"] == sub2, "geometry"
            ].centroid.values[0]

            # Split the line at the substations
            split_segments = split_line_at_intersections(
                original_line, [sub1_geom, sub2_geom]
            )

            # Store the split segments in the result GeoDataFrame
            for segment in split_segments:
                split_lines_list.append(
                    {
                        "line_id": line_id,
                        "geometry": segment,
                        "substation_a": sub1,
                        "substation_b": sub2,
                    }
                )

    # Convert list of dictionaries to a GeoDataFrame
    split_tl_gdf_subset = gpd.GeoDataFrame(split_lines_list, crs=tl_gdf_subset.crs)

    # Reset the index
    split_tl_gdf_subset.reset_index(drop=True, inplace=True)

    # Create new line ids by extending existing line id with index of appearance
    split_tl_gdf_subset["new_line_id"] = (
        split_tl_gdf_subset["line_id"]
        + "_"
        + split_tl_gdf_subset.groupby("line_id").cumcount().astype(str)
    )

    #  Query for line voltages from lines within FERC
    split_tl_gdf_subset = split_tl_gdf_subset.merge(
        trans_lines_within_FERC_filtered[["line_id", "VOLTAGE"]],
        on="line_id",
        how="left",
    )

    # Drop rows in 'trans_lines_within_FERC_filtered_' corresponding to line ids in 'split_tl_gdf_subset'
    trans_lines_within_FERC_filtered_ = trans_lines_within_FERC_filtered[
        ~trans_lines_within_FERC_filtered["line_id"].isin(
            split_tl_gdf_subset["line_id"]
        )
    ]

    # Select relevant columns ('line_id', 'VOLTAGE', 'geometry')
    trans_lines_within_FERC_filtered_ = trans_lines_within_FERC_filtered_[
        ["line_id", "VOLTAGE", "geometry"]
    ]

    # Drop the old 'line_id' column from 'split_tl_gdf_subset'
    split_tl_gdf_subset.drop(columns=["line_id"], inplace=True)

    # Rename 'new_line_id' column to 'line_id' in 'split_tl_gdf_subset'
    split_tl_gdf_subset.rename(columns={"new_line_id": "line_id"}, inplace=True)

    # Concatenate the filtered GeoDataFrame with the split lines GeoDataFrame
    trans_lines_within_FERC_filtered_ = gpd.GeoDataFrame(
        pd.concat(
            [
                trans_lines_within_FERC_filtered_,
                split_tl_gdf_subset[["line_id", "VOLTAGE", "geometry"]],
            ]
        ),
        crs=trans_lines_within_FERC_filtered_.crs,
    )

    # filter substation_df with concatenated substation_a and substation_b
    sub_filtered_gdf = substation_gdf.copy()

    subs = list(
        set(
            split_tl_gdf_subset.substation_a.unique().tolist()
            + split_tl_gdf_subset.substation_b.unique().tolist()
        )
    )
    sub_filtered_gdf = sub_filtered_gdf[sub_filtered_gdf["SS_ID"].isin(subs)]

    # Set rep Point as geometry and save the two gdfs for furhter analysis in QGIS
    sub_filtered_gdf = sub_filtered_gdf.set_geometry("rep_point")

    # Drop the original geometry column if it exists
    if "geometry" in sub_filtered_gdf.columns:
        sub_filtered_gdf = sub_filtered_gdf.drop(columns=["geometry"])

    split_tl_gdf_subset = split_tl_gdf_subset.set_geometry("geometry")

    # Filter the line_to_substations_map to include only those lines with exactly two substations
    filtered_line_to_substations = {
        line: subs for line, subs in line_to_substations_map.items() if len(subs) == 1
    }

    # Create a DataFrame from the filtered data
    line_to_substations_df = pd.DataFrame(
        [
            {"line_id": line, "substation_a": subs[0][0], "substation_b": subs[0][1]}
            for line, subs in filtered_line_to_substations.items()
        ]
    )

    # Concat with split line strings
    line_to_substations_df = pd.concat(
        [
            line_to_substations_df,
            split_tl_gdf_subset[["line_id", "substation_a", "substation_b"]],
        ]
    )
    
    # USe bezpy to calculate the length
    trans_lines_within_FERC_filtered_["obj"] = trans_lines_within_FERC_filtered_.apply(bezpy.tl.TransmissionLine, axis=1) 
    trans_lines_within_FERC_filtered_["length"] = trans_lines_within_FERC_filtered_["obj"].apply(lambda x: x.length)  

    # Calculate geometry length

    # Filter infinite vals, although we lose some transmission lines
    trans_lines_within_FERC_filtered_ = trans_lines_within_FERC_filtered_[
        ~(trans_lines_within_FERC_filtered_.VOLTAGE == -9.99999e05)
    ]

    # Remove is na
    trans_lines_within_FERC_filtered_ = trans_lines_within_FERC_filtered_.dropna()

    # EHV lines filtered
    df_lines_EHV = line_to_substations_df[
        line_to_substations_df.substation_a.isin(tm_ss_df_gt_300v.SS_ID)
    ]
    df_lines_EHV = df_lines_EHV[df_lines_EHV.substation_b.isin(tm_ss_df_gt_300v.SS_ID)]

    # Merge with trans_lines_within_FERC_filtered__filtered
    df_lines_EHV = df_lines_EHV.merge(
        trans_lines_within_FERC_filtered_[["line_id", "VOLTAGE", "length"]],
        left_on="line_id",
        right_on="line_id",
        how="left",
    )
    df_lines_EHV = df_lines_EHV[~df_lines_EHV.duplicated()]
    df_lines_EHV = df_lines_EHV[~(df_lines_EHV.VOLTAGE.isna())]
    df_lines_EHV["VOLTAGE"] = df_lines_EHV["VOLTAGE"].astype(int)

    # Focus on high voltage transmission lines. Drop < 200
    df_lines_EHV = df_lines_EHV[df_lines_EHV.VOLTAGE >= 200]

    # Concat with transmission lines to get geometry and convert to gdf
    df_lines_EHV = df_lines_EHV.merge(
        trans_lines_within_FERC_filtered_[["line_id", "geometry"]],
        left_on="line_id",
        right_on="line_id",
        how="left",
    )
    df_lines_EHV = gpd.GeoDataFrame(df_lines_EHV, geometry="geometry")

    # Drop dupliocated\
    df_lines_EHV = df_lines_EHV[~df_lines_EHV.duplicated()]

    # Create a unique mapping dictionary based on buses on substation_a and substation_b
    # and line_id concat (sub1+line_id, sub2+line_id)

    df_lines_EHV["from_bus_id"] = (
        df_lines_EHV["substation_a"].astype(str)
        + "_"
        + df_lines_EHV["VOLTAGE"].astype(int).astype(str)
    )
    df_lines_EHV["to_bus_id"] = (
        df_lines_EHV["substation_b"].astype(str)
        + "_"
        + df_lines_EHV["VOLTAGE"].astype(int).astype(str)
    )

    # Extract unique substation-voltage pairs
    sub1_voltage_pairs = df_lines_EHV[
        ["substation_a", "VOLTAGE", "from_bus_id"]
    ].drop_duplicates()
    sub2_voltage_pairs = df_lines_EHV[
        ["substation_b", "VOLTAGE", "to_bus_id"]
    ].drop_duplicates()

    # Rename columns for merging
    sub1_voltage_pairs.columns = ["substation", "voltage", "bus_id"]
    sub2_voltage_pairs.columns = ["substation", "voltage", "bus_id"]

    # Combine unique substation-voltage pairs
    unique_sub_voltage_pairs = pd.concat(
        [sub1_voltage_pairs, sub2_voltage_pairs]
    ).drop_duplicates()

    # Reset index
    unique_sub_voltage_pairs.reset_index(drop=True, inplace=True)

    # Drop duplicates
    unique_sub_voltage_pairs.drop_duplicates(inplace=True)
    unique_sub_voltage_pairs.reset_index(drop=True, inplace=True)

    # Merge with substation to get geometries
    unique_sub_voltage_pairs = unique_sub_voltage_pairs.merge(
        substation_gdf[["SS_ID", "rep_point", "SS_TYPE", "REGION"]],
        left_on="substation",
        right_on="SS_ID",
        how="left",
    )

    # Set rep_point as geometry
    unique_sub_voltage_pairs = gpd.GeoDataFrame(
        unique_sub_voltage_pairs, geometry="rep_point"
    )

    # Drop SS_ID
    unique_sub_voltage_pairs.drop("SS_ID", axis=1, inplace=True)

    # Drop or interpolate transmission lines whose voltage rating isn't within the US. Voltage standards
    line_voltage_ratings = {
        345: 345,
        230: 230,
        450: np.nan,
        500: 500,
        765: 765,
        250: 230,
        400: np.nan,
        232: 230,
        1000: np.nan,
        220: 230,
        273: np.nan,
        218: 230,
        236: 230,
        287: np.nan,
        238: 230,
        200: np.nan,
    }

    df_lines_EHV["VOLTAGE"] = df_lines_EHV.VOLTAGE.map(line_voltage_ratings)
    df_lines_EHV = df_lines_EHV.dropna(subset=["VOLTAGE"])

    return (
        unique_sub_voltage_pairs,
        substation_to_line_voltages,
        ss_df,
        df_lines_EHV,
        trans_lines_within_FERC_filtered_,
        tl_gdf_subset[["LINE_ID", "LINE_VOLTAGE", "geometry"]],
    )


# %%
# Load the tl data
translines_gdf = load_and_process_transmission_lines(
    transmission_lines_path, ferc_gdf_path
)

# load substations data
substation_gdf = load_and_process_substations(substations_data_path, ferc_gdf_path)

# Graph the nodes and edges
substation_gdf, tl_gdf_subset, ss_gdf_subset = graph_the_nodes_edges(
    substation_gdf, translines_gdf
)

# Get the EHV lines
(
    unique_sub_voltage_pairs,
    substation_to_line_voltages,
    ss_df,
    df_lines_EHV,
    trans_lines_within_FERC_filtered_,
    tl_gdf_subset,
) = get_EHV_lines(substation_gdf, tl_gdf_subset, translines_gdf)

# %%
# Pickle unique_sub_voltage_pairs
with open(data_loc / 'unique_sub_voltage_pairs.pkl', 'wb') as f:
    pickle.dump(unique_sub_voltage_pairs, f)

# Pickle substation to line voltages
with open(data_loc / 'substation_to_line_voltages.pkl', 'wb') as f:
    pickle.dump(substation_to_line_voltages, f)

# Pickle substations ss_df
with open(data_loc / 'ss_df.pkl', 'wb') as f:
    pickle.dump(ss_df, f)

# Pickle df_lines_EHV
with open(data_loc / 'df_lines_EHV.pkl', 'wb') as f:
    pickle.dump(df_lines_EHV, f)

# Save tranlines within FERC filtered as pkl
with open(data_loc / "trans_lines_within_FERC_filtered.pkl", 'wb') as f:
    pickle.dump(trans_lines_within_FERC_filtered_, f)

# Pickle tl_gdf_subset line_id and geometry only
with open(data_loc / "tl_gdf_subset.pkl", 'wb') as f:
    pickle.dump(tl_gdf_subset[['LINE_ID', "LINE_VOLTAGE", "geometry"]].drop_duplicates(subset='LINE_ID'), f)