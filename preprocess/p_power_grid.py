"""
Script to prepare extra-high voltage (EHV) substation and transmission lines.

Authors:
- Dennies Bor
- Ed Oughton

Date:
- February 2025
"""
import os
import gc
import pickle
from pathlib import Path
from shapely.geometry import (
    Point, Polygon, LineString, MultiPolygon, MultiPoint
)
import geopandas as gpd
import pandas as pd
import numpy as np
import bezpy

DATA_LOC = Path(__file__).resolve().parent.parent / "data"

def load_and_process_transmission_lines(transmission_lines_path, ferc_gdf_path):
    """
    Load and process transmission line data, filtering for extra-high voltage 
    (EHV) lines and associating them with FERC regions.

    Parameters:
    transmission_lines_path : str
        Path to the transmission lines GeoJSON or shapefile.
    ferc_gdf_path : str
        Path to the FERC region boundaries GeoJSON or shapefile.

    Returns:
    gpd.GeoDataFrame
        Processed transmission lines with unique IDs, voltage filtering, 
        FERC regions, and computed lengths.
    """
    gdf = gpd.read_file(transmission_lines_path).to_crs("EPSG:4326")
    gdf.rename(columns={"ID": "line_id"}, inplace=True)
    gdf = gdf.reset_index(drop=True).explode(index_parts=True).reset_index(level=1)
    gdf["line_id"] = gdf.apply(
        lambda row: f"{row['line_id']}_{row['level_1']}" if row["level_1"] > 0 else row["line_id"],
        axis=1,
    )
    gdf = gdf[gdf["VOLTAGE"] >= 200].drop(columns=["level_1"])
    ferc_gdf = process_ferc_gdf(ferc_gdf_path)
    gdf = gpd.sjoin(gdf, ferc_gdf, how="inner", predicate="intersects").drop(columns="index_right")
    gdf["length"] = gdf.apply(lambda row: bezpy.tl.TransmissionLine(row).length, axis=1)
    del ferc_gdf
    gc.collect()
    return gdf

def process_ferc_gdf(ferc_gdf_path):
    """
    Process FERC region GeoDataFrame by standardizing region names and 
    filtering out non-relevant regions.

    Parameters:
    ferc_gdf_path : str
        Path to the FERC region boundaries GeoJSON or shapefile.

    Returns:
    gpd.GeoDataFrame
        Processed FERC GeoDataFrame with standardized names and filtering.
    """
    ferc_gdf = gpd.read_file(ferc_gdf_path).to_crs("EPSG:4326")
    rename_mapping = {
        "NorthernGridConnected": "NorthGC",
        "WestConnect": "WestC",
        "NorthernGridUnconnected": "NorthGUC",
        "WestConnectNonEnrolled": "WestCNE"
    }
    ferc_gdf["REGIONS"] = ferc_gdf["REGIONS"].replace(rename_mapping)
    ferc_gdf = ferc_gdf[ferc_gdf["REGIONS"] != "NotOrder1000"]
    return ferc_gdf

def load_and_process_substations(substations_data_path, ferc_gdf_path):
    """
    Load and process substation data, standardizing geometry, filtering by 
    FERC regions, and categorizing substations.

    Parameters:
    substations_data_path : str
        Path to the substations GeoJSON or shapefile.
    ferc_gdf_path : str
        Path to the FERC region boundaries GeoJSON or shapefile.

    Returns:
    gpd.GeoDataFrame
        Processed substations with standard geometry and filtered by FERC regions.
    """
    substations_gdf = gpd.read_file(substations_data_path).to_crs("EPSG:4326")
    substations_gdf["ss_id"] = substations_gdf["osmid"]
    substations_gdf["geometry"] = substations_gdf["geometry"].apply(
        lambda geom: geom.buffer(0.00001) if isinstance(geom, (Point, LineString)) else geom
    )
    substations_gdf.dropna(subset=["geometry"], inplace=True)
    ferc_gdf = process_ferc_gdf(ferc_gdf_path)
    substations_gdf = gpd.sjoin(substations_gdf, ferc_gdf, how="inner", predicate="intersects")
    substations_gdf.drop(columns="index_right", inplace=True)
    del ferc_gdf
    gc.collect()
    return substations_gdf


if __name__ == "__main__":

    dir_out = DATA_LOC / "grid_processed"
    os.makedirs(dir_out, exist_ok=True)

    filename = "Electric__Power_Transmission_Lines.shp"
    folder = DATA_LOC / "Electric__Power_Transmission_Lines"
    transmission_lines_path = folder / filename
    ferc_gdf_path = DATA_LOC / "nerc_gdf.geojson"
    translines_gdf = load_and_process_transmission_lines(transmission_lines_path, ferc_gdf_path)
    translines_gdf.to_file(dir_out / 'processed_transmission_lines.gpkg', driver='GPKG')
    with open(dir_out / 'processed_transmission_lines.pkl', 'wb') as f:
        pickle.dump(translines_gdf, f)

    substations_data_path = DATA_LOC / "substation_locations" / "substations.geojson"
    substation_gdf = load_and_process_substations(substations_data_path, ferc_gdf_path)
    substation_gdf.to_file(dir_out / 'processed_substations.gpkg', driver='GPKG')
    with open(dir_out / 'processed_substations.pkl', 'wb') as f:
        pickle.dump(substation_gdf, f)
