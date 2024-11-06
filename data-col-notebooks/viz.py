# %%

# ...........................................................................................
# Visualize the GICs along the transmission lines and transformers in the network
# ...........................................................................................
import os
import sys
import gc
import string
import logging
import pickle
from pathlib import Path
from memory_profiler import profile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RectBivariateSpline
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import geopandas as gpd

# Set font as times new roman
plt.rcParams["font.family"] = "Times New Roman"

# lower case alphabet
alphabet = string.ascii_lowercase


# set up custom logger
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
grid_regions_gdf = gpd.read_file(data_loc / "nerc_gdf.geojson")

# Load data from preprocess storm data
df_lines_path = data_loc / "final_tl_data.pkl"
line_coords_file = data_loc / "line_coords.pkl"


# Read pickle function
def read_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


line_coordinates, valid_indices = read_pickle(line_coords_file)
df_lines, mt_coords, mt_names = read_pickle(df_lines_path)

grid_e_100_path = data_loc / "grid_e_100.pkl"  # 100 year e-field
grid_e_500_path = data_loc / "grid_e_500.pkl"  # 500 year e-field
grid_e_1000_path = data_loc / "grid_e_1000.pkl"  # 1000 year e-field
grid_e_gannon_path = data_loc / "grid_e_gannon.pkl"  # Gannon storm e-field

# GIC data
df_gic_halloween = pd.read_csv(data_loc / "gic_halloween.csv")
df_gic_st_patricks = pd.read_csv(data_loc / "gic_st_patricks.csv")
df_gic_gannon = pd.read_csv(data_loc / "gic_gannon.csv")
df_gic_100 = pd.read_csv(data_loc / "gic_100.csv")
df_gic_500 = pd.read_csv(data_loc / "gic_500.csv")
df_gic_1000 = pd.read_csv(data_loc / "gic_1000.csv")

# Read GIC data
# df_gic = pd.read_csv(data_loc / "econ_data" / "winding_median_df.csv")
df_gic = pd.read_csv(data_loc / "econ_data" / "winding_mean_df.csv")


# %%
# Line_collection
def plot_transmission_lines(
    ax,
    line_coordinates,
    values,
    cmap="viridis",
    line_width=1,
    alpha=0.7,
    vmin=None,
    vmax=None,
):
    """
    Plot transmission lines on a map with color based on values.

    Parameters:
    - ax: matplotlib axis with cartopy projection
    - line_coordinates: list of numpy arrays containing line coordinates
    - values: list or array of values for coloring the lines
    - cmap: colormap name (default: 'viridis')
    - line_width: width of the lines (default: 1)
    - alpha: transparency of the lines (default: 0.7)

    Returns:
    - line_collection: the plotted LineCollection
    """
    if not line_coordinates:
        logger.error("No valid line segments found.")
        return None

    max_value = vmax if vmax is not None else np.max(values)
    min_value = vmin if vmin is not None else np.min(values)

    norm = colors.SymLogNorm(
        linthresh=(max_value * 0.25),
        linscale=1.0,
        vmin=max_value,
        vmax=min_value,
    )

    line_collection = LineCollection(
        line_coordinates,
        cmap=cmap,
        norm=norm,
        linewidths=line_width,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
    )

    line_collection.set_array(values)
    ax.add_collection(line_collection)
    return line_collection


# %%
# Set up map
def setup_map(ax, spatial_extent=[-120, -75, 25, 50]):
    ax.set_extent(spatial_extent, ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#F0F0F0")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey")
    # ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#E6F3FF")
    ax.add_feature(cfeature.LAKES, alpha=0.5, linewidth=0.5, edgecolor="grey")

    gl = ax.gridlines(
        draw_labels=False, linewidth=0.2, color="grey", alpha=0.5, linestyle="--"
    )

    return ax


# Add grid regions set up
def add_ferc_regions(ax, grid_regions_gdf=grid_regions_gdf, zorder=2):
    """
    Add grid region boundaries to the map.

    Parameters:
    - ax: The axis to which the boundaries will be added.
    - grid_regions_gdf: GeoDataFrame containing the grid regions.
    - zorder: Drawing order for the boundaries (default is 2).
    """
    # Add boundaries without filling
    for idx, region in grid_regions_gdf.iterrows():
        ax.add_geometries(
            [region.geometry],
            crs=ccrs.PlateCarree(),
            edgecolor="grey",
            facecolor="none",
            linewidth=0.8,
            alpha=1,  # Fully opaque
            zorder=zorder,
        )

    return ax


# %%
# Plot for MT sites
def plot_mt_sites_e_fields(ax, mt_coordinates, e_fields, cmap="viridis"):

    sizes = 1 / (1 - np.log10(e_fields / np.nanmax(e_fields)))

    scatter = ax.scatter(
        mt_coordinates[:, 1],  # Longitude
        mt_coordinates[:, 0],  # Latitude
        c=e_fields,
        cmap=cmap,
        s=50 * sizes,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )

    return (scatter,)


# %%
@profile
def plot_mt_sites_e_fields_contour(
    ax,
    data,
    cmap="viridis",
    global_min=None,
    global_max=None,
):
    grid_x, grid_y, grid_z, e_fields = data
    global_min = np.nanmin(e_fields) if global_min is None else global_min
    global_max = np.nanmax(e_fields) if global_max is None else global_max

    # Create the contour plot
    mesh = ax.pcolormesh(
        grid_x,
        grid_y,
        grid_z,
        cmap=cmap,
        alpha=0.7,
        norm=colors.SymLogNorm(
            linthresh=(global_max * 0.1), linscale=1.0, vmin=global_min, vmax=global_max
        ),
        shading="gouraud",
        transform=ccrs.PlateCarree(),
    )

    current_min, current_max = np.nanmin(e_fields), np.nanmax(e_fields)

    # Clear the memory
    del grid_x, grid_y, grid_z
    gc.collect()

    # Add scatter plot for actual MT site locations
    return mesh, current_min, current_max


# %%
def create_custom_colorbar_e_field(
    ax, obj, label, current_min, current_max, title, vmin, vmax, e_field=True
):
    # Create a new axis for the colorbar
    bbox = ax.get_position()
    cax = ax.figure.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])

    # Add the colorbar
    colorbar = plt.colorbar(
        obj,
        cax=cax,  # Use the custom axis for the colorbar
        label=label,
        orientation="vertical",
    )

    # Set the title and label for the colorbar
    colorbar.ax.set_ylabel(title, rotation=90, labelpad=1, fontsize=8)

    if e_field:
        # Create custom ticks at 1, 10, and max
        custom_ticks = [1, 10, vmax]

        # Add current_max if it's different from vmax
        if current_max != vmax and current_max not in custom_ticks:
            custom_ticks.append(current_max)

        # Sort and remove duplicates
        custom_ticks = sorted(list(set(custom_ticks)))

        # Set custom tick locations and labels
        colorbar.set_ticks(custom_ticks)

        # Format the tick labels to handle small numbers
        def tick_formatter(x, p):
            if x < 0.01:
                return f"{x:.2e}"
            else:
                return f"{x:.0f}"  # No decimal places for integers between 1 and 999

        formatted_labels = [tick_formatter(tick, None) for tick in custom_ticks]
        for i, tick in enumerate(custom_ticks):
            if tick == current_max:
                formatted_labels[i] += "*"
                colorbar.ax.yaxis.get_ticklabels()[i].set_color("red")

        colorbar.set_ticklabels(formatted_labels)

        # Disable minor ticks
        colorbar.ax.minorticks_off()

    else:
        # Custom ticks for GIC values
        base_ticks = [10, 100]
        custom_ticks = [tick for tick in base_ticks if vmin <= tick <= vmax]

        if vmax not in custom_ticks:
            custom_ticks.append(vmax)

        # Add current_max if it's different from vmax
        if current_max != vmax and current_max not in custom_ticks:
            custom_ticks.append(current_max)

        # Remove duplicates and sort
        custom_ticks = sorted(list(set(custom_ticks)))

        # Set custom tick locations and labels
        colorbar.set_ticks(custom_ticks)

        # Format the tick labels to handle small and large numbers
        def tick_formatter(x, p):
            if x < 1 or x >= 1001:
                return f"{x:.0f}"
            else:
                return f"{x:.0f}"  # No decimal places for integers between 1 and 999

        formatted_labels = [tick_formatter(tick, None) for tick in custom_ticks]
        for i, tick in enumerate(custom_ticks):
            if tick == current_max:
                formatted_labels[i] += "*"
                colorbar.ax.yaxis.get_ticklabels()[i].set_color('red')

        colorbar.set_ticklabels(formatted_labels)

        # Disable minor ticks
        colorbar.ax.minorticks_off()

    return colorbar


def carto(
    ax,
    label_titles,
    spatial_extent=[-120, -75, 25, 50],
    add_grid_regions=True,
    df_tl=None,
    df_substations=None,
    cmap="viridis",
    value_column=None,
    show_legend=False,
    gic_global_vals=None,
    data_e=None,
    global_min=None,
    global_max=None,
):
    ax = setup_map(ax, spatial_extent)

    if add_grid_regions:
        ax = add_ferc_regions(ax)

    if df_tl is not None and value_column is not None:
        if value_column not in df_tl.columns:
            logger.error(f"Column '{value_column}' not found in df_tl")
        else:
            values = np.abs(df_tl[value_column].values)
            global_min = np.nanmin(values) if global_min is None else global_min
            global_max = np.nanmax(values) if global_max is None else global_max

            line_collection = plot_transmission_lines(
                ax,
                line_coordinates,
                values,
                cmap=cmap,
                vmin=global_min,
                vmax=global_max,
            )

            # Set colorbar
            label_title = label_titles.get(value_column, value_column)

            if line_collection is not None:
                create_custom_colorbar_e_field(
                    ax,
                    line_collection,
                    current_min=np.nanmin(values),
                    current_max=np.nanmax(values),
                    label=label_title,
                    title=label_title,
                    vmin=global_min,
                    vmax=global_max,
                    e_field=False,
                )

                # cleear memory
                del line_collection
            else:
                logger.error("Failed to create line collection.")

    # if df_substations is not None and value_column is not None:
    #     _, _, global_bins = get_discrete_sizes_and_colors(
    #         gic_global_vals, cmap, num_bins=10
    #     )
    #     if value_column not in df_substations.columns:
    #         logger.error(    f"Column '{value_column}' not found in df_substations")
    #     else:
    #         gic_values = df_substations[value_column].values

    #     scatter, bins, cmap_obj, norm = plot_transformer_gic(
    #         ax, df_substations, gic_values, gic_global_vals=gic_global_vals, cmap=cmap
    #     )

    #     if show_legend:
    #         legend_ax = create_gic_bubble_legend(ax, global_bins, cmap, title="GIC (A)")
    #         return ax, legend_ax

    if data_e is not None:

        # scatter = plot_mt_sites_e_fields(ax, mt_coordinates, e_fields, cmap=cmap)
        mesh, current_min, current_max = plot_mt_sites_e_fields_contour(
            ax,
            data_e,
            cmap=cmap,
            global_min=global_min,
            global_max=global_max,
        )

        # Create custom colorbar
        cax = create_custom_colorbar_e_field(
            ax,
            mesh,
            current_min=current_min,
            current_max=current_max,
            label="E Field (V/km) \n (log scale)",
            title="E Field (V/km) \n (log scale)",
            vmin=global_min,
            vmax=global_max,
            e_field=True,
        )

        # clear memory
        del mesh, cax, global_min, global_max

    return ax


# %%
# Load the data
_, _, _, e_fields_100 = read_pickle(grid_e_100_path)
_, _, _, e_fields_500 = read_pickle(grid_e_500_path)
_, _, _, e_fields_1000 = read_pickle(grid_e_1000_path)
_, _, _, e_fields_gannon = read_pickle(grid_e_gannon_path)

stacked_vals_e = np.hstack([e_fields_100, e_fields_500, e_fields_1000, e_fields_gannon])

max_e_field = np.nanmax(stacked_vals_e)
min_e_field = np.nanmin(stacked_vals_e)

# %%
# Plot the GIC for 100 year hazard maps
projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
fig = plt.figure(figsize=(8, 7))
gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.3, hspace=0.01)
data_e_100 = read_pickle(grid_e_100_path)
ax = fig.add_subplot(gs[0], projection=projection)
ax = carto(
    ax,
    label_titles={},
    data_e=data_e_100,
    cmap="magma",
    global_max=max_e_field,
    global_min=min_e_field,
)

plt.tight_layout()
plt.show()

gc.collect()
# %%
# Hazard Maps (50, 100, 500, 1000 years)

projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
fig = plt.figure(figsize=(8, 7))
gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.3, hspace=0.01)

ax = fig.add_subplot(gs[0], projection=projection)
# Plot the GIC for 100 year hazard maps
# Global mins - stack the V_100, V_500, V_1000

stacked_values = np.hstack(df_lines[["V_100", "V_500", "V_1000", "V_gannon"]].values)

global_min_v, global_max_v = np.nanmin(stacked_values), np.nanmax(stacked_values)

offset = 1e-10
global_min = max(global_min_v, offset)  # Ensure global min is not zero

ax = carto(
    ax,
    label_titles={"V_100": "V"},
    df_tl=df_lines,
    cmap="magma",
    value_column="V_100",
    add_grid_regions=True,
    global_min=global_min_v,
    global_max=global_max_v,
)

plt.tight_layout()
plt.show()

gc.collect()


# %%
years = [100, 500, 1000]
field_names = {
    "gannon": "gannon_e",
    100: "e_field_100",
    500: "e_field_500",
    1000: "e_field_1000",
}

storm_titles = {
    "gannon": {
        "e_field": "Gannon Storm E-field Hazard Map",
        "v_field": "Gannon Storm Transmission Lines Hazard Map",
        "cmap": "magma",
    },
    100: {
        "e_field": "100-year E-field Hazard Map",
        "v_field": "100-year Transmission Lines Hazard Map",
        "cmap": "magma",
    },
    500: {
        "e_field": "500-year E-field Hazard Map",
        "v_field": "500-year Transmission Lines Hazard Map",
        "cmap": "magma",
    },
    1000: {
        "e_field": "1000-year E-field Hazard Map",
        "v_field": "1000-year Transmission Lines Hazard Map",
        "cmap": "magma",
    },
}


# figures path
figures_path = Path("figures")

# Set up the figure with GridSpec
projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
fig = plt.figure(figsize=(8, 10), dpi=300)
gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.3, hspace=0.1)

j = 0
for i, year in enumerate(field_names.keys()):

    ax_e = fig.add_subplot(gs[i, 0], projection=projection)

    # V title
    title_v = storm_titles[year]["v_field"]

    # E title
    title_e = storm_titles[year]["e_field"]

    # cmap
    cmap = storm_titles[year]["cmap"]

    data_e = read_pickle(data_loc / f"grid_e_{year}.pkl")

    ax_e = carto(
        ax_e,
        label_titles={field_names[year]: f"E-field (V/km)"},
        data_e=data_e,
        cmap=cmap,
        add_grid_regions=True,
        global_max=max_e_field,
        global_min=min_e_field,
    )

    ax_e.set_title(f"({alphabet[j]}) {title_e}", loc="left", fontsize=10)
    j += 1
    # Transmission lines plot
    ax_tl = fig.add_subplot(gs[i, 1], projection=projection)

    v_nodal_column = f"V_{year}"

    ax_tl = carto(
        ax_tl,
        label_titles={v_nodal_column: f"Voltage (V)\n (log scale)"},
        df_tl=df_lines,
        cmap=cmap,
        value_column=v_nodal_column,
        add_grid_regions=True,
        global_min=global_min_v,
        global_max=global_max_v,
    )
    ax_tl.set_title(
        f"({alphabet[j]}) {title_v}",
        loc="left",
        fontsize=10,
    )
    j += 1

    del v_nodal_column, title_e, title_v, ax_e, ax_tl  # Clear memory

plt.tight_layout()
plt.show()

# Save to figure
fig.savefig(figures_path / "hazard_maps.png", dpi=300, bbox_inches="tight")

plt.close(fig)


plt.clf()
plt.close("all")
gc.collect()


# %%
# --------------------------------------------------------------------------
# Plot the GICs for the transformers in the network
# --------------------------------------------------------------------------
def calculate_global_bins(all_gic_values, num_bins=5, threshold=10):
    """Calculate log-normalized bins from the minimum threshold to the maximum value."""
    abs_gic_values = np.abs(all_gic_values)
    filtered_values = abs_gic_values[abs_gic_values >= threshold]

    if len(filtered_values) == 0:
        logger.error("No values above the threshold.")
        return np.array([threshold, threshold])

    max_value = filtered_values.max()

    # Create log-spaced bins
    log_min = np.log10(threshold)
    log_max = np.log10(max_value)
    log_bins = np.logspace(log_min, log_max, num_bins)

    # Round the bin values for cleaner presentation
    bins = np.round(log_bins, 2)

    # Remove any duplicate bins that might occur due to rounding
    bins = np.unique(bins)

    return bins


def get_discrete_sizes_and_colors(
    gic_values, global_bins, cmap, min_size=10, max_size=500, threshold=10
):
    abs_gic_values = np.abs(gic_values)

    # Filter out values below threshold
    mask = abs_gic_values >= threshold
    filtered_values = abs_gic_values[mask]

    # Digitize values into bins
    bin_indices = np.digitize(filtered_values, global_bins) - 1

    # Define sizes using a linear scale
    size_range = np.logspace(np.log10(min_size), np.log10(max_size), len(global_bins))
    discrete_sizes = size_range[bin_indices]

    # Define colors using a perceptually uniform colormap
    cmap_obj = plt.get_cmap(cmap)
    color_range = np.linspace(0, 1, len(global_bins))
    discrete_colors = cmap_obj(color_range)[bin_indices]

    return discrete_sizes, discrete_colors, global_bins, mask


def create_legend_scatter(bins, cmap, min_size=10, max_size=500):

    num_bins = len(bins)
    y_pos = np.linspace(0, 1, num_bins)
    x_pos = np.full_like(y_pos, 0.5)
    sizes = np.logspace(np.log10(min_size), np.log10(max_size), len(bins))
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_bins))

    sizes = (sizes / max(bins)) * max_size  # Normalize sizes

    return x_pos, y_pos, sizes, colors


def format_with_commas(number):
    """Format a number with comma separators for thousands."""
    return f"{number:,}"


def create_gic_bubble_legend(
    ax, bins, cmap, title="GIC (A)", legend_width=30, norm_val=1
):
    fig = ax.figure
    fig_width, fig_height = fig.get_size_inches()

    # Convert legend_width from inches to figure-relative units
    legend_width_fig = legend_width / fig_width

    # Get the position of the main axes in figure coordinates
    bbox = ax.get_position()

    # Calculate the legend position
    legend_left = bbox.x1 + 0.01  # Small gap between plot and legend
    legend_bottom = bbox.y0
    legend_height = bbox.height

    # Create the legend axes using figure coordinates
    legend_ax = fig.add_axes(
        [legend_left, legend_bottom, legend_width_fig, legend_height]
    )

    x_pos, y_pos, sizes, colors = create_legend_scatter(bins, cmap)
    sizes = sizes * norm_val

    # Adjust y_pos to prevent clipping
    y_padding = 0.05  # 5% padding at top and bottom
    y_pos = y_pos * (1 - 2 * y_padding) + y_padding

    print(len(x_pos), len(y_pos), len(sizes), len(colors))

    # Plot the legend bubbles
    legend_ax.scatter(x_pos, y_pos, s=sizes, c=colors, alpha=0.8, edgecolors=colors)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_xticks([])
    legend_ax.yaxis.tick_right()

    # Set y-ticks to match bin locations
    y_ticks = y_pos
    legend_ax.set_yticks(y_ticks)

    # Format labels with commas
    y_labels = [format_with_commas(int(val)) for val in bins]
    legend_ax.set_yticklabels(y_labels, fontsize=7)
    legend_ax.set_title(title, fontsize=7, pad=5)

    return legend_ax


def plot_transformer_gic(
    ax,
    df_substations,
    quant_values,
    cmap,
    alpha=0.8,
    min_size=10,
    max_size=500,
    zorder=3,
    all_gic_values=None,
    num_bins=10,
    threshold=10,
    norm_val=2,
):

    if all_gic_values is not None:
        global_values = np.abs(all_gic_values)
    else:
        global_values = np.abs(quant_values)

    global_bins = calculate_global_bins(global_values, num_bins, threshold)

    plot_sizes, plot_colors, bins, mask = get_discrete_sizes_and_colors(
        quant_values,
        global_bins,
        cmap,
        min_size,
        max_size,
        threshold=10,
    )

    # Plot plot sizes
    plot_sizes = (
        (plot_sizes / max(global_bins)) * max_size * norm_val
    )  # Normalize sizes

    # Lat and lon values
    lons, lats = df_substations["longitude"][mask], df_substations["latitude"][mask]

    scatter = ax.scatter(
        lons,
        lats,
        transform=ccrs.PlateCarree(),
        c=plot_colors,
        s=plot_sizes,
        alpha=alpha,
        edgecolors=plot_colors,
        zorder=zorder,
    )

    norm = colors.Normalize(vmin=bins[0], vmax=bins[-1])

    return scatter, bins, plt.get_cmap(cmap), norm


# %%
# Cartopy plot for the transformers
def carto(
    ax,
    label_titles,
    spatial_extent=[-120, -75, 25, 50],
    add_grid_regions=True,
    df_tl=None,
    df_substations=None,
    cmap="viridis",
    value_column=None,
    show_legend=False,
    global_min=None,
    all_gic_values=None,
    global_max=None,
    legend_width=0.7,
    num_bins=10,
    norm_val=1,
):

    ax = setup_map(ax, spatial_extent)

    if add_grid_regions:
        ax = add_ferc_regions(ax)

    if df_tl is not None and value_column is not None:
        if value_column not in df_tl.columns:
            logger.error(f"Column '{value_column}' not found in df_tl")
        else:
            values = np.abs(df_tl[value_column].values)
            line_collection = plot_transmission_lines(
                ax, line_coordinates, values, cmap=cmap
            )
            ax.add_collection(line_collection)

            # Set colorbar
            vmin, vmax = values.min(), values.max()
            label_title = label_titles.get(value_column, value_column)

            if line_collection is not None:
                create_custom_colorbar_e_field(
                    ax, line_collection, label_title, label_title
                )
            else:
                logger.error("Failed to create line collection.")

    if df_substations is not None and value_column is not None:
        # _, _, global_bins = get_discrete_sizes_and_colors(gic_global_vals, cmap, num_bins=10)
        if value_column not in df_substations.columns:
            logger.error(f"Column '{value_column}' not found in df_substations")
        else:
            gic_values = df_substations[value_column].values

        scatter, bins, cmap_obj, norm = plot_transformer_gic(
            ax,
            df_substations,
            gic_values,
            all_gic_values=all_gic_values,
            cmap=cmap,
            num_bins=num_bins,
            norm_val=norm_val,
        )

        if show_legend:
            legend_ax = create_gic_bubble_legend(
                ax,
                bins,
                cmap,
                title="GIC (A/ph)",
                legend_width=legend_width,
                norm_val=norm_val,
            )
            return ax, legend_ax

    return ax


# Function to plot gICs for transformers
def create_gic_plots(
    df, spatial_extent=[-120, -75, 25, 50], cmap="YlOrRd", norm_val=2, storm_col=None
):

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    fig = plt.figure(figsize=(8, 8))  # Adjust figure size as needed
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.1, hspace=0.3)

    plot_order = [
        (1, 0, "Auto", "Common"),
        (1, 1, "Auto", "Series"),
        (2, 0, "GY-GY/GY-GY-D/GY-D", "HV"),
        (2, 1, "GY-GY/GY-GY-D", "LV"),
    ]

    titles = {
        "Common": "Autotransformers",
        "Series": "Autotransformers",
        "HV": "GY-GY-D, GY-GY, and GY-D Transformers.",
        "LV": "GY-GY-D and GY-GY Transformers.",
    }

    substitles = {
        "Common": "Common Winding GIC Estimates.",
        "Series": "Series Winding GIC Estimates.",
        "HV": "Primary Winding GIC",
        "LV": "Secondary Winding GIC",
    }

    all_gic_values = df[storm_col].values

    gic_global_vals = np.abs(np.array(all_gic_values))

    global_min = min(np.abs(all_gic_values))
    global_max = max(np.abs(all_gic_values))

    legend_width = 0.3

    for i, (row, col, _, winding) in enumerate(plot_order):

        ax = fig.add_subplot(gs[row, col], projection=projection)

        df_type = df_gic[df_gic["Winding"] == winding]

        ax_legend_ax = carto(
            ax,
            spatial_extent,
            df_substations=df_type,
            value_column=storm_col,
            cmap=cmap,
            global_min=global_min,
            global_max=global_max,
            all_gic_values=gic_global_vals,
            show_legend=(i % 2 != 0),
            legend_width=legend_width,
            num_bins=6,
            norm_val=norm_val,
        )
        title = f"({alphabet[i]}) {titles[winding]}"

        # ax.set_title(title, fontsize=9, loc='left')
        ax.text(0.01, 1.14, title, transform=ax.transAxes, fontsize=9)
        # Add subtitle
        ax.text(
            0.01, 1.03, f"{substitles[winding]}", transform=ax.transAxes, fontsize=8
        )

    plt.tight_layout()
    return fig


# # %%
# # Usage, starting with hallloween storm
# fig = create_gic_plots(df_gic, cmap="Reds", norm_val=0.2, storm_col="Hallloween A/ph")
# # File name
# file_name = figures_path / "gic_plots_halloween.png"
# # fig.suptitle("Halloween Storm", fontsize=12)
# fig.savefig(file_name, dpi=300, bbox_inches="tight")
# plt.show()

# %%
fig = create_gic_plots(
    df_gic, cmap="YlOrRd", norm_val=0.5, storm_col="100-year-hazard A/ph"
)
file_name = figures_path / "gic_plots_100.png"
# Add suptitle to the figure
# fig.suptitle("100 Year Hazard Maps", fontsize=12)
fig.savefig(file_name, dpi=300, bbox_inches="tight")
plt.show()

# %%
# Plot 500 year hazard maps
fig = create_gic_plots(df_gic, cmap="YlOrRd", norm_val=0.5, storm_col="500-year-hazard A/ph")
file_name = figures_path / "gic_plots_500.png"
# Add suptitle to the figure
# fig.suptitle("500 Year Hazard Maps", fontsize=12)
fig.savefig(file_name, dpi=300, bbox_inches="tight")
plt.show()

# %%
fig = create_gic_plots(df_gic, cmap="YlOrRd", norm_val=0.5, storm_col="1000-year-hazard A/ph")
file_name = figures_path / "gic_plots_1000.png"
fig.savefig(file_name, dpi=300, bbox_inches="tight")
plt.show()

# # %%
# # Plot for St. Patricks storm
# fig = create_gic_plots(df_gic, cmap="YlOrRd", norm_val=0.1, storm_col="St. Patricks A/ph")
# file_name = figures_path / "gic_plots_st_patricks.png"
# fig.savefig(file_name, dpi=300, bbox_inches="tight")
# plt.show()

# %%
# Plot for Gannon storm
fig = create_gic_plots(df_gic, cmap="YlOrRd", norm_val=0.1, storm_col="Gannon A/ph")
file_name = figures_path / "gic_plots_gannon.png"
fig.savefig(file_name, dpi=300, bbox_inches="tight")
plt.show()

# %%
