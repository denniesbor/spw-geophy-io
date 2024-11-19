def viz_voltages():
    
    # Convert back the crs of tl_gdf
    tl_gdf_not_na = tl_gdf_not_na.to_crs(pjm_gdf.crs)

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.27, 8.27), dpi=300)  # 1 row, 2 columns

    # Adjust the colormap and the normalization
    cmap = plt.cm.magma
    vmin, vmax = np.min(tl_gdf_not_na["voltage"]), np.max(tl_gdf_not_na["voltage"])
    normV = colors.LogNorm(vmin=vmin, vmax=vmax)
    vmin, vmax = np.min(tl_gdf_not_na["E_field"]), np.max(tl_gdf_not_na["E_field"])
    normE = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap.set_under('#cccccc')  # Set under-color for out-of-range low values


    tl_gdf_not_na.to_crs(epsg=3857).plot(ax=ax1, column='voltage', cmap=cmap, linewidth=0.6, norm=normV, alpha=0.8)
    pjm_gdf.plot(ax=ax1, edgecolor='blue', facecolor="none", zorder=2, linewidth=0.8, alpha=0.8)
    ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron, attribution="")
    ax1.set_title('(A) Voltage Distribution during the Mid-May 2024 Storm', fontdict={'fontsize': 8})
    ax_cbar = inset_axes(ax1,
                        width=0.7,
                        height=0.15,
                        loc='lower right',
                        bbox_to_anchor=(0.15, 0.1, 0.11, 0.09),
                        bbox_transform=ax1.transAxes,
                        borderpad=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normV)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cbar.ax.set_title("Voltage (V)", fontsize=6, pad=3, loc='left')
    cbar.ax.tick_params(labelsize=6)
    ax1.axis("off")

    tl_gdf_not_na.to_crs(epsg=3857).plot(ax=ax2, column='E_field', cmap=cmap, linewidth=0.6, norm=normE, alpha=0.8)
    pjm_gdf.plot(ax=ax2, edgecolor='blue', facecolor="none", zorder=2, linewidth=0.8, alpha=0.8)
    ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron, attribution="")
    ax2.set_title('(B) E Field Distribution during the Mid-May 2024 Storm', fontdict={'fontsize': 8})
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normE)
    ax_cbar_1 = inset_axes(ax2,
                        width=0.7,
                        height=0.15,
                        loc='lower right',
                        bbox_to_anchor=(0.15, 0.1, 0.11, 0.09),
                        bbox_transform=ax2.transAxes,
                        borderpad=0)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar_1, orientation='horizontal')
    cbar.ax.set_title("E Field (V/km)", fontsize=6, pad=3, loc='left')
    cbar.ax.tick_params(labelsize=6)
    ax2.axis('off')

    # Show the plot
    plt.subplots_adjust(top=0.85)
    plt.show()