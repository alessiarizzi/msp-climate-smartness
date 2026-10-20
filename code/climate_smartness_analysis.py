"""
Climate-smart assessment of Marine Spatial Planning scenarios (3D)


What the script does
---------------------
1)  For each pilot area and depth biozone, it loads analogue-based climate velocity and classifies every grid cell (lat x lon x depth) by climate exposure:
    - horizontal: refugia, low, moderate-to-high, hotspot
    - vertical: no vertical shift (vz = 0) or vertical shift (vz != 0)
    - no analogue cells (no climate analogue found)
2)  It maps exposure: the dominant class along the water column, and each depth layer separately
3)  It overlays the maritime use polygons of each MSP scenario and computes the climate-smartness indicators:
    I1    mean horizontal velocity within each conservation use
    I2a   % of refugia covered by each conservation use
    I2b   % of hotspots covered by each conservation use
    I3a   % of other uses located in refugia
    I3b   % of other uses located in hotspots
    I4a   % of each nature-dependent use in low exposure
    I4b   % of each nature-dependent use in moderate-to-high or hotspot

Inputs (paths set in the CONFIGURATION section)
------
- one NetCDF per biozone with dimensions (depth, lat, lon) and variables
    delta_xy: horizontal distance to the climate analogue (km) over n_years (read as vel_xy; a file already using vel_xy also works)
    delta_z : vertical displacement to the climate analogue (m) over n_years
- one CSV per biozone listing no analogue cells (columns: depth, lat, lon)
- shapefiles: pilot areas, Mediterranean Sea polygon, Italian MSP baseline, and one file per MSP scenario. Use polygons need a "Use_NameIT" field: scenario files also need a "pilot_area" field

Here we use climate velocity results produced with the 3D climate velocity pipeline of Rizzi et al. (2026): https://github.com/alessiarizzi/3D-climate-change-velocity
(the files named {range}_dist_vel_{threshold}.nc and {range}_no_analogue_{threshold}.csv). 


Outputs (written to output_dir)
-------
- climate_smart_indicators.csv : indicators per pilot area x biozone x scenario
- biozone_stats.csv            : velocity statistics per pilot area x biozone  
- use_exposure_3d_summary.csv  : exposure of each use type
- PNG maps

"""

# IMPORTS
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from shapely.geometry import box


## CONFIGURATION ##
# Number of years over which delta_xy and delta_z are accumulated in the input files
n_years = 25

# Exposure thresholds
# Horizontal thresholds are tied to the grid-cell size, so they adapt automatically if n_years changes:
#   refugia          : velocity = 0 (the analogue is in the same cell)
#   low              : less than one grid cell over the whole period
#   moderate-to-high : between the two thresholds
#   hotspot          : at least one grid cell per year, or no analogue
cell_size_km      = 11.0                      # approx. size of a 0.1° cell
low_threshold     = cell_size_km / n_years    # km/yr (0.44 for 25 years)
hotspot_threshold = cell_size_km              # km/yr

# Annual Vz at or below which a cell counts as having no vertical shift
vz_tolerance = 1e-6


# Input data
# Climate velocity files per depth biozone
threshold       = 0.1
threshold_label = f"t{threshold}".replace(".", "")   # "t01"
 
biozone_files = {
    "euphotic":           f"data/outputs/0-50_dist_vel_{threshold_label}.nc",
    "mesophotic":         f"data/outputs/30-250_dist_vel_{threshold_label}.nc",
    "mesopelagic":        f"data/outputs/200-1250_dist_vel_{threshold_label}.nc",
    "bathyabyssopelagic": f"data/outputs/1000-5500_dist_vel_{threshold_label}.nc",
}

# Depth layers kept for each biozone (m; lower bound included, upper excluded).
depth_bounds = {
    "euphotic":           (0,    40),
    "mesophotic":         (40,   250),
    "mesopelagic":        (200,  1250),
    "bathyabyssopelagic": (1000, 5500),
}

no_analogue_files = {
    "euphotic":           f"data/outputs/0-50_no_analogue_{threshold_label}.csv",
    "mesophotic":         f"data/outputs/30-250_no_analogue_{threshold_label}.csv",
    "mesopelagic":        f"data/outputs/200-1250_no_analogue_{threshold_label}.csv",
    "bathyabyssopelagic": f"data/outputs/1000-5500_no_analogue_{threshold_label}.csv",
}

# Pilot areas 
pilot_areas = {
    "NAS": "data/shapefiles/NAS_pilot_area.shp",
    "NTS": "data/shapefiles/NTS_pilot_area.shp",
    "SoS": "data/shapefiles/SoS_pilot_area.shp",
}

# Map extent per pilot area (decimal degrees)
bbox_extents = {
    "NAS": {"lat_min": 42.657, "lat_max": 45.901, "lon_min": 11.47,  "lon_max": 17.036},
    "NTS": {"lat_min": 41.86,  "lat_max": 44.425, "lon_min": 6.98,   "lon_max": 11.975},
    "SoS": {"lat_min": 35.157, "lat_max": 38.401, "lon_min": 10.789, "lon_max": 16.069},
}

# Shapefiles
med_shapefile      = "data/shapefiles/med_sea.shp"
baseline_shapefile = "data/shapefiles/EMODnet_HA_MSP_Zoning_Element_pg_20251203.shp"
scenario_shapefile = "data/shapefiles/scenari_MSP4Biodiversity.shp"

# MSP scenarios
scenarios = ["SP", "NW", "BD"]
scenario_labels = {"SP": "SP", "NW": "N@W", "BD": "BD"}
biozones  = list(biozone_files)



# Maritime use groups
conservation_uses     = ["Protezione", "OECM"]     # OECM = fishing restrictions
nature_dependent_uses = ["Acquacoltura", "Turismo", "Pesca"]     
core_uses             = conservation_uses + nature_dependent_uses

# OECMs
oecm_patterns = [
    r"^FRA (esistente|nuova)",          # FRAs
    r"^ZTB",                            # ZTBs
    r"nursery",                         # nursery areas
    r"pesca artigianale",               # areas relevant for artisanal fishing
    r"\bMN\b",                          # coastal trawling bans (0-3 NM and their extensions to 4-6 NM)
]

# Depth limits of fishing
fishing_max_depth = 1000                # international ban below 1000 m, whole Mediterranean
fishing_depth_limits = {                # Pesca polygons (UseDescIT) with a deeper trawling ban
    "FRA divieto pesca a strascico 800-1000 m":     800,
    "FRA divieto pesca a strascico sotto i 1000 m": 1000,
}


# Baseline (EMODnet, approved Italian MSP)
baseline_selection = {
    "Protezione":   ["Reserved"],
    "Acquacoltura": ["Priority", "Potential"],
    "Turismo":      ["Priority"],
}

# Additional baselines
extra_baseline_files = {
    "Protezione":   "data/shapefiles/protezione_baseline.shp",
    "OECM":         "data/shapefiles/OECM_baseline.shp",
    "Acquacoltura": "data/shapefiles/acquacultura_baseline.shp",
}

# Scenario precedence: baseline use -> scenario uses that replace it where they overlap
baseline_overridden_by = {
    "Protezione":   ["Protezione"],
    "Acquacoltura": ["Acquacoltura"],
    "Turismo":      ["Turismo"],
}

# Fishing footprint: fishing is allowed everywhere where another use excludes it
fishing_ban_tokens = {"no_fishing"}
fishing_excluded_uses_scenario = ["Protezione", "OECM", "Acquacoltura"]    
fishing_excluded_uses_baseline = {                                         # EMODnet use -> SeaUseFct levels closed to fishing
    "Protezione": ["Reserved", "Priority"],                                 # existing MPAs
    "Difesa":     ["Reserved"],                                             # military areas
    "Acquacoltura": ["Priority", "Potential"],                              # aquaculture sites
}


# Cell assignment
# "centroid": a cell belongs to a use if its centroid falls within a polygon (weight 1)
# "area_weighted": a cell contributes the share of its sea area covered by the use (weight 0-1)
cell_assignment = "area_weighted"
area_crs        = "EPSG:3035"     # equal-area CRS used to compute overlap shares



# Processing and output settings
crs                   = "EPSG:4326"
area_buffer           = 0.01   # buffer around pilot areas (degrees)
no_analogue_tolerance = 0.01  # max distance to match no-analogue points to the grid (degrees / m)
map_pad               = 0.2    # margin around map extent (degrees)
show_figures          = False  # True to display figures while running
export_rasters       = True   # also save the exposure maps as GeoTIFF (for GIS)
raster_nodata        = 255    # no-data value of the GeoTIFFs
 
output_dir = Path("data/outputs/climate-smart_assessment_3d")
output_dir.mkdir(parents=True, exist_ok=True)



## EXPOSURE CLASSES ##
exposure_colors_3d = {
    0: "darkblue",         # refugia, vz = 0
    1: "royalblue",        # refugia, vz != 0
    2: "cornflowerblue",   # low, vz = 0
    3: "lightsteelblue",   # low, vz != 0
    4: "peachpuff",        # moderate-to-high, vz = 0
    5: "coral",            # moderate-to-high, vz != 0
    6: "red",              # hotspot, vz = 0
    7: "firebrick",        # hotspot, vz != 0
    8: "purple",           # no analogue
}
 
exposure_labels_3d = [
    "refugia (vz=0)",
    "refugia (vz≠0)",
    "low (vz=0)",
    "low (vz≠0)",
    "moderate-high (vz=0)",
    "moderate-high (vz≠0)",
    "hotspot (vz=0)",
    "hotspot (vz≠0)",
    "no analogue",
]
 
n_classes_3d      = len(exposure_labels_3d)
exposure_cmap_3d  = ListedColormap([exposure_colors_3d[i] for i in range(n_classes_3d)])
exposure_norm_3d  = plt.Normalize(vmin=-0.5, vmax=n_classes_3d - 0.5)  # centres each class in its colour band
exposure_ticks_3d = list(range(n_classes_3d))



## CLIMATE DATA AND CLASSIFICATION ##
def load_biozone_data(biozone):
    """
    Load the NetCDF dataset and no-analogue CSV for a given biozone, 
    restricted to its depth bounds.
    """
    ds = xr.open_dataset(biozone_files[biozone])
    # The 3D velocity pipeline writes the horizontal distance as delta_xy
    if "vel_xy" not in ds and "delta_xy" in ds:
        ds = ds.rename({"delta_xy": "vel_xy"})
    depth_min, depth_max = depth_bounds[biozone]

    depths     = ds.depth.values
    depth_mask = (depths >= depth_min) & (depths < depth_max)
    ds         = ds.sel(depth=depths[depth_mask])

    no_analogue_df = pd.read_csv(no_analogue_files[biozone])
    no_analogue_df.columns = no_analogue_df.columns.str.strip().str.lower()
    no_analogue_df = no_analogue_df[
        (no_analogue_df["depth"] >= depth_min) &
        (no_analogue_df["depth"] <  depth_max)
    ]

    return ds, no_analogue_df


def classify_horizontal(vel_xy_raw, no_analogue_mask):
    """
    Horizontal exposure class per cell:
    0=refugia, 1=low, 2=moderate-to-high, 3=hotspot, NaN=missing
    """
    vel     = vel_xy_raw / n_years   # km/yr
    classes = np.full(vel.shape, np.nan)
 
    classes[vel == 0.0]                                         = 0
    classes[(vel > 0.0) & (vel < low_threshold)]                = 1
    classes[(vel >= low_threshold) & (vel < hotspot_threshold)] = 2
    classes[(vel >= hotspot_threshold) | no_analogue_mask]      = 3
 
    return classes


def classify_vertical(delta_z_raw, no_analogue_mask):
    """
    Vertical class per cell: 
    0 no vertical shift, 1 vertical shift, NaN missing or no-analogue.
    """
    classes = np.full(delta_z_raw.shape, np.nan)
    has_value = ~np.isnan(delta_z_raw) & ~no_analogue_mask

    classes[has_value & (np.abs(delta_z_raw) <= vz_tolerance * n_years)] = 0
    classes[has_value & (np.abs(delta_z_raw) >  vz_tolerance * n_years)] = 1

    return classes


def combine_classes(horizontal_class, vertical_class, no_analogue_mask):
    """
    Combine horizontal (0-3) and vertical (0-1) classes into the 3D class (0-7), 
    with 8 for no analogue.
    """
    combined = np.full(horizontal_class.shape, np.nan)
 
    for h in range(4):
        for v in range(2):
            combined[(horizontal_class == h) & (vertical_class == v)] = h * 2 + v
 
    combined[no_analogue_mask] = 8
 
    return combined


def build_exposure_ds(biozone, pilot_area, area_gdf):
    """
    Build the masked and classified exposure dataset for one biozone and pilot area.
    """
    ds, no_analogue_df = load_biozone_data(biozone)

    lon    = ds.lon.values
    lat    = ds.lat.values
    depths = ds.depth.values

    def _as_da(arr):
        return xr.DataArray(
            arr,
            dims=["depth", "lat", "lon"],
            coords={"depth": depths, "lat": lat, "lon": lon}
        )
 
    # Spatial mask: grid cells inside the (buffered) pilot area, repeated over depth
    buffered_geom = area_gdf.unary_union.buffer(area_buffer)
    lon2d, lat2d  = np.meshgrid(lon, lat)
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()),
        crs=crs
    )
    mask_2d = points.within(buffered_geom).values.reshape(len(lat), len(lon))
    mask_3d = np.broadcast_to(mask_2d, (len(depths), *mask_2d.shape))

    # No analogue mask
    na_depth = no_analogue_df["depth"].to_numpy(dtype=float)
    na_lat   = no_analogue_df["lat"].to_numpy(dtype=float)
    na_lon   = no_analogue_df["lon"].to_numpy(dtype=float)
 
    i = np.abs(depths[np.newaxis, :] - na_depth[:, np.newaxis]).argmin(axis=1)
    j = np.abs(lat[np.newaxis, :]    - na_lat[:, np.newaxis]).argmin(axis=1)
    k = np.abs(lon[np.newaxis, :]    - na_lon[:, np.newaxis]).argmin(axis=1)
 
    matched = (
        np.isclose(depths[i], na_depth, atol=no_analogue_tolerance) &
        np.isclose(lat[j],    na_lat,   atol=no_analogue_tolerance) &
        np.isclose(lon[k],    na_lon,   atol=no_analogue_tolerance)
    )
 
    no_analogue_3d = np.zeros((len(depths), len(lat), len(lon)), dtype=bool)
    no_analogue_3d[i[matched], j[matched], k[matched]] = True
 
    # Keep cells inside the area that have a velocity value or are no-analogue
    valid_ocean = ~np.isnan(ds["vel_xy"].values) | no_analogue_3d
    final_mask  = mask_3d & valid_ocean
 
    ds_masked = ds.where(final_mask)
    ds_masked["vel_xy_km_yr"] = ds_masked["vel_xy"] / n_years
    ds_masked["delta_z_m_yr"] = ds_masked["delta_z"] / n_years
 
    no_analogue_final = no_analogue_3d & final_mask
 
    horizontal = classify_horizontal(ds_masked["vel_xy"].values, no_analogue_final)
    vertical   = classify_vertical(ds_masked["delta_z"].values, no_analogue_final)
    combined   = combine_classes(horizontal, vertical, no_analogue_final)
 
    ds_masked["no_analogue"]       = _as_da(no_analogue_final)
    ds_masked["horizontal_class"]  = _as_da(horizontal)
    ds_masked["vertical_class"]    = _as_da(vertical)
    ds_masked["exposure_class_3d"] = _as_da(combined)
 
    # Summary of horizontal classes (no-analogue cells are included in hotspots)
    h_names = {0: "refugia", 1: "low", 2: "moderate_high", 3: "hotspot"}
    total   = np.sum(~np.isnan(horizontal))
    print(f"\n[{biozone.upper()} — {pilot_area}]")
    print(f"  Total valid cells: {total}")
    for val, name in h_names.items():
        n   = np.sum(horizontal == val)
        pct = 100 * n / total if total > 0 else 0
        print(f"  {name:>15}: {n:>6} cells ({pct:.1f}%)")
    n_na = np.sum(no_analogue_final)
    print(f"  {'no_analogue':>15}: {n_na:>6} cells ({100*n_na/total:.1f}%)")
 
    return ds_masked


def compute_biozone_stats(ds_masked, biozone, pilot_area):
    """Mean, median and standard deviation of annual horizontal velocity and vertical displacement."""
    vel = ds_masked["vel_xy_km_yr"].values.ravel()
    dz  = ds_masked["delta_z_m_yr"].values.ravel()
 
    vel = vel[~np.isnan(vel)]
    dz  = dz[~np.isnan(dz)]
 
    stats = {
        "biozone":        biozone,
        "pilot_area":     pilot_area,
        "vel_xy_mean":    np.mean(vel),
        "vel_xy_median":  np.median(vel),
        "vel_xy_std":     np.std(vel),
        "delta_z_mean":   np.mean(dz),
        "delta_z_median": np.median(dz),
        "delta_z_std":    np.std(dz),
    }
 
    print(f"\n  [STATS] {biozone.upper()} — {pilot_area}")
    print(f"    vel_xy  : mean={stats['vel_xy_mean']:.3f}  "
          f"median={stats['vel_xy_median']:.3f}  std={stats['vel_xy_std']:.3f} km/yr")
    print(f"    delta_z : mean={stats['delta_z_mean']:.3f}  "
          f"median={stats['delta_z_median']:.3f}  std={stats['delta_z_std']:.3f} m/yr")
 
    return stats



## MAPS ##
def _format_ax(ax, area_gdf, med_gdf):
    """Draw land (grey background), sea and pilot area boundary."""
    ax.set_facecolor("#d9d9d9")
    med_gdf.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5, zorder=1)
    area_gdf.boundary.plot(ax=ax, color="black", linewidth=1, zorder=3)
 
 
def _apply_extent(ax, bbox):
    """Set the map extent to the pilot area bounding box plus a margin."""
    ax.set_xlim(bbox["lon_min"] - map_pad, bbox["lon_max"] + map_pad)
    ax.set_ylim(bbox["lat_min"] - map_pad, bbox["lat_max"] + map_pad)
 
 
def _apply_style(ax):
    """Set ticks, degree labels and grid (reapplied because xarray plotting resets them)."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f°"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f°"))
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, color="grey", linewidth=0.5, linestyle="--", alpha=0.7, zorder=2)
 
 
def plot_exposure_map(ds_masked, biozone, pilot_area, area_gdf, med_gdf):
    """Save the dominant-class map, one map per depth layer, and a multi-panel figure of all layers."""
    bbox      = bbox_extents[pilot_area]
    cls_3d_da = ds_masked["exposure_class_3d"]
    cls_3d    = cls_3d_da.values   # (depth, lat, lon)
 
    def _clip_to_area(da):
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs(crs)
        return da.rio.clip(area_gdf.geometry, area_gdf.crs, drop=False, all_touched=True)
 
    # Depth layers with at least one classified cell
    has_data           = ~np.all(np.isnan(cls_3d), axis=(1, 2))
    valid_depth_levels = list(ds_masked.depth.values[has_data])
 
    if not valid_depth_levels:
        print(f"  No valid depth layers for {biozone}/{pilot_area} — skipping maps.")
        return
 
    # Dominant class: most frequent 3D class along the water column
    # (on ties, the lowest class is kept)
    counts       = np.stack([(cls_3d == c).sum(axis=0) for c in range(n_classes_3d)])
    dominant_arr = np.where(counts.sum(axis=0) > 0, counts.argmax(axis=0), np.nan)
    dominant_da  = _clip_to_area(xr.DataArray(
        dominant_arr,
        dims=["lat", "lon"],
        coords={"lat": ds_masked.lat.values, "lon": ds_masked.lon.values}
    ))
 
    cls_3d_clipped = _clip_to_area(cls_3d_da)

    # Single maps: (data, title, title size, file name, display)
    single_maps = [(
        dominant_da, f"{biozone.upper()} / {pilot_area}", 13,
        f"exposure_dominant_3d_{biozone}_{pilot_area}.png", show_figures
    )]
    single_maps += [(
        cls_3d_clipped.sel(depth=depth), f"{depth} m", 16,
        f"exposure_3d_depth{int(depth)}m_{biozone}_{pilot_area}.png", False
    ) for depth in valid_depth_levels]
 
    for da, title, title_size, filename, show in single_maps:
        fig, ax = plt.subplots(figsize=(10, 8))
        _format_ax(ax, area_gdf, med_gdf)
        im = da.plot(
            ax=ax, cmap=exposure_cmap_3d, norm=exposure_norm_3d,
            add_colorbar=True, zorder=2,
            cbar_kwargs={"ticks": exposure_ticks_3d, "shrink": 0.7, "label": ""}
        )
        _apply_extent(ax, bbox)
        _apply_style(ax)
        im.colorbar.ax.set_yticklabels(exposure_labels_3d, fontsize=14)
        ax.set_title(title, fontsize=title_size)
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        if show:
            plt.show()
        plt.close(fig)
 
    # Multi-panel figure with all depth layers (max 4 columns)
    n_valid   = len(valid_depth_levels)
    ncols     = min(4, n_valid)
    nrows     = int(np.ceil(n_valid / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes      = np.array(axes).ravel()
 
    for ax, depth in zip(axes, valid_depth_levels):
        _format_ax(ax, area_gdf, med_gdf)
        cls_3d_clipped.sel(depth=depth).plot(
            ax=ax, cmap=exposure_cmap_3d, norm=exposure_norm_3d,
            add_colorbar=False, zorder=2
        )
        _apply_extent(ax, bbox)
        _apply_style(ax)
        ax.set_title(f"{depth} m", fontsize=16)
        ax.tick_params(labelsize=7)
 
    for ax in axes[n_valid:]:
        ax.set_visible(False)
 
    fig.suptitle(f"{biozone.upper()} / {pilot_area}", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f"exposure_3d_by_depth_{biozone}_{pilot_area}.png", dpi=300)
    if show_figures:
        plt.show()
    plt.close(fig)


def export_biozone_rasters(all_biozone_datasets, area_gdfs):
    """Save one GeoTIFF per biozone with the 3D exposure classes of all pilot areas, one band per depth layer."""
    for biozone in biozones:
        # Classes of each pilot area, clipped as in the maps, merged on the common grid
        combined = None
        for pilot_area, area_gdf in area_gdfs.items():
            da = (all_biozone_datasets[pilot_area][biozone]["exposure_class_3d"]
                  .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                  .rio.write_crs(crs)
                  .rio.clip(area_gdf.geometry, area_gdf.crs, drop=False, all_touched=True))
            combined = da if combined is None else combined.combine_first(da)
 
        # Depth layers with at least one classified cell
        has_data = ~np.all(np.isnan(combined.values), axis=(1, 2))
        if not has_data.any():
            print(f"  No valid depth layers for {biozone} — no raster saved.")
            continue
        layers = combined.isel(depth=np.where(has_data)[0])
 
        # 8-bit integer classes, north-up, no data = raster_nodata
        out = (layers.sortby("lat", ascending=False)
               .fillna(raster_nodata).astype("uint8")
               .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
               .rio.write_crs(crs))
        out.attrs, out.encoding = {}, {}
        out = out.rio.write_nodata(raster_nodata)
        out.attrs["classes"]   = "; ".join(f"{i}={label}" for i, label in enumerate(exposure_labels_3d))
        out.attrs["long_name"] = tuple(f"{depth:.1f} m" for depth in layers.depth.values)
        out.rio.to_raster(output_dir / f"exposure_3d_{biozone}.tif")
        print(f"  Saved exposure raster: {biozone} ({len(layers.depth)} depth layers)")



## MARITIME USES ## 
def harmonise_scenario_uses(scenario_gdf):
    """Pesca with fishing restrictions in Effect -> OECM; military areas -> Difesa."""
    gdf = scenario_gdf.copy()
    is_fishing = gdf["Use_NameIT"] == "Pesca"
    gdf.loc[is_fishing & gdf["Effect"].notna(), "Use_NameIT"] = "OECM"
    gdf.loc[gdf["SeaUseName"] == "Military areas", "Use_NameIT"] = "Difesa"
    return gdf    

def harmonise_emodnet_uses(emodnet_gdf):
    """Use_NameIT from OriginUN (names already match the scenario file)."""
    gdf = emodnet_gdf.copy()
    gdf["Use_NameIT"] = gdf["OriginUN"]
    return gdf

def load_extra_baselines():
    """Additional baseline polygons (MPAs, OECMs, aquaculture), with the use taken from the file."""
    parts = []
    for use, path in extra_baseline_files.items():
        gdf = gpd.read_file(path).to_crs(crs)
        gdf["geometry"] = gdf.make_valid()
        gdf["Use_NameIT"] = use
        parts.append(gdf[["Use_NameIT", "geometry"]])
    return pd.concat(parts, ignore_index=True)


def drop_duplicate_polygons(gdf):
    """Keep one copy of polygons with the same use and identical geometry."""
    wkb = gdf.geometry.to_wkb()
    return gdf[~gdf.assign(wkb=wkb).duplicated(["Use_NameIT", "wkb"])]


def has_fishing_ban(effect):
    """True if Effect contains a full fishing ban"""
    if not isinstance(effect, str):
        return False
    return any(token.strip() in fishing_ban_tokens for token in effect.split(","))


def select_baseline(baseline_gdf, selection, pilot_area_geom):
    """EMODnet polygons matching {use: [SeaUseFct levels]} that touch the pilot area."""
    parts = [
        baseline_gdf[(baseline_gdf["Use_NameIT"] == use) & baseline_gdf["SeaUseFct"].isin(levels)]
        for use, levels in selection.items()
    ]
    selected = pd.concat(parts, ignore_index=True)
    return selected[selected.intersects(pilot_area_geom)].copy()


def apply_scenario_precedence(baseline_use, scenario_gdf):
    """Remove from each baseline use the parts where the scenario assigns an overriding use."""
    parts = []
    for use, group in baseline_use.groupby("Use_NameIT"):
        overriding = scenario_gdf[scenario_gdf["Use_NameIT"].isin(baseline_overridden_by.get(use, [use]))]
        if not overriding.empty:
            group = group.copy()
            group["geometry"] = group.difference(overriding.unary_union)
            group = group[~group.is_empty]
        parts.append(group)
    if not parts:
        return baseline_use
    return pd.concat(parts, ignore_index=True)


def build_fishing_footprint(scenario_gdf, baseline_gdf, extra_baseline, pilot_area_geom):
    """Pilot area minus every area closed to fishing (scenario bans and uses, EMODnet MPAs and military areas, extra baselines)."""
    closed_scenario = scenario_gdf[
        scenario_gdf["Effect"].apply(has_fishing_ban)
        | scenario_gdf["Use_NameIT"].isin(fishing_excluded_uses_scenario)
    ]
    closed_baseline = select_baseline(baseline_gdf, fishing_excluded_uses_baseline, pilot_area_geom)
    closed = pd.concat([closed_scenario[["geometry"]], closed_baseline[["geometry"]],
                        extra_baseline[["geometry"]]], ignore_index=True)

    footprint = pilot_area_geom.difference(closed.unary_union) if not closed.empty else pilot_area_geom
    return gpd.GeoDataFrame(
        {"Use_NameIT": ["Pesca"], "source": ["fishing footprint"]}, geometry=[footprint], crs=crs
    )


def load_use_polygons(scenario_gdf, baseline_gdf, extra_baseline_gdf, area_gdf, pilot_area):
    """
    Use polygons of one scenario and pilot area: scenario uses (duplicates removed), plus the baseline
    (EMODnet + extra files) where the scenario does not assign an overriding use, plus the fishing footprint.
    """
    pilot_area_geom = area_gdf.unary_union

    scenario = scenario_gdf[scenario_gdf["pilot_area"] == pilot_area].copy()
    scenario = drop_duplicate_polygons(scenario)
    scenario["source"] = "scenario"
    scenario_uses = scenario[scenario["Use_NameIT"] != "Pesca"]   # fishing comes from the footprint

    # Baseline = EMODnet selection + extra baselines touching the pilot area
    extra    = extra_baseline_gdf[extra_baseline_gdf.intersects(pilot_area_geom)]
    baseline = select_baseline(baseline_gdf, baseline_selection, pilot_area_geom)
    baseline = pd.concat([baseline[["Use_NameIT", "geometry"]], extra], ignore_index=True)
    baseline = apply_scenario_precedence(baseline, scenario_uses)
    baseline["source"] = "baseline"

    fishing = build_fishing_footprint(scenario, baseline_gdf, extra, pilot_area_geom)

    cols = ["Use_NameIT", "source", "geometry"]
    return pd.concat([scenario_uses[cols], baseline[cols], fishing[cols]], ignore_index=True)


def ds_to_points_gdf(ds_masked):
    """
    Convert classified grid cells (one point per cell and depth layer) to a GeoDataFrame.
    """
    h_cls  = ds_masked["horizontal_class"]
    cls_3d = ds_masked["exposure_class_3d"]
    vel    = ds_masked["vel_xy_km_yr"]
    valid  = ~np.isnan(h_cls.values)

    depths_3d, lats_3d, lons_3d = np.meshgrid(
        h_cls.depth.values, h_cls.lat.values, h_cls.lon.values, indexing="ij"
    )

    df = pd.DataFrame({
        "depth":             depths_3d[valid],
        "lat":               lats_3d[valid],
        "lon":               lons_3d[valid],
        "exposure_class":    h_cls.values[valid],
        "exposure_class_3d": cls_3d.values[valid],
        "vel_xy_km_yr":      vel.values[valid],
    })

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=crs)


def build_cells(points_gdf, sea_geom=None):
    """
    One square per grid cell (unique lat/lon of the exposure points), in the equal-area CRS.
    ref_area is the denominator of the weights: the sea part of the cell if sea_geom is given, else the whole cell.
    """
    if points_gdf.empty:   # no valid cell in this biozone (e.g. no water deeper than 1000 m)
        return gpd.GeoDataFrame({"lat": [], "lon": [], "ref_area": [], "cell_id": []},
                                geometry=gpd.GeoSeries([], crs=area_crs), crs=area_crs)
        
    cells = points_gdf[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    lats, lons = np.unique(cells["lat"]), np.unique(cells["lon"])
    half_lat = np.diff(lats).min() / 2 if len(lats) > 1 else 0.05
    half_lon = np.diff(lons).min() / 2 if len(lons) > 1 else 0.05

    squares = [box(lo - half_lon, la - half_lat, lo + half_lon, la + half_lat)
               for la, lo in zip(cells["lat"], cells["lon"])]
    cells = gpd.GeoDataFrame(cells, geometry=squares, crs=crs)

    if sea_geom is not None:
        sea_local = sea_geom.intersection(box(*cells.total_bounds))
        sea_local = gpd.GeoSeries([sea_local], crs=crs).to_crs(area_crs).buffer(0).iloc[0]
        cells = cells.to_crs(area_crs)
        cells["ref_area"] = cells.intersection(sea_local).area
    else:
        cells = cells.to_crs(area_crs)
        cells["ref_area"] = cells.area

    cells["cell_id"] = np.arange(len(cells))
    return cells[cells["ref_area"] > 0]


def cell_weights(cells, use_gdf, by_polygon=False):
    """
    Share of each cell's reference area covered by each use (polygons of the same use dissolved,
    so overlaps count once) or, with by_polygon=True, by each polygon. Returns lat, lon, key(s), weight.
    """
    if cells.empty or use_gdf.empty:
        cols = ["cell_id", "lat", "lon", "Use_NameIT", "weight"] + (["polygon_id"] if by_polygon else [])
        return pd.DataFrame({c: pd.Series(dtype="object" if c == "Use_NameIT" else float) for c in cols})
        
    polys = use_gdf[["Use_NameIT", "geometry"]].to_crs(area_crs)
    polys["geometry"] = polys.buffer(0)
    if by_polygon:
        polys["polygon_id"] = use_gdf.index.values
        keys = ["polygon_id", "Use_NameIT"]
    else:
        polys = polys.dissolve(by="Use_NameIT").reset_index()
        keys = ["Use_NameIT"]

    inter = gpd.overlay(cells[["cell_id", "lat", "lon", "ref_area", "geometry"]],
                        polys[keys + ["geometry"]], how="intersection", keep_geom_type=True)
    inter["weight"] = inter.area / inter["ref_area"]
    weights = inter.groupby(["cell_id", "lat", "lon"] + keys, as_index=False)["weight"].sum()
    weights["weight"] = weights["weight"].clip(upper=1)
    return weights[weights["weight"] > 0]


def spatial_join_with_uses(points_gdf, use_gdf, cells=None):
    """
    Exposure points within each use, with a weight column: 1 per point (centroid) or the covered
    share of the cell (area_weighted). Each point is kept once per use, even under overlapping polygons.
    """
    if cell_assignment == "area_weighted":
        weights = cell_weights(cells, use_gdf)
        return (points_gdf.rename_axis("point_id").reset_index()
                .merge(weights[["lat", "lon", "Use_NameIT", "weight"]], on=["lat", "lon"])
                .set_index("point_id"))

    joined = gpd.sjoin(
        points_gdf,
        use_gdf[["Use_NameIT", "geometry"]],
        how="inner",
        predicate="within"
    ).drop(columns=["index_right"])

    joined["point_id"] = joined.index
    joined = joined.drop_duplicates(["point_id", "Use_NameIT"]).drop(columns="point_id")
    joined["weight"] = 1.0
    return joined


def join_cells_to_polygons(points_gdf, use_gdf, cells=None):
    """Exposure points within each polygon (for I1), with polygon_id and weight."""
    if cell_assignment == "area_weighted":
        weights = cell_weights(cells, use_gdf, by_polygon=True)
        return (points_gdf.rename_axis("point_id").reset_index()
                .merge(weights[["lat", "lon", "polygon_id", "weight"]], on=["lat", "lon"])
                .set_index("point_id"))

    joined = gpd.sjoin(points_gdf, use_gdf[["Use_NameIT", "geometry"]], how="inner", predicate="within")
    joined = joined.rename(columns={"index_right": "polygon_id"})
    joined["weight"] = 1.0
    return joined


def weighted_share(joined, mask):
    """Weighted % of joined cells where mask is True (NaN if no cells)."""
    total = joined["weight"].sum()
    return joined.loc[mask, "weight"].sum() / total * 100 if total > 0 else np.nan
    

## CLIMATE-SMART INDICATORS ##
def compute_indicators_ecosystems(points_gdf, gdf, scenario, pilot_area, biozone, cells=None):
    """
    Indicators I1-I3 for conservation uses and anthropogenic pressures.
    """
    results = {"biozone": biozone, "scenario": scenario, "pilot_area": pilot_area}
 
    pressures_gdf = gdf[~gdf["Use_NameIT"].isin(core_uses)]
 
    # Denominators for I2: all refugia and hotspot cells in the area
    n_refugia_total = (points_gdf["exposure_class"] == 0).sum()
    n_hotspot_total = (points_gdf["exposure_class"] == 3).sum()
 
    # --- I1, I1b, I2a, I2b: per conservation use ---
    for use_type in conservation_uses:
        use_gdf = gdf[gdf["Use_NameIT"] == use_type]
        use_key = use_type.lower()
 
        if use_gdf.empty:
            for name in ["mean_vel_per_polygon", "pct_no_analogue", "pct_refugia_protected", "pct_hotspot_protected"]:
                results[f"{name}_{use_key}"] = np.nan
            print(f"  [I1] No {use_type} found for {scenario}/{pilot_area}.")
            continue
 
        # I1, I1b: per polygon, then averaged across polygons (each polygon weighs the same)
        by_polygon = join_cells_to_polygons(points_gdf, use_gdf, cells)
 
        n_empty = len(use_gdf) - by_polygon["polygon_id"].nunique()
        if n_empty:
            print(f"  [{use_type}] {n_empty} of {len(use_gdf)} polygons contain no cell.")
 
        has_vel = by_polygon["vel_xy_km_yr"].notna()
        wv      = (by_polygon["weight"] * by_polygon["vel_xy_km_yr"])[has_vel]
        w_vel   = by_polygon["weight"][has_vel]
        poly_means = (wv.groupby(by_polygon.loc[has_vel, "polygon_id"]).sum()
                      / w_vel.groupby(by_polygon.loc[has_vel, "polygon_id"]).sum())
        results[f"mean_vel_per_polygon_{use_key}"] = poly_means.mean()
        print(f"  [I1] Mean vel ({use_type}): "
              f"{results[f'mean_vel_per_polygon_{use_key}']:.4f} km/yr")
 
        is_na = (by_polygon["exposure_class_3d"] == 8)
        w_na  = (by_polygon["weight"] * is_na).groupby(by_polygon["polygon_id"]).sum()
        w_all = by_polygon["weight"].groupby(by_polygon["polygon_id"]).sum()
        results[f"pct_no_analogue_{use_key}"] = (w_na / w_all * 100).mean()
        print(f"  [I1b] % no-analogue ({use_type}): "
              f"{results[f'pct_no_analogue_{use_key}']:.1f}%")
 
        # I2: covered share of all refugia / hotspot cells (each cell counted once per use)
        joined = spatial_join_with_uses(points_gdf, use_gdf, cells)
 
        w_refugia = joined.loc[joined["exposure_class"] == 0, "weight"].sum()
        results[f"pct_refugia_protected_{use_key}"] = (
            w_refugia / n_refugia_total * 100 if n_refugia_total > 0 else np.nan
        )
        print(f"  [I2a] % refugia protected ({use_type}): "
              f"{results[f'pct_refugia_protected_{use_key}']:.1f}%")
 
        w_hotspot = joined.loc[joined["exposure_class"] == 3, "weight"].sum()
        results[f"pct_hotspot_protected_{use_key}"] = (
            w_hotspot / n_hotspot_total * 100 if n_hotspot_total > 0 else np.nan
        )
        print(f"  [I2b] % hotspots protected ({use_type}): "
              f"{results[f'pct_hotspot_protected_{use_key}']:.1f}%")
 
    # --- I3: anthropogenic pressures (all non-conservation uses together) ---
    if pressures_gdf.empty:
        results["pct_pressures_refugia"] = np.nan
        results["pct_pressures_hotspot"] = np.nan
        print("  [I3] No anthropogenic pressures found.")
    else:
        joined_pressures = spatial_join_with_uses(points_gdf, pressures_gdf, cells)
 
        results["pct_pressures_refugia"] = weighted_share(joined_pressures, joined_pressures["exposure_class"] == 0)
        print(f"  [I3a] % anthropogenic pressures in refugia: {results['pct_pressures_refugia']:.1f}%")
 
        results["pct_pressures_hotspot"] = weighted_share(joined_pressures, joined_pressures["exposure_class"] == 3)
        print(f"  [I3b] % anthropogenic pressures in hotspots: {results['pct_pressures_hotspot']:.1f}%")
 
    return results
 
 
def compute_indicators_uses(points_gdf, gdf, scenario, pilot_area, biozone, cells=None):
    """Indicators I4 for nature-dependent uses."""
    results = {"biozone": biozone, "scenario": scenario, "pilot_area": pilot_area}
 
    for use_type in nature_dependent_uses:
        use_key = use_type.lower()
        use_gdf = gdf[gdf["Use_NameIT"] == use_type]
 
        if use_gdf.empty:
            results[f"pct_low_exposure_{use_key}"]     = np.nan
            results[f"pct_mod_high_hotspot_{use_key}"] = np.nan
            print(f"  [I4] No {use_type} found for {scenario}/{pilot_area}.")
            continue
 
        joined = spatial_join_with_uses(points_gdf, use_gdf, cells)
 
        # I4a: % of use cells in low exposure
        results[f"pct_low_exposure_{use_key}"] = weighted_share(joined, joined["exposure_class"] == 1)
        print(f"  [I4a] % {use_type} in low exposure: "
              f"{results[f'pct_low_exposure_{use_key}']:.1f}%")
 
        # I4b: % of use cells in moderate-to-high exposure or hotspots
        results[f"pct_mod_high_hotspot_{use_key}"] = weighted_share(joined, joined["exposure_class"].isin([2, 3]))
        print(f"  [I4b] % {use_type} in mod-high/hotspot: "
              f"{results[f'pct_mod_high_hotspot_{use_key}']:.1f}%")
 
    return results



## USE EXPOSURE SUMMARY ##
exposure_use_groups = {
    "MPAs":        ["Protezione"],
    "OECMs":       ["OECM"],
    "Aquaculture": ["Acquacoltura"],
    "Tourism":     ["Turismo"],
    "Fishing":     ["Pesca"],
}
pressures_group_name = "Anthropogenic pressures"
 
 
def collect_use_exposure(all_biozone_datasets, all_use_gdfs, sea_geom=None):
    """3D exposure class and weight of every cell in each use group, per pilot area, biozone and scenario."""
    parts = []
 
    for pilot_area, biozone_datasets in all_biozone_datasets.items():
        for biozone, ds_masked in biozone_datasets.items():
            if np.all(np.isnan(ds_masked["horizontal_class"].values)):
                continue
 
            points_gdf = ds_to_points_gdf(ds_masked)
            cells = build_cells(points_gdf, sea_geom) if cell_assignment == "area_weighted" else None
 
            for scenario in scenarios:
                gdf = all_use_gdfs[pilot_area][scenario]
 
                pressure_uses = gdf.loc[~gdf["Use_NameIT"].isin(core_uses), "Use_NameIT"].unique().tolist()
                groups = {**exposure_use_groups, pressures_group_name: pressure_uses}
 
                for group_name, use_types in groups.items():
                    use_gdf = gdf[gdf["Use_NameIT"].isin(use_types)]
                    if use_gdf.empty:
                        continue
 
                    joined = spatial_join_with_uses(points_gdf, use_gdf, cells)
                    joined = joined.dropna(subset=["exposure_class_3d"])
                    if joined.empty:
                        continue
 
                    parts.append(pd.DataFrame({
                        "pilot_area":        pilot_area,
                        "biozone":           biozone,
                        "scenario":          scenario,
                        "use":               group_name,
                        "exposure_class_3d": joined["exposure_class_3d"].astype(int).to_numpy(),
                        "weight":            joined["weight"].to_numpy(),
                    }))
 
    if not parts:
        return pd.DataFrame(columns=["pilot_area", "biozone", "scenario", "use", "exposure_class_3d", "weight"])
    return pd.concat(parts, ignore_index=True)
 
 
def plot_use_exposure(use_exposure_df, output_dir):
    """
    For each use group, plot a figure with rows=valid biozones,
    columns=scenarios, each subplot showing a stacked bar per case study area
    with the proportion of cells in each of the nine 3D exposure classes.
    Also returns the same percentages used in the plots.
    """
    use_names = list(exposure_use_groups) + [pressures_group_name]
 
    summary_rows = []
 
    for use_name in use_names:
        df_use = use_exposure_df[use_exposure_df["use"] == use_name]
 
        if df_use.empty:
            print(f"  No data for {use_name} — skipping.")
            continue
 
        valid_biozones = [b for b in biozones if not df_use[df_use["biozone"] == b].empty]
        if not valid_biozones:
            continue
 
        nrows = len(valid_biozones)
        ncols = len(scenarios)
 
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            sharey=True, sharex=True
        )
        axes = np.array(axes).reshape(nrows, ncols)
 
        for row_idx, biozone in enumerate(valid_biozones):
            df_bz = df_use[df_use["biozone"] == biozone]
 
            for col_idx, scenario in enumerate(scenarios):
                ax      = axes[row_idx, col_idx]
                df_scen = df_bz[df_bz["scenario"] == scenario]
 
                x         = np.arange(len(pilot_areas))
                bar_width = 0.6
                bottom    = np.zeros(len(pilot_areas))
 
                # Nine classes stacked in order (0-8)
                for cls_int in range(9):
                    proportions = []
                    for pilot_area in pilot_areas:
                        df_area = df_scen[df_scen["pilot_area"] == pilot_area]
                        w_total = df_area["weight"].sum()
                        pct = 100 * df_area.loc[df_area["exposure_class_3d"] == cls_int, "weight"].sum() / w_total if w_total else 0
                        proportions.append(pct)
 
                        summary_rows.append({
                            "use":               use_name,
                            "biozone":           biozone,
                            "scenario":          scenario,
                            "pilot_area":        pilot_area,
                            "exposure_class_3d": cls_int,
                            "exposure_label":    exposure_labels_3d[cls_int],
                            "percentage":        round(pct, 2),
                        })
 
                    ax.bar(x, proportions, bar_width,
                           bottom=bottom,
                           color=exposure_colors_3d[cls_int],
                           label=exposure_labels_3d[cls_int],
                           zorder=2)
                    bottom += np.array(proportions)
 
                ax.set_xticks(x)
                ax.set_xticklabels(list(pilot_areas.keys()), fontsize=18)
                ax.set_ylim(0, 100)
                ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
 
                if row_idx == 0:
                    ax.set_title(scenario_labels.get(scenario, scenario), fontsize=18, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{biozone}\n% of cells", fontsize=18)
 
        handles = [
            mpatches.Patch(color=exposure_colors_3d[i], label=exposure_labels_3d[i])
            for i in range(9)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=16,
                   frameon=False, bbox_to_anchor=(0.5, -0.05))
 
        fig.suptitle(use_name, fontsize=18, fontweight="bold")
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(output_dir / f"use_exposure_3d_{use_name.lower().replace(' ', '_')}.png", dpi=300)
        if show_figures:
            plt.show()
        plt.close(fig)
 
    return pd.DataFrame(summary_rows)
    

    
### MAIN ###
def run_full_assessment():
    """Run exposure classification, maps and indicators for every pilot area x biozone x scenario."""
    # Shared vector data, read once
    med_gdf = gpd.read_file(med_shapefile).to_crs(crs)
    sea_geom = med_gdf.unary_union   # sea area used as denominator of the cell weights
 
    # MSP scenarios: one shapefile, split by the Scenario column
    scenarios_gdf = gpd.read_file(scenario_shapefile, encoding="ISO-8859-1").to_crs(crs)
    scenarios_gdf["geometry"] = scenarios_gdf.make_valid()
    scenarios_gdf = harmonise_scenario_uses(scenarios_gdf)
    scenario_gdfs = {s: scenarios_gdf[scenarios_gdf["Scenario"] == s] for s in scenarios}
 
    # Baseline: approved Italian MSP from EMODnet
    baseline_gdf = gpd.read_file(baseline_shapefile, where="MS = 'Italy'").to_crs(crs)
    baseline_gdf["geometry"] = baseline_gdf.make_valid()
    baseline_gdf = harmonise_emodnet_uses(baseline_gdf)
    extra_baseline_gdf = load_extra_baselines()    
 
    all_indicators       = []
    all_stats            = []
    all_biozone_datasets = {}
    all_use_gdfs         = {}
    area_gdfs            = {}
 
    for pilot_area, area_shapefile in pilot_areas.items():
        print(f"\n{'='*60}\nPILOT AREA: {pilot_area}\n{'='*60}")
 
        area_gdf = gpd.read_file(area_shapefile).to_crs(crs)
        area_gdfs[pilot_area] = area_gdf
 
        # Use polygons depend only on pilot area and scenario
        use_gdfs = {
            s: load_use_polygons(scenario_gdfs[s], baseline_gdf, extra_baseline_gdf, area_gdf, pilot_area)
            for s in scenarios
        }
 
        # Optional: save the use polygons to check them in QGIS (one layer per scenario)
        for s, use_gdf in use_gdfs.items():
            use_gdf.to_file(output_dir / f"use_polygons_{pilot_area}.gpkg", layer=s)
 
        biozone_datasets = {}
 
        for biozone in biozones:
            print(f"\n  BIOZONE: {biozone.upper()}")
 
            # Exposure depends only on pilot area and biozone
            ds_masked                 = build_exposure_ds(biozone, pilot_area, area_gdf)
            biozone_datasets[biozone] = ds_masked
 
            all_stats.append(compute_biozone_stats(ds_masked, biozone, pilot_area))
            plot_exposure_map(ds_masked, biozone, pilot_area, area_gdf, med_gdf)
 
            points_gdf = ds_to_points_gdf(ds_masked)
            cells = build_cells(points_gdf, sea_geom) if cell_assignment == "area_weighted" else None
 
            for scenario in scenarios:
                print(f"\n    SCENARIO: {scenario}")
                gdf  = use_gdfs[scenario]
                eco  = compute_indicators_ecosystems(points_gdf, gdf, scenario, pilot_area, biozone, cells)
                uses = compute_indicators_uses(points_gdf, gdf, scenario, pilot_area, biozone, cells)
                all_indicators.append({**eco, **uses})
 
        all_biozone_datasets[pilot_area] = biozone_datasets
        all_use_gdfs[pilot_area]         = use_gdfs
 
    # Exposure rasters: one per biozone, all pilot areas together
    if export_rasters:
        export_biozone_rasters(all_biozone_datasets, area_gdfs)
 
    # Use exposure summary across all pilot areas
    print("\n  Collecting use exposure data...")
    use_exposure_df = collect_use_exposure(all_biozone_datasets, all_use_gdfs, sea_geom)
    summary_df      = plot_use_exposure(use_exposure_df, output_dir)
    summary_df.to_csv(output_dir / "use_exposure_3d_summary.csv", index=False)
 
    indicators_df = pd.DataFrame(all_indicators)
    stats_df      = pd.DataFrame(all_stats)
 
    id_cols       = ["pilot_area", "biozone", "scenario"]
    indicators_df = indicators_df[id_cols + [c for c in indicators_df.columns if c not in id_cols]]
 
    indicators_df.to_csv(output_dir / "climate_smart_indicators.csv", index=False)
    stats_df.to_csv(output_dir / "biozone_stats.csv", index=False)
 
    print(f"\nResults saved to {output_dir}")
    return indicators_df, stats_df
 
 



## INDICATORS SCORECARDS
# Layout shared by both scorecards: rows = biozones (shallow to deep), upper x axis = indicators,
# lower x axis = scenarios within each indicator.
 
# I1: mean velocity per conservation use, coloured by exposure class as in the maps
velocity_scorecard = {"protezione": "MPAs", "oecm": "OECMs"}
vel_class_colors   = ["royalblue", "lightsteelblue", "peachpuff", "red"]   # refugia, low, mod-high, hotspot
vel_class_labels   = ["refugia (0)", f"low (<{low_threshold:.2f})",
                      f"moderate-high (<{hotspot_threshold:.0f})", f"hotspot (≥{hotspot_threshold:.0f})"]
 
# I2-I4: column: (label, direction, decimals)
# direction: +1 if higher values are more climate-smart, -1 if lower values are
indicator_scorecard = {
    "pct_refugia_protected_protezione":  ("I2a Refugia in MPAs (%)",                    +1, 1),
    "pct_hotspot_protected_protezione":  ("I2b Hotspots in MPAs (%)",                   +1, 1),
    "pct_refugia_protected_oecm":        ("I2a Refugia in OECMs (%)",                   +1, 1),
    "pct_hotspot_protected_oecm":        ("I2b Hotspots in OECMs (%)",                  +1, 1),
    "pct_pressures_refugia":             ("I3a Anthropogenic pressures in refugia (%)", -1, 1),
    "pct_pressures_hotspot":             ("I3b Anthropogenic pressures in hotspots (%)",-1, 1),
    "pct_low_exposure_acquacoltura":     ("I4a Aquaculture in low exposure (%)",        +1, 1),
    "pct_mod_high_hotspot_acquacoltura": ("I4b Aquaculture in mod-high/hotspot (%)",    -1, 1),
    "pct_low_exposure_turismo":          ("I4a Tourism in low exposure (%)",            +1, 1),
    "pct_mod_high_hotspot_turismo":      ("I4b Tourism in mod-high/hotspot (%)",        -1, 1),
    "pct_low_exposure_pesca":            ("I4a Fishing in low exposure (%)",            +1, 1),
    "pct_mod_high_hotspot_pesca":        ("I4b Fishing in mod-high/hotspot (%)",        -1, 1),
}
 
scorecard_cmap = "PiYG"     # purple-pink = poor, green = good absolute level (colourblind-safe)
no_data_color  = "#eeeeee"
 
# Absolute classes for percentages: 0 | <5 | <10 | <25 | <50 | <75 | <90 | <95 | <100 | 100
pct_class_edges  = [5, 10, 25, 50, 75, 90, 95]
pct_class_labels = ["0", "<5", "5–10", "10–25", "25–50", "50–75", "75–90", "90–95", ">95", "100"]
 
 
def pct_class(v):
    """Index 0-9 of a percentage in pct_class_labels."""
    if v <= 0:
        return 0
    if v >= 100:
        return 9
    return int(np.searchsorted(pct_class_edges, v, side="right")) + 1
 
 
def vel_class(v):
    """Index 0-3 of a velocity (km/yr) in vel_class_labels."""
    if v <= 0:
        return 0
    if v < low_threshold:
        return 1
    if v < hotspot_threshold:
        return 2
    return 3
 
 
def absolute_level(v, sign):
    """Absolute level of a percentage on a 0 (poor) - 1 (good) scale, from its class."""
    if np.isnan(v):
        return np.nan
    level = pct_class(v) / (len(pct_class_labels) - 1)
    return level if sign > 0 else 1 - level
 
 
def _text_color(rgba):
    """Black or white, whichever reads better on the cell colour."""
    r, g, b = rgba[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
 
 
def _format_pct(v, decimals):
    """Value as written in the cell: '–' if missing, no decimals for 100."""
    if np.isnan(v):
        return "–"
    return "100" if v >= 100 else f"{v:.{decimals}f}"
 
 
def _scenario_values(indicators_df, pilot_area, biozone, column):
    """Values of one indicator for one pilot area and biozone, in scenario order."""
    sub = indicators_df[(indicators_df["pilot_area"] == pilot_area) & (indicators_df["biozone"] == biozone)]
    return sub.set_index("scenario").reindex(scenarios)[column].to_numpy(dtype=float)
 
 
def _best_scenarios(values, sign):
    """True for the best scenario(s), ties included; none if fewer than two scenarios have a value."""
    oriented = values * sign
    if np.sum(~np.isnan(oriented)) < 2:
        return np.zeros(len(values), dtype=bool)
    return np.isclose(oriented, np.nanmax(oriented)) & ~np.isnan(oriented)
 
 
# Cell geometry: gaps (in cell units) between indicators, biozones and scenarios
group_gap, row_gap, cell_gap = 0.35, 0.12, 0.05
 
 
def _xpos(c):
    """x position of column c, with a gap after every indicator group."""
    return c + (c // len(scenarios)) * group_gap
 
 
def _ypos(r):
    """y position of row r."""
    return r * (1 + row_gap)
 
 
def _frame_cell(ax, r, c):
    """Black frame around a cell (best scenario)."""
    w, h = 1 - cell_gap, 1
    ax.add_patch(plt.Rectangle((_xpos(c) - w / 2, _ypos(r) - h / 2), w, h, fill=False,
                               edgecolor="black", linewidth=3, zorder=4))
 
 
def _draw_scorecard(ax, colors, group_labels):
    """Draw the cells (biozone x indicator/scenario) and the shared axes layout."""
    n_s    = len(scenarios)
    n_rows = len(biozones)
    n_cols = len(group_labels) * n_s
    w, h   = 1 - cell_gap, 1
 
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle((_xpos(c) - w / 2, _ypos(r) - h / 2), w, h,
                                       facecolor=colors[r, c], edgecolor="none", zorder=1))
    ax.set_xlim(_xpos(0) - 0.6, _xpos(n_cols - 1) + 0.6)
    ax.set_ylim(_ypos(n_rows - 1) + 0.6, _ypos(0) - 0.6)
 
    # Lower x axis: scenarios; upper x axis: indicators, centred on their scenarios
    ax.set_xticks([_xpos(c) for c in range(n_cols)])
    ax.set_xticklabels([scenario_labels[s] for s in scenarios] * len(group_labels), fontsize=13)
    top = ax.secondary_xaxis("top")
    top.set_xticks([_xpos(g * n_s) + (n_s - 1) / 2 for g in range(len(group_labels))])
    top.set_xticklabels(group_labels, rotation=45, ha="left", rotation_mode="anchor", fontsize=14)
    top.tick_params(length=0)
    top.spines["top"].set_visible(False)
 
    # y axis: biozones from shallow to deep
    ax.set_yticks([_ypos(r) for r in range(n_rows)])
    ax.set_yticklabels([f"{b}\n{depth_bounds[b][0]}–{depth_bounds[b][1]} m" for b in biozones], fontsize=14)
 
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
 
 
def plot_velocity_scorecard(indicators_df):
    """
    I1 per pilot area: mean velocity per conservation use, coloured by exposure class,
    with the % of no-analogue cells written below each value; framed = lowest velocity.
    """
    n_s, uses = len(scenarios), list(velocity_scorecard)
    no_data   = np.array(plt.matplotlib.colors.to_rgba(no_data_color))
 
    for pilot_area in pilot_areas:
        colors = np.tile(no_data, (len(biozones), len(uses) * n_s, 1))
        fig, ax = plt.subplots(figsize=(1.15 * len(uses) * n_s + 3, 1.1 * len(biozones) + 3),
                               constrained_layout=True)
 
        for r, biozone in enumerate(biozones):
            for g, use_key in enumerate(uses):
                vel  = _scenario_values(indicators_df, pilot_area, biozone, f"mean_vel_per_polygon_{use_key}").round(2)
                na   = _scenario_values(indicators_df, pilot_area, biozone, f"pct_no_analogue_{use_key}")
                best = _best_scenarios(vel, -1)
                for s in range(n_s):
                    c = g * n_s + s
                    if np.isnan(vel[s]):
                        ax.text(_xpos(c), _ypos(r), "–", ha="center", va="center", fontsize=14)
                        continue
                    colors[r, c] = plt.matplotlib.colors.to_rgba(vel_class_colors[vel_class(vel[s])])
                    tc = _text_color(colors[r, c])
                    ax.text(_xpos(c), _ypos(r) - 0.12, f"{vel[s]:.2f}", ha="center", va="center",
                            fontsize=19, fontweight="bold", color=tc, zorder=5)
                    if not np.isnan(na[s]):
                        ax.text(_xpos(c), _ypos(r) + 0.22, f"NA {na[s]:.1f}%", ha="center", va="center",
                                fontsize=14, color=tc, zorder=5)
                    if best[s]:
                        _frame_cell(ax, r, c)
 
        _draw_scorecard(ax, colors, list(velocity_scorecard.values()))
 
        handles  = [mpatches.Patch(color=col, label=lab) for col, lab in zip(vel_class_colors, vel_class_labels)]
        handles += [mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.5, label="best scenario")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=13, frameon=False,
                   title="Mean velocity (km/yr); NA = % of no-analogue cells", title_fontsize=13)
 
        fig.suptitle(f"{pilot_area} – I1 Mean velocity in conservation areas", fontsize=20, fontweight="bold")
        fig.savefig(output_dir / f"indicator_scorecard_I1_{pilot_area}.png", dpi=300, bbox_inches="tight")
        if show_figures:
            plt.show()
        plt.close(fig)
        print(f"  Saved I1 scorecard: {pilot_area}")
 
 
def plot_indicator_scorecard(indicators_df, prefix, title, name):
    """
    Scorecard per pilot area for the indicators whose label starts with one of prefix (e.g. ("I2", "I3")):
    colour = absolute level of each value (class-based, purple poor / green good), text = value,
    framed = best scenario per indicator and biozone.
    """
    cols  = [c for c in indicator_scorecard
             if c in indicators_df.columns and indicator_scorecard[c][0].startswith(prefix)]
    n_s   = len(scenarios)
    cmap  = plt.get_cmap(scorecard_cmap)
    no_data = np.array(plt.matplotlib.colors.to_rgba(no_data_color))
 
    for pilot_area in pilot_areas:
        colors = np.tile(no_data, (len(biozones), len(cols) * n_s, 1))
        fig, ax = plt.subplots(figsize=(0.95 * len(cols) * n_s + 3, 1.0 * len(biozones) + 5.5),
                               constrained_layout=True)
 
        for r, biozone in enumerate(biozones):
            for g, col in enumerate(cols):
                _, sign, d = indicator_scorecard[col]
                values = _scenario_values(indicators_df, pilot_area, biozone, col).round(d)
                best   = _best_scenarios(values, sign)
                for s, v in enumerate(values):
                    c = g * n_s + s
                    if not np.isnan(v):
                        colors[r, c] = cmap(absolute_level(v, sign))
                    ax.text(_xpos(c), _ypos(r), _format_pct(v, d), ha="center", va="center",
                            fontsize=17, fontweight="bold", color=_text_color(colors[r, c]), zorder=5)
                    if best[s]:
                        _frame_cell(ax, r, c)
 
        _draw_scorecard(ax, colors, [indicator_scorecard[c][0] for c in cols])
 
        # Continuous colour bar from poor (purple) to good (green) performance
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.6, aspect=25, pad=0.02)
        cbar.set_ticks([0, 1], labels=["less climate-smart", "more climate-smart"], fontsize=14)
        cbar.ax.tick_params(length=0)
        cbar.outline.set_visible(False)
 
        frame = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.5, label="best scenario")
        fig.legend(handles=[frame], loc="outside lower right", fontsize=14, frameon=False)
 
        fig.suptitle(f"{pilot_area} – {title}", fontsize=24, fontweight="bold")
        fig.savefig(output_dir / f"indicator_scorecard_{name}_{pilot_area}.png", dpi=300, bbox_inches="tight")
        if show_figures:
            plt.show()
        plt.close(fig)
        print(f"  Saved {name} scorecard: {pilot_area}")


## MAIN ##
if __name__ == "__main__":
    indicators_df, stats_df = run_full_assessment()

    # Scorecards
    plot_velocity_scorecard(indicators_df)
    plot_indicator_scorecard(indicators_df, ("I2", "I3"), "I2–I3 Conservation areas and pressures", "I2-I3")
    plot_indicator_scorecard(indicators_df, ("I4",), "I4 Nature-dependent uses", "I4")
