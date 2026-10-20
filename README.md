# Climate-smart MSP assessment

Code to assess how **climate-smart** Marine Spatial Planning (MSP) scenarios are, using 3D analogue-based climate velocity. Applied to three Italian pilot areas within the NBFC-MSP4Biodiversity project.

> Paper in preparation. 

\---

## Overview

The script classifies every grid cell (depth × lat × lon) of each pilot area by **climate exposure**, then overlays the maritime uses of each MSP scenario and computes a set of climate-smartness indicators.

**Exposure classes**

|Horizontal velocity|Definition|
|-|-|
|Refugia|0 km/yr (analogue in the same cell)|
|Low|less than one grid cell over the period|
|Moderate-to-high|between low and hotspot|
|Hotspot|at least one grid cell per year, or no analogue|

Each class is further split into cells with and without a vertical shift (vz = 0 / vz ≠ 0).

**Indicators**

|Indicator|Description|
|-|-|
|I1|Mean horizontal velocity within each conservation use|
|I2a / I2b|% of refugia / hotspots covered by each conservation use|
|I3a / I3b|% of other uses located in refugia / hotspots|
|I4a / I4b|% of each nature-dependent use in low / moderate-to-high or hotspot exposure|

**Pilot areas:** Northern Adriatic Sea (NAS), Northern Tyrrhenian Sea (NTS), Strait of Sicily (SoS)
**Scenarios:** SlowPace (SP), Nature@Work (N@W), BlueDevelopment (BD), compared with the approved Italian MSP baseline
**Biozones:** euphotic, mesophotic, mesopelagic, bathyabyssopelagic

## Installation

```bash
git clone https://github.com/alessiarizzi/climate-smart-MSP.git
cd climate-smart-MSP
conda env create -f environment.yml
conda activate climate-smart-msp
```

## Usage

```bash
python code/climate\_smartness\_analysis.py
```

All paths and settings are in the `CONFIGURATION` section at the top of the script: velocity threshold, `n\_years`, exposure thresholds, depth bounds, use groups and cell assignment (`centroid` or `area\_weighted`).

## Input data

**Climate velocity** comes from the [3D climate velocity pipeline](https://github.com/alessiarizzi/3D-climate-change-velocity) (Rizzi et al., 2026). For each biozone:

* `{range}\_dist\_vel\_{threshold}.nc`, with `delta\_xy` (km) and `delta\_z` (m) over the period
* `{range}\_no\_analogue\_{threshold}.csv`, with columns `depth`, `lat`, `lon`

**Shapefiles** (in `data/shapefiles/`):

|File|Content|Required fields|
|-|-|-|
|`{NAS,NTS,SoS}\_pilot\_area.shp`|Pilot area boundaries|–|
|`med\_sea.shp`|Mediterranean Sea polygon|–|
|`EMODnet\_HA\_MSP\_Zoning\_Element\_\*.shp`|Approved Italian MSP ([EMODnet Human Activities](https://emodnet.ec.europa.eu/en/human-activities))|`MS`, `SeaUseFct`|
|`protezione\_baseline.shp`, `OECM\_baseline.shp`, `acquacultura\_baseline.shp`|Additional baselines|`Use\_NameIT`|
|`scenari\_MSP4Biodiversity.shp`|MSP4Biodiversity scenarios|`Use\_NameIT`, `pilot\_area`|

Shapefiles are not included in this repository. The approved Italian MSP is available from EMODnet Human Activities. Pilot area boundaries, additional baselines and MSP4Biodiversity scenarios are available from the authors on reasonable request.

## Outputs

Written to `data/outputs/climate-smart\_assessment\_3d/`:

* `climate\_smart\_indicators.csv`: indicators per pilot area × biozone × scenario
* `biozone\_stats.csv`: velocity statistics per pilot area × biozone
* `use\_exposure\_3d\_summary.csv`: exposure of each use type
* exposure maps (PNG, optionally GeoTIFF) and indicator scorecards (PNG)

## Related work

* Rizzi A., Menegon S., Fianchini M., Melaku Canu D., Gissi E. (2026). Exploring climate change exposure of the Mediterranean Sea through 3D climate velocity. *Ecological Indicators*, 188, 115018. https://doi.org/10.1016/j.ecolind.2026.115018
* Code: [3D-climate-change-velocity](https://github.com/alessiarizzi/3D-climate-change-velocity)

## License

Released under the [MIT License](LICENSE).

## 

