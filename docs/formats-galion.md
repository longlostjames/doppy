# Galion Scan File Format (.scn)

Galion scan files (`.scn`) contain Doppler lidar data from Galion wind profiler systems. These files are structured text files with a header section followed by data records.

## File Structure

### Header Section
The header contains metadata about the scan, including:
- Filename
- Campaign code and number  
- Number of rays in scan
- Start time
- System configuration parameters

Example header:
```
Filename: 20250627_000001.scn
Campaign code: TEAMX
Campaign number: 1
Rays in scan: 360
Start time: 2025-06-27 00:00:01.000
```

### Data Section
After the header, data is organized with one line per range gate measurement:

```
Range gate  Doppler  Intensity  Date        Time         Azimuth  Elevation  Pitch  Roll
0           -2.45    0.85      2025-06-27  00:00:01.123  0.0      75.0      0.1    0.2
1           -1.23    0.92      2025-06-27  00:00:01.123  0.0      75.0      0.1    0.2
...
```

## Data Fields

- **Range gate**: Gate number (0-based indexing)
- **Doppler**: Radial velocity (m/s)
- **Intensity**: Signal intensity (normalized)
- **Date/Time**: Timestamp for this measurement
- **Azimuth**: Azimuth angle (degrees from North)
- **Elevation**: Elevation angle (degrees from horizontal)
- **Pitch/Roll**: Platform orientation (degrees)

## Usage in Doppy

### Loading Raw Data

```python
from doppy.raw import GalionScn

# Load single file
raw = GalionScn.from_src("scan_file.scn")

# Load multiple files
raws = GalionScn.from_srcs([
    "file1.scn", 
    "file2.scn"
])

# With overlapped gates
raw = GalionScn.from_src("scan_file.scn", overlapped_gates=True)
```

### Generating Wind Products

```python
from doppy.product import Wind

# Create wind product from Galion data
wind = Wind.from_galion_data([
    "scan1.scn",
    "scan2.scn"
], options=Wind.Options(overlapped_gates=False))

# Save to NetCDF
wind.write_to_netcdf("wind_output.nc")
```

## Range Gate Calculation

### Standard Gates
For standard range gates:
- Range = (gate_number + 0.5) × 30m
- Gate 0: 15m, Gate 1: 45m, Gate 2: 75m, etc.

### Overlapped Gates
For overlapped gates (`overlapped_gates=True`):
- Gate 0: 15m (center of first 30m gate)
- Gate 1: 18m (15 + 3m)
- Gate 2: 21m (15 + 6m)
- Gate 3: 24m (15 + 9m)
- etc.

This reflects the physical arrangement where overlapped gates are spaced 3m apart instead of the full 30m gate length.

## System Integration

The Galion support integrates with doppy's existing wind processing algorithms:
- VAD (Velocity Azimuth Display) wind retrieval
- Quality control and masking
- NetCDF output with CF-compliant metadata
- Support for azimuth offset corrections

## File Locations

Galion scan files are typically found in directories like:
```
/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-wind-profiler-1/incoming/
└── 20250603_teamx/
    └── 2025/
        └── 202506/
            └── 20250627/
                └── 00/
                    ├── 20250627_000001.scn
                    ├── 20250627_000002.scn
                    └── ...
```
