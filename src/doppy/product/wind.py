from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path
from typing import Sequence, Any, Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage import generic_filter
from sklearn.cluster import KMeans

import doppy

# ngates, gate points, elevation angle, tuple of sorted azimuth angles
SelectionGroupKeyType = Tuple[int, int, Tuple[int, ...]]


@dataclass
class Options:
    azimuth_offset_deg: float | None = None
    overlapped_gates: bool = False
    gate_length_div: float = 2.0
    gate_index_mul: float = 3.0
    # ... add any other options you need ...


@dataclass
class Wind:
    time: npt.NDArray[np.datetime64]
    height: npt.NDArray[np.float64]
    zonal_wind: npt.NDArray[np.float64]
    meridional_wind: npt.NDArray[np.float64]
    vertical_wind: npt.NDArray[np.float64]
    mask: npt.NDArray[np.bool_]
    system_id: str
    options: Options | None

    @property
    def mask_zonal_wind(self) -> npt.NDArray[np.bool_]:
        return np.isnan(self.zonal_wind)

    @property
    def mask_meridional_wind(self) -> npt.NDArray[np.bool_]:
        return np.isnan(self.meridional_wind)

    @property
    def mask_vertical_wind(self) -> npt.NDArray[np.bool_]:
        return np.isnan(self.vertical_wind)

    @functools.cached_property
    def horizontal_wind_speed(self) -> npt.NDArray[np.float64]:
        return np.sqrt(self.zonal_wind**2 + self.meridional_wind**2)

    @functools.cached_property
    def horizontal_wind_direction(self) -> npt.NDArray[np.float64]:
        direction = np.arctan2(self.zonal_wind, self.meridional_wind)
        direction[direction < 0] += 2 * np.pi
        return np.array(np.degrees(direction), dtype=np.float64)

    @classmethod
    def from_halo_data(
        cls,
        data: Sequence[str]
        | Sequence[Path]
        | Sequence[bytes]
        | Sequence[BufferedIOBase],
        options: Options | None = None,
    ) -> Wind:
        raws = doppy.raw.HaloHpl.from_srcs(
            data, overlapped_gates=options.overlapped_gates if options else False
        )

        if len(raws) == 0:
            raise doppy.exceptions.NoDataError("HaloHpl data missing")

        raw = (
            doppy.raw.HaloHpl.merge(_select_raws_for_wind(raws))
            .sorted_by_time()
            .non_strictly_increasing_timesteps_removed()
            .nans_removed()
        )
        if len(raw.time) == 0:
            raise doppy.exceptions.NoDataError("No suitable data for the wind product")

        if options and options.azimuth_offset_deg:
            raw.azimuth += options.azimuth_offset_deg

        groups = _group_scans_by_azimuth_rotation(raw)
        time_list = []
        elevation_list = []
        wind_list = []
        rmse_list = []

        for group_index in set(groups):
            pick = group_index == groups
            if pick.sum() < 4:
                continue
            time_, elevation_, wind_, rmse_ = _compute_wind(raw[pick])
            time_list.append(time_)
            elevation_list.append(elevation_)
            wind_list.append(wind_[np.newaxis, :, :])
            rmse_list.append(rmse_[np.newaxis, :])
        time = np.array(time_list)
        if len(time) == 0:
            raise doppy.exceptions.NoDataError(
                "Probably something wrong with scan grouping"
            )
        elevation = np.array(elevation_list)
        wind = np.concatenate(wind_list)
        rmse = np.concatenate(rmse_list)
        if not np.allclose(elevation, elevation[0]):
            raise ValueError("Elevation is expected to stay same")
        height = raw.radial_distance * np.sin(np.deg2rad(elevation[0]))
        mask = _compute_mask(wind, rmse)
        return Wind(
            time=time,
            height=height,
            zonal_wind=wind[:, :, 0],
            meridional_wind=wind[:, :, 1],
            vertical_wind=wind[:, :, 2],
            mask=mask,
            system_id=raw.header.system_id,
            options=options,
        )

    @classmethod
    def from_windcube_data(
        cls,
        data: Sequence[str]
        | Sequence[Path]
        | Sequence[bytes]
        | Sequence[BufferedIOBase],
        options: Options | None = None,
    ) -> Wind:
        raws = doppy.raw.WindCube.from_vad_or_dbs_srcs(data)

        if len(raws) == 0:
            raise doppy.exceptions.NoDataError("WindCube data missing")

        raw = (
            doppy.raw.WindCube.merge(raws)
            .sorted_by_time()
            .non_strictly_increasing_timesteps_removed()
            .reindex_scan_indices()
        )
        # select scans with most frequent elevation angle from range (15,85)
        raw = raw[(raw.elevation > 15) & (raw.elevation < 85)]
        elevation_ints = raw.elevation.round().astype(int)
        unique_elevations, counts = np.unique(elevation_ints, return_counts=True)
        most_frequent_elevation = unique_elevations[np.argmax(counts)]
        raw = raw[elevation_ints == most_frequent_elevation]

        if len(raw.time) == 0:
            raise doppy.exceptions.NoDataError("No suitable data for the wind product")

        if options and options.azimuth_offset_deg:
            raw.azimuth += options.azimuth_offset_deg

        time_list = []
        elevation_list = []
        wind_list = []
        rmse_list = []

        for scan_index in set(raw.scan_index):
            pick = raw.scan_index == scan_index
            if pick.sum() < 4:
                continue
            time_, elevation_, wind_, rmse_ = _compute_wind(raw[pick])
            time_list.append(time_)
            elevation_list.append(elevation_)
            wind_list.append(wind_[np.newaxis, :, :])
            rmse_list.append(rmse_[np.newaxis, :])

        time = np.array(time_list)
        elevation = np.array(elevation_list)
        wind = np.concatenate(wind_list)
        rmse = np.concatenate(rmse_list)
        mask = _compute_mask(wind, rmse) | np.any(np.isnan(wind), axis=2)
        if not np.allclose(elevation, elevation[0]):
            raise ValueError("Elevation is expected to stay same")
        if not (raw.height == raw.height[0]).all():
            raise ValueError("Unexpected heights")
        height = np.array(raw.height[0], dtype=np.float64)
        return Wind(
            time=time,
            height=height,
            zonal_wind=wind[:, :, 0],
            meridional_wind=wind[:, :, 1],
            vertical_wind=wind[:, :, 2],
            mask=mask,
            system_id=raw.system_id,
            options=options,
        )

    @classmethod
    def from_wls70_data(
        cls,
        data: Sequence[str]
        | Sequence[Path]
        | Sequence[bytes]
        | Sequence[BufferedIOBase],
        options: Options | None = None,
    ) -> Wind:
        raws = doppy.raw.Wls70.from_srcs(data)

        if len(raws) == 0:
            raise doppy.exceptions.NoDataError("Wls70 data missing")

        raw = (
            doppy.raw.Wls70.merge(raws)
            .sorted_by_time()
            .non_strictly_increasing_timesteps_removed()
        )

        if options and options.azimuth_offset_deg:
            theta = np.deg2rad(options.azimuth_offset_deg)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            meridional_wind = (
                sin_theta * raw.zonal_wind + cos_theta * raw.meridional_wind
            )
            zonal_wind = cos_theta * raw.zonal_wind - sin_theta * raw.meridional_wind
        else:
            meridional_wind = raw.meridional_wind
            zonal_wind = raw.zonal_wind

        mask = (
            np.isnan(raw.meridional_wind)
            | np.isnan(raw.zonal_wind)
            | np.isnan(raw.vertical_wind)
        )
        return Wind(
            time=raw.time,
            height=raw.altitude,
            zonal_wind=zonal_wind,
            meridional_wind=meridional_wind,
            vertical_wind=raw.vertical_wind,
            mask=mask,
            system_id=raw.system_id,
            options=options,
        )

    @classmethod
    def from_galion_data(
        cls,
        data: Sequence[str]
        | Sequence[Path]
        | Sequence[bytes]
        | Sequence[BufferedIOBase],
        options: Options | None = None,
        azimuth_offset_deg: float | None = None,
        overlapped_gates: bool = False,
    ) -> Wind:
        """Create Wind product from Galion .scn files
        
        This method processes each .scn file individually and then merges the results.
        This approach prevents issues with mixing different scan types and handles
        time gaps between files properly.
        
        Parameters
        ----------
        data : Sequence
            Sequence of file paths, bytes, or file-like objects for Galion .scn files
        options : Options, optional
            Wind processing options. If provided, azimuth_offset_deg and overlapped_gates 
            parameters are ignored in favor of the options object.
        azimuth_offset_deg : float, optional
            Azimuth offset in degrees. Only used if options is None.
        overlapped_gates : bool, optional
            Whether to use overlapped gates processing. Only used if options is None.
        """
        # Import here to avoid circular imports
        from doppy.raw.galion_scn import GalionScn

        # If options not provided, create from individual parameters
        if options is None:
            options = Options(
                azimuth_offset_deg=azimuth_offset_deg,
                overlapped_gates=overlapped_gates
            )

        # Process each file individually
        all_wind_results = []
        processed_files = 0
        failed_files = 0
        
        for file_data in data:
            try:
                # Load single file
                raw = GalionScn.from_src(
                    file_data, 
                    overlapped_gates=options.overlapped_gates if options else False
                )
                
                if len(raw.time) == 0:
                    print(f"Warning: No data in file, skipping")
                    continue
                
                # Process this single file
                file_wind_result = _process_single_galion_file(raw, options)
                
                if file_wind_result is not None:
                    all_wind_results.append(file_wind_result)
                    processed_files += 1
                else:
                    failed_files += 1
                    
            except Exception as e:
                print(f"Warning: Failed to process file: {e}")
                failed_files += 1
                continue
        
        if len(all_wind_results) == 0:
            raise doppy.exceptions.NoDataError(
                f"No valid wind data could be extracted from {len(data)} files. "
                f"Processed: {processed_files}, Failed: {failed_files}"
            )
        
        print(f"Successfully processed {processed_files} files, {failed_files} failed")
        
        # Merge results from all files
        return _merge_galion_wind_results(all_wind_results, options)

    def write_to_netcdf(self, filename: str | Path) -> None:
        with doppy.netcdf.Dataset(filename) as nc:
            nc.add_dimension("time")
            nc.add_dimension("height")
            nc.add_time(
                name="time",
                dimensions=("time",),
                standard_name="time",
                long_name="Time UTC",
                data=self.time,
                dtype="f8",
            )
            nc.add_variable(
                name="height",
                dimensions=("height",),
                units="m",
                data=self.height,
                dtype="f4",
            )
            nc.add_variable(
                name="uwind_raw",
                dimensions=("time", "height"),
                units="m s-1",
                data=self.zonal_wind,
                mask=self.mask_zonal_wind,
                dtype="f4",
                long_name="Non-screened zonal wind",
            )
            nc.add_variable(
                name="uwind",
                dimensions=("time", "height"),
                units="m s-1",
                data=self.zonal_wind,
                mask=self.mask | self.mask_zonal_wind,
                dtype="f4",
                long_name="Zonal wind",
            )
            nc.add_variable(
                name="vwind_raw",
                dimensions=("time", "height"),
                units="m s-1",
                data=self.meridional_wind,
                mask=self.mask_meridional_wind,
                dtype="f4",
                long_name="Non-screened meridional wind",
            )
            nc.add_variable(
                name="vwind",
                dimensions=("time", "height"),
                units="m s-1",
                data=self.meridional_wind,
                mask=self.mask | self.mask_meridional_wind,
                dtype="f4",
                long_name="Meridional wind",
            )
            nc.add_variable(
                name="wwind_raw",
                dimensions=("time", "height"),
                units="m s-1",
                data=self.vertical_wind,
                mask=self.mask_vertical_wind,
                dtype="f4",
                long_name="Non-screened vertical wind",
            )
            nc.add_variable(
                name="wwind",
                dimensions=("time", "height"),
                units="m s-1",
                data=self.vertical_wind,
                mask=self.mask | self.mask_vertical_wind,
                dtype="f4",
                long_name="Vertical wind",
            )
            nc.add_attribute("serial_number", self.system_id)
            nc.add_attribute("doppy_version", doppy.__version__)
            if self.options is not None and self.options.azimuth_offset_deg is not None:
                nc.add_scalar_variable(
                    name="azimuth_offset",
                    units="degrees",
                    data=self.options.azimuth_offset_deg,
                    dtype="f4",
                    long_name="Azimuth offset of the instrument "
                    "(positive clockwise from north)",
                )


def _compute_wind(
    raw: doppy.raw.HaloHpl | doppy.raw.WindCube,
) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns
    -------
    time

    elevation


    wind (range,component):
        Wind components for each range gate.
        Components:
        0: zonal wind
        1: meridional wind
        2: vertical wind

    rmse (range,):
        Root-mean-square error of radial velocity fit for each range gate.

    References
    ----------
    An assessment of the performance of a 1.5 µm Doppler lidar for
    operational vertical wind profiling based on a 1-year trial
        authors: E. Päschke, R. Leinweber, and V. Lehmann
        doi: 10.5194/amt-8-2251-2015
    """
    elevation = np.deg2rad(raw.elevation)
    azimuth = np.deg2rad(raw.azimuth)
    radial_velocity = raw.radial_velocity

    cos_elevation = np.cos(elevation)
    A = np.hstack(
        (
            (np.sin(azimuth) * cos_elevation).reshape(-1, 1),
            (np.cos(azimuth) * cos_elevation).reshape(-1, 1),
            (np.sin(elevation)).reshape(-1, 1),
        )
    )
    A_inv = np.linalg.pinv(A)

    w = A_inv @ radial_velocity
    r_appr = A @ w
    rmse = np.sqrt(np.sum((r_appr - radial_velocity) ** 2, axis=0) / r_appr.shape[0])
    wind = w.T
    time = raw.time[len(raw.time) // 2]
    elevation = np.round(raw.elevation)
    if not np.allclose(elevation, elevation[0]):
        raise ValueError("Elevations in the scan differ")
    return time, elevation[0], wind, rmse


def _compute_mask(
    wind: npt.NDArray[np.float64], rmse: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Parameters
    ----------

    wind (time,range,component)
    intensty (time,range)
    rmse (time,range)
    """

    def neighbour_diff(X: npt.NDArray[np.float64]) -> np.float64:
        mdiff = np.max(np.abs(X - X[len(X) // 2]))
        return np.float64(mdiff)

    WIND_NEIGHBOUR_DIFFERENCE = 20
    neighbour_mask = np.any(
        generic_filter(wind, neighbour_diff, size=(1, 3, 1))
        > WIND_NEIGHBOUR_DIFFERENCE,
        axis=2,
    )

    rmse_th = 5
    return np.array((rmse > rmse_th) | neighbour_mask, dtype=np.bool_)


def _group_scans_by_azimuth_rotation(raw: doppy.raw.HaloHpl) -> npt.NDArray[np.int64]:
    max_timedelta_in_scan = np.timedelta64(30, "s")
    if len(raw.time) < 4:
        raise doppy.exceptions.NoDataError(
            "Less than 4 profiles is not sufficient for wind product."
        )
    groups = -1 * np.ones_like(raw.time, dtype=np.int64)

    group = 0
    first_azimuth_of_scan = _wrap_and_round_angle(raw.azimuth[0])
    groups[0] = group
    for i, (time_prev, time, azimuth) in enumerate(
        zip(raw.time[:-1], raw.time[1:], raw.azimuth[1:]), start=1
    ):
        if (
            angle := _wrap_and_round_angle(azimuth)
        ) == first_azimuth_of_scan or time - time_prev > max_timedelta_in_scan:
            group += 1
            first_azimuth_of_scan = angle
        groups[i] = group
    return groups


def _wrap_and_round_angle(a: np.float64) -> int:
    return int(np.round(a)) % 360


def _group_scans(raw: doppy.raw.HaloHpl) -> npt.NDArray[np.int64]:
    if len(raw.time) < 4:
        raise ValueError("Expected at least 4 profiles to compute wind profile")
    if raw.time.dtype != "<M8[us]":
        raise TypeError("time expected to be in numpy datetime[us]")
    time = raw.time.astype(np.float64) * 1e-6
    timediff_in_seconds = np.diff(time)
    kmeans = KMeans(n_clusters=2, n_init="auto").fit(timediff_in_seconds.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    scanstep_timediff = centers[np.argmin(centers)]

    if scanstep_timediff < 0.1 or scanstep_timediff > 30:
        raise ValueError(
            "Time difference between profiles in one scan "
            "expected to be between 0.1 and 30 seconds"
        )
    scanstep_timediff_upperbound = 2 * scanstep_timediff
    groups_by_time = -1 * np.ones_like(time, dtype=np.int64)
    groups_by_time[0] = 0
    scan_index = 0
    for i, (t_prev, t) in enumerate(zip(time[:-1], time[1:]), start=1):
        if t - t_prev > scanstep_timediff_upperbound:
            scan_index += 1
        groups_by_time[i] = scan_index

    return _subgroup_scans(raw, groups_by_time)


def _subgroup_scans(
    raw: doppy.raw.HaloHpl, time_groups: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """
    Groups scans further based on the azimuth angles
    """
    group = -1 * np.ones_like(raw.time, dtype=np.int64)
    i = -1
    for time_group in set(time_groups):
        i += 1
        (pick,) = np.where(time_group == time_groups)
        raw_group = raw[pick]
        first_azimuth_angle = int(np.round(raw_group.azimuth[0])) % 360
        group[pick[0]] = i
        for j, azi in enumerate(
            (int(np.round(azi)) % 360 for azi in raw_group.azimuth[1:]), start=1
        ):
            if azi == first_azimuth_angle:
                i += 1
            group[pick[j]] = i
    return group


def _select_raws_for_wind(
    raws: Sequence[doppy.raw.HaloHpl],
) -> Sequence[doppy.raw.HaloHpl]:
    if len(raws) == 0:
        raise doppy.exceptions.NoDataError(
            "Cannot select raws for wind from empty list"
        )
    print(f"Selecting raws for wind from {len(raws)} total raws")
    for i, raw in enumerate(raws):
        print(f"Raw #{i}, Elevation angles: {raw.elevation_angles}, Azimuth angles: {raw.azimuth_angles}")
    
    # Filter raws by dominant elevation angle
    raws_wind = []
    for raw in raws:
        # Find the most frequent elevation angle
        elevation_ints = raw.elevation.round().astype(int)
        unique_elevations, counts = np.unique(elevation_ints, return_counts=True)
        dominant_elevation = unique_elevations[np.argmax(counts)]
        
        # Check if dominant elevation is in valid range and has sufficient azimuth coverage
        if (dominant_elevation < 80 and dominant_elevation > 4 
            and len(raw.azimuth_angles) > 3):
            
            # Filter the raw data to keep only the dominant elevation
            elevation_mask = elevation_ints == dominant_elevation
            if np.sum(elevation_mask) >= 4:  # Need at least 4 points for VAD
                print(f"  Using dominant elevation {dominant_elevation}° ({np.sum(elevation_mask)} points)")
                
                # Create filtered version of raw data
                filtered_raw = raw[elevation_mask]
                raws_wind.append(filtered_raw)
            else:
                print(f"  Insufficient points ({np.sum(elevation_mask)}) at dominant elevation {dominant_elevation}°")
        else:
            print(f"  Dominant elevation {dominant_elevation}° not suitable for wind processing")
    
    if len(raws_wind) == 0:
        raise doppy.exceptions.NoDataError(
            "No data suitable for winds: "
            "No dominant elevation angle in range (4°, 80°) with >3 azimuth angles and >=4 data points"
        )

    groups: dict[SelectionGroupKeyType, int] = defaultdict(int)

    for raw in raws_wind:
        groups[_selection_key(raw)] += len(raw.time)

    def key_func(key: SelectionGroupKeyType) -> int:
        return groups[key]

    select_tuple = max(groups, key=key_func)

    return [raw for raw in raws_wind if _selection_key(raw) == select_tuple]


def _selection_key(raw: doppy.raw.HaloHpl) -> SelectionGroupKeyType:
    if len(raw.elevation_angles) != 1:
        raise ValueError("Expected only one elevation angle")
    return (
        raw.header.mergeable_hash(),
        next(iter(raw.elevation_angles)),
        tuple(sorted(raw.azimuth_angles)),
    )


def _merge_galion_raws(raws: Sequence) -> Any:
    """Merge Galion raw data into a single object for wind processing"""
    if not raws:
        raise doppy.exceptions.NoDataError("No Galion raw data to merge")
    
    # Create a simple container class for merged Galion data
    class MergedGalionData:
        def __init__(self):
            all_times = []
            all_elevations = []
            all_azimuths = []
            all_radial_velocities = []
            all_intensities = []
            
            # Collect all data
            for raw in raws:
                all_times.append(raw.time)
                all_elevations.append(raw.elevation)
                all_azimuths.append(raw.azimuth)
                all_radial_velocities.append(raw.radial_velocity)
                all_intensities.append(raw.intensity)
            
            # Concatenate arrays
            self.time = np.concatenate(all_times)
            self.elevation = np.concatenate(all_elevations)
            self.azimuth = np.concatenate(all_azimuths)
            self.radial_velocity = np.concatenate(all_radial_velocities)
            self.intensity = np.concatenate(all_intensities)
            
            # Use radial distance from first raw (should be consistent)
            self.radial_distance = raws[0].radial_distance
            
            # Store campaign info for system_id
            self.campaign_code = raws[0].campaign_code
            
        def __getitem__(self, index):
            # Allow indexing like other raw data classes
            new_obj = MergedGalionData.__new__(MergedGalionData)
            new_obj.time = self.time[index]
            new_obj.elevation = self.elevation[index]
            new_obj.azimuth = self.azimuth[index]
            new_obj.radial_velocity = self.radial_velocity[index]
            new_obj.intensity = self.intensity[index]
            new_obj.radial_distance = self.radial_distance
            new_obj.campaign_code = self.campaign_code
            return new_obj
    
    return MergedGalionData()


def _compute_wind_galion(
    raw,
) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute wind from Galion radial velocity data
    
    Handles both VAD scans (multiple azimuths) and vertical stare (single azimuth).
    For vertical stare, only the vertical wind component is meaningful.

    Parameters
    ----------
    raw
        Raw data object from Galion .scn file

    Returns
    -------
    tuple
        Contains time, elevation, wind components (zonal, meridional, vertical),
        and RMSE of the wind computation
    """
    elevation = np.deg2rad(raw.elevation)
    azimuth = np.deg2rad(raw.azimuth)
    radial_velocity = raw.radial_velocity

    # Check for single azimuth (vertical stare) case
    unique_azimuths = len(np.unique(np.round(np.rad2deg(azimuth))))
    
    if unique_azimuths == 1:
        # Vertical stare mode - only meaningful output is vertical wind
        print(f"Warning: Single azimuth detected ({np.rad2deg(azimuth[0]):.1f}°). "
              f"Only vertical wind component is meaningful.")
        
        # For vertical stare, if elevation is truly 90°, radial velocity = vertical velocity
        mean_elevation_deg = np.mean(np.rad2deg(elevation))
        
        if mean_elevation_deg > 85:  # Nearly vertical
            # Direct assignment for vertical stare
            wind = np.zeros((raw.radial_velocity.shape[1], 3))  # (gates, components)
            wind[:, 2] = np.mean(radial_velocity, axis=0)  # vertical wind = mean radial velocity
            wind[:, 0] = np.nan  # zonal wind undefined
            wind[:, 1] = np.nan  # meridional wind undefined
            rmse = np.std(radial_velocity, axis=0)  # standard deviation as error measure
        else:
            # Non-vertical single beam - compute line-of-sight contribution to vertical
            sin_elevation = np.sin(elevation)
            wind = np.zeros((raw.radial_velocity.shape[1], 3))
            wind[:, 2] = np.mean(radial_velocity / sin_elevation[0], axis=0)  # project to vertical
            wind[:, 0] = np.nan  # zonal wind undefined  
            wind[:, 1] = np.nan  # meridional wind undefined
            rmse = np.std(radial_velocity, axis=0)
    else:
        # VAD mode - multiple azimuths, can compute full 3D wind vector
        cos_elevation = np.cos(elevation)
        A = np.hstack(
            (
                (np.sin(azimuth) * cos_elevation).reshape(-1, 1),
                (np.cos(azimuth) * cos_elevation).reshape(-1, 1),
                (np.sin(elevation)).reshape(-1, 1),
            )
        )
        A_inv = np.linalg.pinv(A)

        w = A_inv @ radial_velocity
        r_appr = A @ w
        rmse = np.sqrt(np.sum((r_appr - radial_velocity) ** 2, axis=0) / r_appr.shape[0])
        wind = w.T
    
    time = raw.time[len(raw.time) // 2]
    elevation_deg = np.round(np.rad2deg(elevation))
    if not np.allclose(elevation_deg, elevation_deg[0]):
        raise ValueError("Elevations in the scan differ")
    return time, elevation_deg[0], wind, rmse


def _group_scans_by_azimuth_rotation_galion(raw) -> npt.NDArray[np.int64]:
    """
    Group Galion scans by azimuth rotation or time for single-azimuth cases
    
    For VAD scans, groups by azimuth rotation (full 360° cycles).
    For vertical stare (single azimuth), groups by time proximity.
    """
    max_timedelta_in_scan = np.timedelta64(30, "s")
    
    # Check if this is a single-azimuth scan (vertical stare)
    unique_azimuths = len(np.unique(np.round(raw.azimuth, 1)))
    
    if unique_azimuths == 1:
        # Single azimuth - group by time proximity only
        print(f"Single azimuth scan detected - grouping by time proximity")
        
        # Relax minimum profile requirement for vertical stare
        if len(raw.time) < 1:
            raise doppy.exceptions.NoDataError(
                "No profiles available for wind product."
            )
        
        groups = np.zeros_like(raw.time, dtype=np.int64)
        
        if len(raw.time) == 1:
            return groups
            
        group = 0
        groups[0] = group
        for i, (time_prev, time) in enumerate(zip(raw.time[:-1], raw.time[1:]), start=1):
            if time - time_prev > max_timedelta_in_scan:
                group += 1
            groups[i] = group
        return groups
    
    else:
        # Multiple azimuths - use standard VAD grouping
        if len(raw.time) < 4:
            raise doppy.exceptions.NoDataError(
                "Less than 4 profiles is not sufficient for VAD wind product."
            )
        
        groups = -1 * np.ones_like(raw.time, dtype=np.int64)

        group = 0
        first_azimuth_of_scan = _wrap_and_round_angle(raw.azimuth[0])
        groups[0] = group
        for i, (time_prev, time, azimuth) in enumerate(
            zip(raw.time[:-1], raw.time[1:], raw.azimuth[1:]), start=1
        ):
            if (
                angle := _wrap_and_round_angle(azimuth)
            ) == first_azimuth_of_scan or time - time_prev > max_timedelta_in_scan:
                group += 1
                first_azimuth_of_scan = angle
            groups[i] = group
        return groups


def _process_single_galion_file(raw, options: Options | None = None):
    """
    Process a single Galion .scn file to extract wind data
    
    Parameters
    ----------
    raw : GalionScn
        Raw data from a single Galion .scn file
    options : Options, optional
        Processing options
        
    Returns
    -------
    dict or None
        Dictionary containing wind data for this file, or None if processing failed
    """
    try:
        if options and options.azimuth_offset_deg:
            raw.azimuth += options.azimuth_offset_deg

        # Convert to merged format for compatibility with existing functions
        merged_raw = _convert_single_galion_to_merged(raw)
        
        # Group scans within this file
        groups = _group_scans_by_azimuth_rotation_galion(merged_raw)
        
        time_list = []
        elevation_list = []
        wind_list = []
        rmse_list = []

        for group_index in set(groups):
            pick = group_index == groups
            
            # Check if this is single-azimuth (vertical stare) mode
            group_azimuths = merged_raw.azimuth[pick]
            unique_azimuths = len(np.unique(np.round(group_azimuths, 1)))
            
            if unique_azimuths == 1:
                # Vertical stare mode - allow single profiles
                if pick.sum() < 1:
                    continue
            else:
                # VAD mode - require at least 4 profiles
                if pick.sum() < 4:
                    continue
            
            time_, elevation_, wind_, rmse_ = _compute_wind_galion(merged_raw[pick])
            time_list.append(time_)
            elevation_list.append(elevation_)
            wind_list.append(wind_)
            rmse_list.append(rmse_)

        if len(time_list) == 0:
            print(f"Warning: No valid wind groups found in file")
            return None
        
        # Return the data for this file
        return {
            'time': np.array(time_list),
            'elevation': np.array(elevation_list),
            'wind': wind_list,  # List of wind arrays
            'rmse': rmse_list,  # List of rmse arrays
            'height': merged_raw.radial_distance * np.sin(np.deg2rad(elevation_list[0])),
            'campaign_code': raw.campaign_code
        }
        
    except Exception as e:
        print(f"Error processing single file: {e}")
        return None


def _convert_single_galion_to_merged(raw):
    """Convert single GalionScn to merged format for compatibility"""
    class SingleGalionData:
        def __init__(self, raw_data):
            self.time = raw_data.time
            self.elevation = raw_data.elevation
            self.azimuth = raw_data.azimuth
            self.radial_velocity = raw_data.radial_velocity
            self.intensity = raw_data.intensity
            self.radial_distance = raw_data.radial_distance
            self.campaign_code = raw_data.campaign_code
            
        def __getitem__(self, index):
            new_obj = SingleGalionData.__new__(SingleGalionData)
            new_obj.time = self.time[index]
            new_obj.elevation = self.elevation[index]
            new_obj.azimuth = self.azimuth[index]
            new_obj.radial_velocity = self.radial_velocity[index]
            new_obj.intensity = self.intensity[index]
            new_obj.radial_distance = self.radial_distance
            new_obj.campaign_code = self.campaign_code
            return new_obj
    
    return SingleGalionData(raw)


def _merge_galion_wind_results(wind_results: list, options: Options | None = None) -> Wind:
    """
    Merge wind results from multiple Galion files
    
    Parameters
    ----------
    wind_results : list
        List of wind result dictionaries from individual files
    options : Options, optional
        Processing options
        
    Returns
    -------
    Wind
        Merged wind product
    """
    # Collect data from all files
    all_times = []
    all_winds = []
    all_rmses = []
    all_elevations = []
    campaign_codes = set()
    
    # Check height consistency
    reference_height = None
    
    for result in wind_results:
        all_times.extend(result['time'])
        all_winds.extend(result['wind'])
        all_rmses.extend(result['rmse'])
        all_elevations.extend(result['elevation'])
        campaign_codes.add(result['campaign_code'])
        
        if reference_height is None:
            reference_height = result['height']
        elif not np.allclose(reference_height, result['height'], rtol=1e-3):
            print("Warning: Height grids differ between files, using first file's grid")
    
    if len(all_times) == 0:
        raise doppy.exceptions.NoDataError("No valid wind data found in any file")
    
    # Convert to arrays
    time = np.array(all_times)
    elevation = np.array(all_elevations)
    
    # Stack wind and rmse data
    wind = np.stack(all_winds, axis=0)  # (time, gates, components)
    rmse = np.stack(all_rmses, axis=0)  # (time, gates)
    
    # Sort by time
    time_order = np.argsort(time)
    time = time[time_order]
    wind = wind[time_order]
    rmse = rmse[time_order]
    elevation = elevation[time_order]
    
    # Check elevation consistency
    if not np.allclose(elevation, elevation[0], atol=1.0):
        print(f"Warning: Elevations vary from {elevation.min():.1f}° to {elevation.max():.1f}°")
        print("Using first elevation for height calculation")
    
    height = reference_height
    mask = _compute_mask(wind, rmse)
    
    # Create system ID from campaign codes
    if len(campaign_codes) == 1:
        system_id = f"Galion-{campaign_codes.pop()}"
    else:
        system_id = f"Galion-{'-'.join(sorted(campaign_codes))}"

    return Wind(
        time=time,
        height=height,
        zonal_wind=wind[:, :, 0],
        meridional_wind=wind[:, :, 1],
        vertical_wind=wind[:, :, 2],
        mask=mask,
        system_id=system_id,
        options=options,
    )
