from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from io import BufferedIOBase
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import numpy.typing as npt
from numpy import datetime64

from doppy import exceptions
from doppy.raw.utils import bytes_from_src


@dataclass
class GalionScn:
    filename: str
    campaign_code: str
    campaign_number: int
    rays_in_scan: int
    start_time: datetime64
    time: npt.NDArray[datetime64]  # dim: (time, )
    radial_distance: npt.NDArray[np.float64]  # dim: (radial_distance, )
    azimuth: npt.NDArray[np.float64]  # dim: (time, )
    elevation: npt.NDArray[np.float64]  # dim: (time, )
    pitch: npt.NDArray[np.float64]  # dim: (time, )
    roll: npt.NDArray[np.float64]  # dim: (time, )
    radial_velocity: npt.NDArray[np.float64]  # dim: (time, radial_distance)
    intensity: npt.NDArray[np.float64]  # dim: (time, radial_distance)

    @classmethod
    def from_srcs(
        cls, data: Sequence[str | bytes | Path | BufferedIOBase], overlapped_gates: bool = False
    ) -> list[GalionScn]:
        """Load multiple Galion .scn files"""
        return [cls.from_src(src, overlapped_gates) for src in data]

    @classmethod
    def from_src(cls, data: str | Path | bytes | BufferedIOBase, overlapped_gates: bool = False) -> GalionScn:
        """Load a single Galion .scn file"""
        if isinstance(data, (str, Path)):
            with open(data, 'r', encoding='utf-8') as f:
                content = f.read()
        elif isinstance(data, bytes):
            content = data.decode('utf-8')
        elif isinstance(data, BufferedIOBase):
            content = data.read().decode('utf-8')
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        return _parse_galion_scn(content, overlapped_gates)

    def __getitem__(self, index) -> GalionScn:
        """Allow indexing like other raw data classes"""
        if isinstance(index, (int, slice, list, np.ndarray)):
            return GalionScn(
                filename=self.filename,
                campaign_code=self.campaign_code,
                campaign_number=self.campaign_number,
                rays_in_scan=self.rays_in_scan,
                start_time=self.start_time,
                time=self.time[index],
                radial_distance=self.radial_distance,
                azimuth=self.azimuth[index],
                elevation=self.elevation[index],
                pitch=self.pitch[index],
                roll=self.roll[index],
                radial_velocity=self.radial_velocity[index],
                intensity=self.intensity[index],
            )
        raise TypeError

    @property
    def azimuth_angles(self) -> set[int]:
        """Get unique azimuth angles"""
        return set(int(x) % 360 for x in np.round(self.azimuth))

    @property
    def elevation_angles(self) -> set[int]:
        """Get unique elevation angles"""
        return set(int(x) for x in np.round(self.elevation))

    def sorted_by_time(self) -> GalionScn:
        """Sort by time"""
        sort_indices = np.argsort(self.time)
        return self[sort_indices]

    def nans_removed(self) -> GalionScn:
        """Remove profiles with NaN values"""
        is_ok = np.array(~np.isnan(self.intensity).any(axis=1), dtype=np.bool_)
        return self[is_ok]


def _parse_galion_scn(content: str, overlapped_gates: bool = False) -> GalionScn:
    """Parse a Galion .scn file content
    
    Note: Galion lidars use non-overlapped range gates. The overlapped_gates 
    parameter should be False for Galion data.
    """
    lines = content.strip().split('\n')
    
    # Parse header
    header = {}
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        if ':' in line and not line.startswith('Range gate'):
            key, value = line.split(':', 1)
            header[key.strip()] = value.strip()
        elif line.startswith('Range gate'):
            data_start_idx = i + 1
            break
    
    # Extract header information
    filename = header.get('Filename', '')
    campaign_code = header.get('Campaign code', '')
    campaign_number = int(header.get('Campaign number', 0))
    rays_in_scan = int(header.get('Rays in scan', 0))
    start_time_str = header.get('Start time', '')
    
    # Parse start time
    start_time = datetime64(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f'))
    
    # Parse data lines
    data_lines = lines[data_start_idx:]
    
    # Parse data into arrays
    gate_nums = []
    doppler_vals = []
    intensity_vals = []
    ray_times = []
    azimuths = []
    elevations = []
    pitches = []
    rolls = []
    
    for line in data_lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 8:
                gate_nums.append(int(parts[0]))
                doppler_vals.append(float(parts[1]))
                intensity_vals.append(float(parts[2]))
                ray_times.append(parts[3] + ' ' + parts[4])  # Combine date and time
                azimuths.append(float(parts[5]))
                elevations.append(float(parts[6]))
                pitches.append(float(parts[7]))
                rolls.append(float(parts[8]))
    
    # Convert to numpy arrays
    gate_nums = np.array(gate_nums)
    doppler_vals = np.array(doppler_vals)
    intensity_vals = np.array(intensity_vals)
    azimuths = np.array(azimuths)
    elevations = np.array(elevations)
    pitches = np.array(pitches)
    rolls = np.array(rolls)
    
    # Parse ray times
    unique_times = []
    for time_str in ray_times:
        unique_times.append(datetime64(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')))
    
    # Determine the structure
    unique_gates = np.unique(gate_nums)
    max_gate = int(np.max(gate_nums))
    n_gates = max_gate + 1
    
    # Group data by time/ray
    time_indices = {}
    for i, time_str in enumerate(ray_times):
        if time_str not in time_indices:
            time_indices[time_str] = []
        time_indices[time_str].append(i)
    
    n_times = len(time_indices)
    unique_time_strs = sorted(time_indices.keys())
    
    # Create time array
    time_array = np.array([
        datetime64(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')) 
        for t in unique_time_strs
    ])
    
    # Calculate radial distances
    if overlapped_gates:
        # For overlapped gates: first gate at center, subsequent gates increment by 3m
        # Assuming typical range gate length of 30m for Galion systems
        range_gate_length = 30.0  # This might need to be configurable
        radial_distance = np.zeros(n_gates, dtype=np.float64)
        radial_distance[0] = 0.5 * range_gate_length
        for i in range(1, n_gates):
            radial_distance[i] = radial_distance[0] + i * 3.0
    else:
        # Standard calculation: assume 30m range gate length for Galion
        range_gate_length = 30.0
        radial_distance = (unique_gates + 0.5) * range_gate_length
    
    # Initialize output arrays
    radial_velocity = np.full((n_times, n_gates), np.nan, dtype=np.float64)
    intensity = np.full((n_times, n_gates), np.nan, dtype=np.float64)
    azimuth_out = np.zeros(n_times, dtype=np.float64)
    elevation_out = np.zeros(n_times, dtype=np.float64)
    pitch_out = np.zeros(n_times, dtype=np.float64)
    roll_out = np.zeros(n_times, dtype=np.float64)
    
    # Fill arrays
    for t_idx, time_str in enumerate(unique_time_strs):
        indices = time_indices[time_str]
        
        # Get metadata for this time (should be the same for all gates)
        azimuth_out[t_idx] = azimuths[indices[0]]
        elevation_out[t_idx] = elevations[indices[0]]
        pitch_out[t_idx] = pitches[indices[0]]
        roll_out[t_idx] = rolls[indices[0]]
        
        # Fill gate data
        for idx in indices:
            gate_idx = gate_nums[idx]
            if gate_idx < n_gates:
                radial_velocity[t_idx, gate_idx] = doppler_vals[idx]
                intensity[t_idx, gate_idx] = intensity_vals[idx]
    
    return GalionScn(
        filename=filename,
        campaign_code=campaign_code,
        campaign_number=campaign_number,
        rays_in_scan=rays_in_scan,
        start_time=start_time,
        time=time_array,
        radial_distance=radial_distance,
        azimuth=azimuth_out,
        elevation=elevation_out,
        pitch=pitch_out,
        roll=roll_out,
        radial_velocity=radial_velocity,
        intensity=intensity,
    )
