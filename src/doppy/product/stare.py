from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path
from typing import Sequence, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy
from scipy.ndimage import median_filter, uniform_filter
from sklearn.cluster import KMeans

import doppy
from doppy import defaults, options
from doppy.product.noise_utils import detect_wind_noise

SelectionGroupKeyType: TypeAlias = tuple[int,]


@dataclass(slots=True)
class Options:
    overlapped_gates: bool = False


@dataclass(slots=True)
class RayAccumulationTime:
    # in seconds
    value: float


@dataclass(slots=True)
class PulsesPerRay:
    value: int


@dataclass(slots=True)
class Stare:
    time: npt.NDArray[np.datetime64]
    radial_distance: npt.NDArray[np.float64]
    elevation: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    snr: npt.NDArray[np.float64]
    radial_velocity: npt.NDArray[np.float64]
    mask_beta: npt.NDArray[np.bool_]
    mask_radial_velocity: npt.NDArray[np.bool_]
    wavelength: float
    system_id: str
    ray_info: RayAccumulationTime | PulsesPerRay

    def __getitem__(
        self,
        index: int
        | slice
        | list[int]
        | npt.NDArray[np.int64]
        | npt.NDArray[np.bool_]
        | tuple[slice, slice],
    ) -> Stare:
        if isinstance(index, (int, slice, list, np.ndarray)):
            return Stare(
                time=self.time[index],
                radial_distance=self.radial_distance,
                elevation=self.elevation[index],
                beta=self.beta[index],
                snr=self.snr[index],
                radial_velocity=self.radial_velocity[index],
                mask_beta=self.mask_beta[index],
                mask_radial_velocity=self.mask_radial_velocity[index],
                wavelength=self.wavelength,
                system_id=self.system_id,
                ray_info=self.ray_info,
            )
        raise TypeError

    @classmethod
    def mask_nan(cls, x: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        return np.isnan(x)

    @classmethod
    def from_windcube_data(
        cls,
        data: Sequence[str]
        | Sequence[Path]
        | Sequence[bytes]
        | Sequence[BufferedIOBase],
    ) -> Stare:
        raws = doppy.raw.WindCubeFixed.from_srcs(data)
        raw = (
            doppy.raw.WindCubeFixed.merge(raws).sorted_by_time().nan_profiles_removed()
        )

        wavelength = defaults.WindCube.wavelength
        beta = _compute_beta(
            snr=raw.cnr,
            radial_distance=raw.radial_distance,
            wavelength=wavelength,
            beam_energy=defaults.WindCube.beam_energy,
            receiver_bandwidth=defaults.WindCube.receiver_bandwidth,
            focus=defaults.WindCube.focus,
            effective_diameter=defaults.WindCube.effective_diameter,
        )

        mask_beta = _compute_noise_mask_for_windcube(raw)
        mask_radial_velocity = detect_wind_noise(
            raw.radial_velocity, raw.radial_distance, mask_beta
        )

        return cls(
            time=raw.time,
            radial_distance=raw.radial_distance,
            elevation=raw.elevation,
            beta=beta,
            snr=raw.cnr,
            radial_velocity=raw.radial_velocity,
            mask_beta=mask_beta,
            mask_radial_velocity=mask_radial_velocity,
            wavelength=wavelength,
            system_id=raw.system_id,
            ray_info=RayAccumulationTime(raw.ray_accumulation_time),
        )

    @classmethod
    def from_halo_data(
        cls,
        data: Sequence[str]
        | Sequence[Path]
        | Sequence[bytes]
        | Sequence[BufferedIOBase],
        data_bg: Sequence[str]
        | Sequence[Path]
        | Sequence[tuple[bytes, str]]
        | Sequence[tuple[BufferedIOBase, str]],
        bg_correction_method: options.BgCorrectionMethod,
        options: Options | None = None,
    ) -> Stare:
        raws = doppy.raw.HaloHpl.from_srcs(
            data, overlapped_gates=options.overlapped_gates if options else False
        )

        if len(raws) == 0:
            raise doppy.exceptions.NoDataError("HaloHpl data missing")

        raw = (
            doppy.raw.HaloHpl.merge(_select_raws_for_stare(raws))
            .sorted_by_time()
            .non_strictly_increasing_timesteps_removed()
            .nans_removed()
        )

        bgs = doppy.raw.HaloBg.from_srcs(data_bg)
        bgs = [bg[:, : raw.header.ngates] for bg in bgs]
        bgs_stare = [bg for bg in bgs if bg.ngates == raw.header.ngates]

        if len(bgs_stare) == 0:
            raise doppy.exceptions.NoDataError("Background data missing")

        bg = (
            doppy.raw.HaloBg.merge(bgs_stare)
            .sorted_by_time()
            .non_strictly_increasing_timesteps_removed()
        )
        raw, intensity_bg_corrected = _correct_background(raw, bg, bg_correction_method)
        if len(raw.time) == 0:
            raise doppy.exceptions.NoDataError("No matching data and bg files")
        intensity_noise_bias_corrected = _correct_intensity_noise_bias(
            raw, intensity_bg_corrected
        )
        wavelength = defaults.Halo.wavelength

        beta = _compute_beta(
            snr=intensity_noise_bias_corrected - 1,
            radial_distance=raw.radial_distance,
            wavelength=wavelength,
            beam_energy=defaults.Halo.beam_energy,
            receiver_bandwidth=defaults.Halo.receiver_bandwidth,
            focus=raw.header.focus_range,
            effective_diameter=defaults.Halo.effective_diameter,
        )

        mask_beta = _compute_noise_mask(
            intensity_noise_bias_corrected, raw.radial_velocity, raw.radial_distance
        )
        mask_radial_velocity = detect_wind_noise(
            raw.radial_velocity, raw.radial_distance, mask_beta
        )

        return cls(
            time=raw.time,
            radial_distance=raw.radial_distance,
            elevation=raw.elevation,
            beta=beta,
            snr=intensity_noise_bias_corrected - 1,
            radial_velocity=raw.radial_velocity,
            mask_beta=mask_beta,
            mask_radial_velocity=mask_radial_velocity,
            wavelength=wavelength,
            system_id=raw.header.system_id,
            ray_info=PulsesPerRay(raw.header.pulses_per_ray),
        )

    def write_to_netcdf(self, filename: str | Path) -> None:
        with doppy.netcdf.Dataset(filename) as nc:
            nc.add_dimension("time")
            nc.add_dimension("range")
            nc.add_time(
                name="time",
                dimensions=("time",),
                standard_name="time",
                long_name="Time UTC",
                data=self.time,
                dtype="f8",
            )
            nc.add_variable(
                name="range",
                dimensions=("range",),
                units="m",
                data=self.radial_distance,
                dtype="f4",
            )
            nc.add_variable(
                name="elevation",
                dimensions=("time",),
                units="degrees",
                data=self.elevation,
                dtype="f4",
                long_name="elevation from horizontal",
            )
            nc.add_variable(
                name="beta_raw",
                dimensions=("time", "range"),
                units="sr-1 m-1",
                data=self.beta,
                dtype="f4",
            )
            nc.add_variable(
                name="beta",
                dimensions=("time", "range"),
                units="sr-1 m-1",
                data=self.beta,
                dtype="f4",
                mask=self.mask_beta,
            )
            nc.add_variable(
                name="v",
                dimensions=("time", "range"),
                units="m s-1",
                long_name="Doppler velocity",
                data=self.radial_velocity,
                dtype="f4",
                mask=self.mask_radial_velocity,
            )
            nc.add_scalar_variable(
                name="wavelength",
                units="m",
                standard_name="radiation_wavelength",
                data=self.wavelength,
                dtype="f4",
            )
            match self.ray_info:
                case RayAccumulationTime(value):
                    nc.add_scalar_variable(
                        name="ray_accumulation_time",
                        units="s",
                        long_name="ray accumulation time",
                        data=value,
                        dtype="f4",
                    )
                case PulsesPerRay(value):
                    nc.add_scalar_variable(
                        name="pulses_per_ray",
                        units="1",
                        long_name="pulses per ray",
                        data=value,
                        dtype="u4",
                    )

            nc.add_attribute("serial_number", self.system_id)
            nc.add_attribute("doppy_version", doppy.__version__)


def _compute_noise_mask_for_windcube(
    raw: doppy.raw.WindCubeFixed,
) -> npt.NDArray[np.bool_]:
    if np.any(np.isnan(raw.cnr)) or np.any(np.isnan(raw.radial_velocity)):
        raise ValueError("Unexpected nans in crn or radial_velocity")

    mask = _mask_with_cnr_norm_dist(raw.cnr) | (np.abs(raw.radial_velocity) > 30)

    cnr = raw.cnr.copy()
    cnr[mask] = np.finfo(float).eps
    cnr_filt = np.array(median_filter(cnr, size=(3, 3)), dtype=np.float64)
    rel_diff = np.abs(cnr - cnr_filt) / np.abs(cnr)
    diff_mask = rel_diff > 0.25

    mask = mask | diff_mask

    return np.array(mask, dtype=np.bool_)


def _mask_with_cnr_norm_dist(cnr: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    th_trunc = -5.5
    std_factor = 2
    log_cnr = np.log(cnr)
    log_cnr_trunc = log_cnr[log_cnr < th_trunc]
    th_trunc_fit = np.percentile(log_cnr_trunc, 90)
    log_cnr_for_fit = log_cnr_trunc[log_cnr_trunc < th_trunc_fit]
    mean, std = scipy.stats.norm.fit(log_cnr_for_fit)
    return np.array(np.log(cnr) < (mean + std_factor * std), dtype=np.bool_)


def _compute_noise_mask(
    intensity: npt.NDArray[np.float64],
    radial_velocity: npt.NDArray[np.float64],
    radial_distance: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    intensity_mean_mask = uniform_filter(intensity, size=(21, 3)) < 1.0025
    velocity_abs_mean_mask = uniform_filter(np.abs(radial_velocity), size=(21, 3)) > 2
    THREE_PULSES_LENGTH = 90
    near_instrument_noise_mask = np.zeros_like(intensity, dtype=np.bool_)
    near_instrument_noise_mask[:, radial_distance < THREE_PULSES_LENGTH] = True
    low_intensity_mask = intensity < 1
    return np.array(
        (intensity_mean_mask & velocity_abs_mean_mask)
        | near_instrument_noise_mask
        | low_intensity_mask,
        dtype=np.bool_,
    )


def _compute_beta(
    snr: npt.NDArray[np.float64],
    radial_distance: npt.NDArray[np.float64],
    wavelength: float,
    beam_energy: float,
    receiver_bandwidth: float,
    focus: float,
    effective_diameter: float,
) -> npt.NDArray[np.float64]:
    """
    Parameters
    ----------
    snr
        for halo: intensity - 1
    radial_distance
        distance from the instrument
    focus
        focal length of the telescope for the transmitter and receiver
    wavelength
        laser wavelength

    Local variables
    ---------------
    eta
        detector quantum efficiency
    E
        beam energy
    nu
        optical frequency
    h
        planc's constant
    c
        speed of light
    B
        receiver bandwidth

    References
    ----------
    Methodology for deriving the telescope focus function and
    its uncertainty for a heterodyne pulsed Doppler lidar
        authors:  Pyry Pentikäinen, Ewan James O'Connor,
            Antti Juhani Manninen, and Pablo Ortiz-Amezcua
        doi: https://doi.org/10.5194/amt-13-2849-2020
    """

    h = scipy.constants.Planck
    c = scipy.constants.speed_of_light
    eta = 1
    E = beam_energy
    B = receiver_bandwidth
    nu = c / wavelength
    A_e = _compute_effective_receiver_energy(
        radial_distance, wavelength, focus, effective_diameter
    )
    beta = 2 * h * nu * B * radial_distance**2 * snr / (eta * c * E * A_e)
    return np.array(beta, dtype=np.float64)


def _compute_effective_receiver_energy(
    radial_distance: npt.NDArray[np.float64],
    wavelength: float,
    focus: float,
    effective_diameter: float,
) -> npt.NDArray[np.float64]:
    """
    NOTE
    ----
    Using uncalibrated values from https://doi.org/10.5194/amt-13-2849-2020


    Parameters
    ----------
    radial_distance
        distance from the instrument
    focus
        effective focal length of the telescope for the transmitter and receiver
    wavelength
        laser wavelength
    """
    D = effective_diameter
    return np.array(
        np.pi
        * D**2
        / (
            4
            * (
                1
                + (np.pi * D**2 / (4 * wavelength * radial_distance)) ** 2
                * (1 - radial_distance / focus) ** 2
            )
        ),
        dtype=np.float64,
    )


def _correct_intensity_noise_bias(
    raw: doppy.raw.HaloHpl, intensity: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Parameters
    ----------
    intensity:
        intensity after background correction
    """
    noise_mask = _locate_noise(intensity)
    # Ignore lower gates
    noise_mask[:, raw.radial_distance <= 90] = False

    A_ = np.concatenate(
        (
            raw.radial_distance[:, np.newaxis],
            np.ones((len(raw.radial_distance), 1)),
        ),
        axis=1,
    )[np.newaxis, :, :]
    A = np.tile(
        A_,
        (len(intensity), 1, 1),
    )
    A_noise = np.tile(noise_mask[:, :, np.newaxis], (1, 1, 2))
    A[~A_noise] = 0
    intensity_ = intensity.copy()
    intensity_[~noise_mask] = 0

    A_pinv = np.linalg.pinv(A)
    x = A_pinv @ intensity_[:, :, np.newaxis]
    noise_fit = (A_ @ x).squeeze(axis=2)
    return np.array(intensity / noise_fit, dtype=np.float64)


def _locate_noise(intensity: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Returns
    -------
    boolean array M
        where M[i,j] = True if intensity[i,j] contains only noise
        and False otherwise
    """

    INTENSITY_THRESHOLD = 1.008
    MEDIAN_KERNEL_THRESHOLD = 1.002
    GAUSSIAN_THRESHOLD = 0.02

    intensity_normalised = intensity / np.median(intensity, axis=1)[:, np.newaxis]
    intensity_mask = intensity_normalised > INTENSITY_THRESHOLD

    median_mask = (
        scipy.signal.medfilt2d(intensity_normalised, kernel_size=5)
        > MEDIAN_KERNEL_THRESHOLD
    )

    gaussian = scipy.ndimage.gaussian_filter(
        (intensity_mask | median_mask).astype(np.float64), sigma=8, radius=16
    )
    gaussian_mask = gaussian > GAUSSIAN_THRESHOLD

    return np.array(~(intensity_mask | median_mask | gaussian_mask), dtype=np.bool_)


def _correct_background(
    raw: doppy.raw.HaloHpl,
    bg: doppy.raw.HaloBg,
    method: options.BgCorrectionMethod,
) -> Tuple[doppy.raw.HaloHpl, npt.NDArray[np.float64]]:
    """
    Returns
    -------
    raw_with_bg:
        Same as input raw: HaloHpl, but the profiles that does not corresponding
        background measurement have been removed.


    intensity_bg_corrected:
        intensity = SNR + 1 = (A_0 * P_0(z)) / (A_bg * P_bg(z)), z = radial_distance
        The measured background signal P_bg contains usually lots of noise that shows as
        vertical stripes in intensity plots. In bg corrected intensity, P_bg is replaced
        with corrected background profile that should represent the noise floor
        more accurately
    """
    bg_relevant = _select_relevant_background_profiles(bg, raw.time)
    match method:
        case options.BgCorrectionMethod.FIT:
            bg_signal_corrected = _correct_background_by_fitting(
                bg_relevant, raw.radial_distance, fit_method=None
            )
        case options.BgCorrectionMethod.MEAN:
            raise NotImplementedError
        case options.BgCorrectionMethod.PRE_COMPUTED:
            raise NotImplementedError

    raw2bg = np.searchsorted(bg_relevant.time, raw.time, side="right") - 1
    raw_with_bg = raw[raw2bg >= 0]
    raw2bg = raw2bg[raw2bg >= 0]
    raw_bg_original = bg_relevant.signal[raw2bg]
    raw_bg_corrected = bg_signal_corrected[raw2bg]

    intensity_bg_corrected = raw_with_bg.intensity * raw_bg_original / raw_bg_corrected
    return raw_with_bg, intensity_bg_corrected


def _correct_background_by_fitting(
    bg: doppy.raw.HaloBg,
    radial_distance: npt.NDArray[np.float64],
    fit_method: options.BgFitMethod | None,
) -> npt.NDArray[np.float64]:
    clusters = _cluster_background_profiles(bg.signal, radial_distance)
    signal_correcred = np.zeros_like(bg.signal)
    for cluster in set(clusters):
        signal_correcred[clusters == cluster] = _fit_background(
            bg[clusters == cluster], radial_distance, fit_method
        )
    return signal_correcred


def _fit_background(
    bg: doppy.raw.HaloBg,
    radial_distance: npt.NDArray[np.float64],
    fit_method: options.BgFitMethod | None,
) -> npt.NDArray[np.float64]:
    if fit_method is None:
        fit_method = _infer_fit_type(bg.signal, radial_distance)
    match fit_method:
        case options.BgFitMethod.LIN:
            return _linear_fit(bg.signal, radial_distance)
        case options.BgFitMethod.EXP:
            return _exponential_fit(bg.signal, radial_distance)
        case options.BgFitMethod.EXPLIN:
            return _exponential_linear_fit(bg.signal, radial_distance)


def _lin_func(
    x: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.array(x[0] * radial_distance + x[1], dtype=np.float64)


def _exp_func(
    x: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.array(x[0] * np.exp(x[1] * radial_distance ** x[2]), dtype=np.float64)


def _explin_func(
    x: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.array(
        _exp_func(x[:3], radial_distance) + _lin_func(x[3:], radial_distance),
        dtype=np.float64,
    )


def _infer_fit_type(
    bg_signal: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> options.BgFitMethod:
    peaks = _detect_peaks(bg_signal, radial_distance)
    dist_mask = (90 < radial_distance) & (radial_distance < 8000)
    mask = dist_mask & ~peaks

    scale = np.median(bg_signal, axis=1)[:, np.newaxis]

    rdist = radial_distance[np.newaxis][:, mask]

    signal = (bg_signal / scale)[:, mask]

    def lin_func_rss(x: npt.NDArray[np.float64]) -> np.float64:
        return np.float64(((signal - _lin_func(x, rdist)) ** 2).sum())

    def exp_func_rss(x: npt.NDArray[np.float64]) -> np.float64:
        return np.float64(((signal - _exp_func(x, rdist)) ** 2).sum())

    def explin_func_rss(x: npt.NDArray[np.float64]) -> np.float64:
        return np.float64(((signal - _explin_func(x, rdist)) ** 2).sum())

    method = "Nelder-Mead"
    res_lin = scipy.optimize.minimize(
        lin_func_rss, [1e-5, 1], method=method, options={"maxiter": 2 * 600}
    )
    res_exp = scipy.optimize.minimize(
        exp_func_rss, [1, -1, -1], method=method, options={"maxiter": 3 * 600}
    )
    res_explin = scipy.optimize.minimize(
        explin_func_rss, [1, -1, -1, 0, 0], method=method, options={"maxiter": 5 * 600}
    )

    fit_lin = _lin_func(res_lin.x, rdist)
    fit_exp = _exp_func(res_exp.x, rdist)
    fit_explin = _explin_func(res_explin.x, rdist)

    lin_rss = ((signal - fit_lin) ** 2).sum()
    exp_rss = ((signal - fit_exp) ** 2).sum()
    explin_rss = ((signal - fit_explin) ** 2).sum()

    #
    if exp_rss / lin_rss < 0.95 or explin_rss / lin_rss < 0.95:
        if (exp_rss - explin_rss) / lin_rss > 0.05:
            return options.BgFitMethod.EXPLIN
        else:
            return options.BgFitMethod.EXP
    else:
        return options.BgFitMethod.LIN


def _detect_peaks(
    background_signal: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    background_signal: dim = (time,range)
    radial_distance: dim = (range,)

    Returns a boolean mask, dim = (range, ), where True denotes locations of peaks
    that should be ignored in fitting
    """
    scale = np.median(background_signal, axis=1)[:, np.newaxis]
    bg = background_signal / scale
    return _set_adjacent_true(
        np.concatenate(
            (
                np.array([False]),
                np.diff(np.diff(bg.mean(axis=0))) < -0.01,
                np.array([False]),
            )
        )
    )


def _set_adjacent_true(arr: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    temp = np.pad(arr, (1, 1), mode="constant")
    temp[:-2] |= arr
    temp[2:] |= arr
    return temp[1:-1]


def _linear_fit(
    bg_signal: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    dist_mask = 90 < radial_distance
    peaks = _detect_peaks(bg_signal, radial_distance)
    mask = dist_mask & ~peaks

    scale = np.median(bg_signal, axis=1)[:, np.newaxis]
    rdist_fit = radial_distance[np.newaxis][:, mask]
    signal_fit = (bg_signal / scale)[:, mask]

    A = np.tile(
        np.concatenate((rdist_fit, np.ones_like(rdist_fit))).T, (signal_fit.shape[0], 1)
    )
    x = np.linalg.pinv(A) @ signal_fit.reshape(-1, 1)
    fit = (
        np.concatenate(
            (radial_distance[:, np.newaxis], np.ones((radial_distance.shape[0], 1))),
            axis=1,
        )
        @ x
    ).T
    return np.array(fit * scale, dtype=np.float64)


def _exponential_fit(
    bg_signal: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    dist_mask = 90 < radial_distance
    peaks = _detect_peaks(bg_signal, radial_distance)
    mask = dist_mask & ~peaks
    scale = np.median(bg_signal, axis=1)[:, np.newaxis]
    rdist_fit = radial_distance[np.newaxis][:, mask]
    signal_fit = (bg_signal / scale)[:, mask]

    def exp_func_rss(x: npt.NDArray[np.float64]) -> np.float64:
        return np.float64(((signal_fit - _exp_func(x, rdist_fit)) ** 2).sum())

    result = scipy.optimize.minimize(
        exp_func_rss, [1, -1, -1], method="Nelder-Mead", options={"maxiter": 3 * 600}
    )
    fit = _exp_func(result.x, radial_distance)[np.newaxis, :]
    return np.array(fit * scale, dtype=np.float64)


def _exponential_linear_fit(
    bg_signal: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    dist_mask = 90 < radial_distance
    peaks = _detect_peaks(bg_signal, radial_distance)
    mask = dist_mask & ~peaks
    scale = np.median(bg_signal, axis=1)[:, np.newaxis]
    rdist_fit = radial_distance[np.newaxis][:, mask]
    signal_fit = (bg_signal / scale)[:, mask]

    def explin_func_rss(x: npt.NDArray[np.float64]) -> np.float64:
        return np.float64(((signal_fit - _explin_func(x, rdist_fit)) ** 2).sum())

    result = scipy.optimize.minimize(
        explin_func_rss,
        [1, -1, -1, 0, 0],
        method="Nelder-Mead",
        options={"maxiter": 5 * 600},
    )
    fit = _explin_func(result.x, radial_distance)[np.newaxis, :]
    return np.array(fit * scale, dtype=np.float64)


def _select_raws_for_stare(
    raws: Sequence[doppy.raw.HaloHpl],
) -> Sequence[doppy.raw.HaloHpl]:
    groups: dict[SelectionGroupKeyType, int] = defaultdict(int)

    if len(raws) == 0:
        raise doppy.exceptions.NoDataError("No data to select from")

    # Select files that stare
    raws_stare = [raw for raw in raws if len(raw.azimuth_angles) == 1]
    if len(raws_stare) == 0:
        raise doppy.exceptions.NoDataError(
            "No data suitable for stare product. Data is probably from scans"
        )
    raws_stare = [raw for raw in raws if len(raw.elevation_angles) == 1]
    if len(raws_stare) == 0:
        raise doppy.exceptions.NoDataError(
            "No data suitable for stare product. "
            "Elevation angle does not remain constant"
        )
    elevation_angles = []
    for raw in raws_stare:
        elevation_angles += list(raw.elevation_angles)
    max_elevation_angle = max(elevation_angles)

    ELEVATION_ANGLE_FLUCTUATION_THRESHOLD = 2
    ELEVATION_ANGLE_VERTICAL_OFFSET_THRESHOLD = 15

    raws_stare = [
        raw
        for raw in raws
        if abs(next(iter(raw.elevation_angles)) - max_elevation_angle)
        < ELEVATION_ANGLE_FLUCTUATION_THRESHOLD
        and abs(next(iter(raw.elevation_angles)) - 90)
        < ELEVATION_ANGLE_VERTICAL_OFFSET_THRESHOLD
    ]

    if len(raws_stare) == 0:
        raise doppy.exceptions.NoDataError("No data suitable for stare product")

    # count the number of profiles each (scan_type,ngates) group has
    for raw in raws_stare:
        groups[_selection_key(raw)] += len(raw.time)

    def key_func(key: SelectionGroupKeyType) -> int:
        return groups[key]

    # (scan_type,ngates, gate_points) group with the most profiles
    select_tuple = max(groups, key=key_func)

    return [raw for raw in raws_stare if _selection_key(raw) == select_tuple]


def _selection_key(raw: doppy.raw.HaloHpl) -> SelectionGroupKeyType:
    return (raw.header.mergeable_hash(),)


def _time2bg_time(
    time: npt.NDArray[np.datetime64], bg_time: npt.NDArray[np.datetime64]
) -> npt.NDArray[np.int64]:
    return np.searchsorted(bg_time, time, side="right") - 1


def _select_relevant_background_profiles(
    bg: doppy.raw.HaloBg, time: npt.NDArray[np.datetime64]
) -> doppy.raw.HaloBg:
    """
    expects bg.time to be sorted
    """
    time2bg_time = _time2bg_time(time, bg.time)

    relevant_indices = list(set(time2bg_time[time2bg_time >= 0]))
    bg_ind = np.arange(bg.time.size)
    is_relevant = np.isin(bg_ind, relevant_indices)
    return bg[is_relevant]


def _cluster_background_profiles(
    background_signal: npt.NDArray[np.float64], radial_distance: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    default_labels = np.zeros(len(background_signal), dtype=int)
    if len(background_signal) < 2:
        return default_labels
    radial_distance_mask = (90 < radial_distance) & (radial_distance < 1500)

    normalised_background_signal = background_signal / np.median(
        background_signal, axis=1, keepdims=True
    )

    profile_median = np.median(
        normalised_background_signal[:, radial_distance_mask], axis=1
    )
    kmeans = KMeans(n_clusters=2, n_init="auto").fit(profile_median[:, np.newaxis])
    cluster_width = np.array([None, None])
    for label in [0, 1]:
        cluster = profile_median[kmeans.labels_ == label]
        cluster_width[label] = np.max(cluster) - np.min(cluster)
    cluster_distance = np.abs(
        kmeans.cluster_centers_[0, 0] - kmeans.cluster_centers_[1, 0]
    )
    max_cluster_width = np.float64(np.max(cluster_width))
    if np.isclose(max_cluster_width, 0):
        return default_labels
    if cluster_distance / max_cluster_width > 3:
        return np.array(kmeans.labels_, dtype=np.int64)
    return default_labels
