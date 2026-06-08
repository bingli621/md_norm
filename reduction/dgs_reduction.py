import numpy as np
import scipp as sc
from utils import *


def determine_INS_windows(
    monitor_data, detecor_positions, energy_transfer_ratio=(-0.8, 0.8)
) -> sc.DataArray:
    """Determine INS time windows per detector pixel, per RRM.
    Keep E = Ei - Ef = ±0.8*Ei by default."""

    sample_to_detectors = sc.norm(
        detecor_positions - monitor_data.coords["sample_position"]
    )
    ei = monitor_data.coords["ei"]
    time_on_sample = monitor_data.coords["time_on_sample"]
    r_gain, r_loss = energy_transfer_ratio
    ef_gain = ei * (1 - r_gain)
    ef_loss = ei * (1 - r_loss)
    vf_gain = energy_to_speed(ef_gain)
    vf_loss = energy_to_speed(ef_loss)
    t_gain = time_on_sample + sample_to_detectors / vf_gain
    t_loss = time_on_sample + sample_to_detectors / vf_loss

    # TODO
    # modify t_gain/t_loss,
    # if rrm >1, check overlap between subframes
    # check for source period, LET has a source frequency of 10 Hz

    norm_factors = sc.DataArray(
        data=sc.zeros(sizes=detecor_positions.sizes | monitor_data.sizes),
        coords={
            "time_on_sample": time_on_sample,
            "toa_min": t_gain,
            "toa_max": t_loss,
            "sample_position": monitor_data.coords["sample_position"],
            "detector_positions": detecor_positions,
            "ki": monitor_data.coords["ki"],
            "ei": monitor_data.coords["ei"],
            "monitor_counts": monitor_data.data,
        },
    )

    return norm_factors


def monitor_single_pulse(tof_monitor):
    """Return monitor TOA COM and counts"""
    counts = sc.array(
        dims=["rrm"],
        values=[
            tof_monitor.Intensity.sum(),
        ],
        unit="counts",
    )
    toa_com = np.sum(tof_monitor.Intensity * tof_monitor.xaxis) / np.sum(
        tof_monitor.Intensity
    )
    toa_com = sc.array(
        dims=["rrm"],
        values=[
            toa_com,
        ],
        unit="us",
    )
    data = sc.DataArray(
        data=counts, coords={"time_on_monitor": sc.to_unit(toa_com, "s")}
    )
    return data


# TODO unwrap frame
def assign_rrm(events, norm_factors):
    """Assgin RRM index of TOA based on (toa_min, toa_max), mask data in-between
    Note:  the periodicity of events are not considered at the moment"""
    toa_min = norm_factors.coords["toa_min"]
    toa_max = norm_factors.coords["toa_max"]

    pid = events.coords["pixel_id"]
    toa = events.coords["toa"]

    # Select the (rrm,) slice corresponding to each event pixel
    # dims = (events, rrm)
    tmin = toa_min["pixel_id", pid].rename_dims({"pixel_id": "events"})
    tmax = toa_max["pixel_id", pid].rename_dims({"pixel_id": "events"})

    # ------------------
    # Assign RRM from toa_min
    # ------------------
    passed = toa >= tmin  # dims=(events, rrm)

    rrm = passed.values.sum(axis=1) - 1
    n_rrm = tmin.sizes["rrm"]

    # Wrap around: toa < first toa_min -> largest rrm
    rrm[rrm < 0] = n_rrm - 1

    events.coords["rrm"] = sc.array(dims=["events"], values=rrm)

    # check multiple matches
    # n_matches = inside.values.sum(axis=1)

    # ------------------
    # Mask invalid events
    # ------------------
    inside = (toa >= tmin) & (toa < tmax)  # dims=(events, rrm)
    mask = sc.any(inside, dim="rrm")
    events.masks["outside"] = ~mask

    # ------------------
    # Add coordinates to events
    # ------------------

    events.coords["ei"] = sc.array(
        dims=["events"],
        values=norm_factors.coords["ei"].values[rrm],
        unit=norm_factors.coords["ei"].unit,
    )
    events.coords["ki"] = sc.vectors(
        dims=["events"],
        values=norm_factors.coords["ki"].values[rrm],
        unit=norm_factors.coords["ki"].unit,
    )

    events.coords["time_on_sample"] = sc.array(
        dims=["events"],
        values=norm_factors.coords["time_on_sample"].values[rrm],
        unit=norm_factors.coords["time_on_sample"].unit,
    )

    events.coords["sample_position"] = norm_factors.coords["sample_position"]
    events.coords["detector_positions"] = sc.vectors(
        dims=["events"],
        values=norm_factors.coords["detector_positions"].values[pid.values],
        unit=norm_factors.coords["detector_positions"].unit,
    )

    return events


# -----------------------------------------------------
#  generate bin boxes
# -----------------------------------------------------


def _make_bins(spec, dim, unit="1/Å"):
    if not isinstance(spec, tuple):
        raise TypeError(f"{dim} must be a tuple of length 2 or 3")

    if len(spec) == 2:
        start, stop = spec
        step = stop - start

    elif len(spec) == 3:
        start, stop, step = spec

    else:
        raise ValueError(f"{dim} tuple must have length 2 or 3")

    return sc.arange(dim=dim, start=start, stop=stop + step, step=step, unit=unit)


def generate_bins(qx=None, qy=None, qz=None, en=None):
    """
    Generate Scipp bin edges for x, y, z, and en.

    Each argument must be:
      - (min, max)
      - (start, stop, step)

    Returns
    -------
    dict[str, sc.Variable]
        Bin edges per dimension
    """
    bins = {}

    if qx is not None:
        bins["qx"] = _make_bins(qx, "qx")

    if qy is not None:
        bins["qy"] = _make_bins(qy, "qy")

    if qz is not None:
        bins["qz"] = _make_bins(qz, "qz")

    if en is not None:
        bins["en"] = _make_bins(en, "en", unit="meV")

    return bins


calculate_ei = {
    "time_on_source": time_on_source,
    "vi": vi_from_one_monitor,
    "ei": ei,
    "time_on_sample": time_on_sample,
    "ki": ki,
}


calculate_qe = {
    # "sample_to_detectors": sample_to_detectors,
    "kf_unit_vec": kf_unit_vec_from_positions,
    "vf": vf,
    "ef": ef,
    "kf": kf,
    "q": momentum_transfer,
    "q_mag": q_mag,
    "qx": qx,
    "qy": qy,
    "qz": qz,
    "en": energy_transfer,
}


calculate_trajectory_endpoints = {
    # "vf_gain": vf_gain,
    # "vf_loss": vf_loss,
    "ef_gain": ef_gain,
    "ef_loss": ef_loss,
    "energy_gain_ratio": en_gain_ratio,
    "energy_loss_ratio": en_loss_ratio,
    "kf_m_mag": kf_m_mag,
    "kf_M_mag": kf_M_mag,
    "kf_unit_vec": kf_unit_vec_from_positions,
    "kf_m": kf_m,
    "kf_M": kf_M,
    "q_m": q_m,
    "q_M": q_M,
}


# dgs_reduction = calculate_ei | calculate_qe
