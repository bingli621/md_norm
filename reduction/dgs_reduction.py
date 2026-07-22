import numpy as np
import scipp as sc
from scipy.signal import find_peaks

from reduction._single_crystal import compute_q_de_norm
from reduction.utils import *


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


def monitor_single_pulse(tof_monitor, unit="us"):
    """Return monitor TOA COM and counts, time unit is microsecond"""
    counts = sc.array(dims=["rrm"], values=[tof_monitor.Intensity.sum()], unit="counts")
    toa_com = np.sum(tof_monitor.Intensity * tof_monitor.xaxis) / np.sum(
        tof_monitor.Intensity
    )
    toa_com = sc.array(dims=["rrm"], values=[toa_com], unit=unit)
    data = sc.DataArray(
        data=counts, coords={"time_on_monitor": sc.to_unit(toa_com, "s")}
    )
    return data


# FIXME
# FIXME unit inconsistency
def monitor_multi_pulse(tof_monitor, rrm=22, threshold_fraction=0.02, unit="us"):
    """
    Detect (Hopefully!) all RRM pulses in the pre-sample monitor return centroid ToA like single_pulse function.

    Uses find_peaks with adaptive distance to avoid splitting broad
    peaks or merging close neighbours.

    Parameters
    ----------
    tof_monitor        : McStas monitor object (xaxis in seconds, Intensity)
    rrm                : RRM factor — used to estimate expected peak spacing.
                         Note: if M1 runs at half speed (by design to reduce
                         frame overlap), the actual number of transmitted
                         frames is RRM/2, and this function accounts for that.
    threshold_fraction : Peak prominence. Hopefuly reduces the errors on 'noisy data'

    Final result is hopefully an accurate array of RRMs that can be reduced?
    """
    intensity = np.asarray(tof_monitor.Intensity, dtype=float)  # intensity in monitor
    xaxis = np.asarray(tof_monitor.xaxis, dtype=float)  # seconds

    n_bins = len(intensity)
    total_time = xaxis[-1] - xaxis[0]

    # Expected number of frames is set to RRM/2 since M1 runs at half speed
    # n_expected = rrm // 2
    n_expected = rrm

    # Minimum bin separation = half the expected inter-peak spacing
    # To prevent two local maxima on one peak being counted separately
    # bins_per_frame = n_bins / n_expected
    # min_distance = max(
    #     3, int(bins_per_frame * 0.4)
    # )  # Either 40% or 3 bins, depending on how coarsely you bin your data
    min_distance = 1

    # Prominence threshold: ignore peaks smaller than fraction of global max
    prominence = intensity.max() * threshold_fraction

    peaks, props = find_peaks(
        intensity,
        prominence=prominence,
        distance=min_distance,
        width=2,  # must span at least 2 bins — filters single-bin noise spikes
    )

    if len(peaks) == 0:
        raise ValueError(
            f"No peaks found in monitor spectrum. "
            f"Try lowering threshold_fraction (currently {threshold_fraction})."
        )

    # Part below spits out the expected and detected ToAs
    # print(f"  Expected ~{n_expected} frames (RRM={rrm}, M1 at half speed)")
    print(f"  Expected ~{n_expected} frames (RRM={rrm})")
    print(f"  Detected {len(peaks)} peaks")
    print(
        f"  Min distance used: {min_distance} bins ({min_distance / n_bins * total_time * 1e6:.1f} µs)"
    )

    # For each detected peak, find its connected region above half-prominence
    # and compute the intensity-weighted centroid within that region
    half_prom = props["prominences"] / 2
    left_bases = props["left_bases"]
    right_bases = props["right_bases"]

    centroids_s = []
    counts = []

    for i, peak_idx in enumerate(peaks):
        # Integration window: use the peak's base-to-base region
        lo = left_bases[i]
        hi = right_bases[i] + 1  # +1 for inclusive slice

        region_x = xaxis[lo:hi]
        region_I = intensity[lo:hi]

        # Only include bins above half the local background
        # (avoids including the flat tail between peaks in the centroid)
        local_bg = min(intensity[lo], intensity[hi - 1])
        above_bg = region_I > local_bg
        if above_bg.sum() < 1:
            above_bg = np.ones_like(above_bg, dtype=bool)

        centroid = np.sum(region_x[above_bg] * region_I[above_bg]) / np.sum(
            region_I[above_bg]
        )
        centroids_s.append(centroid)
        counts.append(region_I[above_bg].sum())

        print(f"    Peak {i:2d}: t={centroid * 1e6:.2f} µs  counts={counts[-1]:.3e}")

    centroids_s = np.array(centroids_s)
    counts_arr = np.array(counts)

    # Sanity check: warn if peak count is far from expected
    if abs(len(peaks) - n_expected) > 2:
        print(
            f"  WARNING: detected {len(peaks)} peaks but expected ~{n_expected}. "
            f"Consider adjusting threshold_fraction or checking the monitor data."
        )

    toa_com = sc.array(dims=["rrm"], values=centroids_s, unit="s")
    counts = sc.array(dims=["rrm"], values=counts_arr, unit="counts")

    return sc.DataArray(data=counts, coords={"time_on_monitor": toa_com})


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
    # tmin = toa_min["pixel_id", pid].rename_dims({"pixel_id": "events"})
    # tmax = toa_max["pixel_id", pid].rename_dims({"pixel_id": "events"})

    pid_vals = pid.values
    toa_vals = toa.values

    tmin_vals = toa_min.values[pid_vals]  # shape: (events, rrm)
    tmax_vals = toa_max.values[pid_vals]

    rrm = (toa_vals[:, None] >= tmin_vals).sum(axis=1) - 1
    rrm %= tmin_vals.shape[1]

    # ------------------
    # Assign RRM from toa_min
    # ------------------
    # passed = toa >= tmin  # dims=(events, rrm)

    # rrm = passed.values.sum(axis=1) - 1
    # n_rrm = tmin.sizes["rrm"]

    # # Wrap around: toa < first toa_min -> largest rrm
    # rrm[rrm < 0] = n_rrm - 1

    events.coords["rrm"] = sc.array(dims=["events"], values=rrm)

    # check multiple matches
    # n_matches = inside.values.sum(axis=1)

    # ------------------
    # Mask invalid events
    # ------------------
    # inside = (toa >= tmin) & (toa < tmax)  # dims=(events, rrm)
    # mask = sc.any(inside, dim="rrm")
    # events.masks["outside"] = ~mask

    row = np.arange(rrm.size)

    valid = (toa_vals >= tmin_vals[row, rrm]) & (toa_vals < tmax_vals[row, rrm])
    events.masks["outside"] = sc.array(dims=["events"], values=~valid)

    # ------------------
    # Add coordinates to events
    # ------------------
    events.coords["sample_position"] = norm_factors.coords["sample_position"]

    coords = norm_factors.coords

    events.coords["ei"] = sc.array(
        dims=["events"], values=coords["ei"].values[rrm], unit=coords["ei"].unit
    )
    events.coords["ki"] = sc.vectors(
        dims=["events"], values=coords["ki"].values[rrm], unit=coords["ki"].unit
    )
    events.coords["time_on_sample"] = sc.array(
        dims=["events"],
        values=coords["time_on_sample"].values[rrm],
        unit=coords["time_on_sample"].unit,
    )
    events.coords["detector_positions"] = sc.vectors(
        dims=["events"],
        values=coords["detector_positions"].values[pid.values],
        unit=coords["detector_positions"].unit,
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


def generate_bins(grid: dict):
    """
    Generate Scipp bin edges for x, y, z, and en.
    """
    bins = {}

    for key, value in grid.items():
        if key in ("qx", "qy", "qz"):
            unit = "1/Å"
        elif key == "en":
            unit = "meV"
        else:  # HKL are dimensionless
            unit = "dimensionless"
        bins[key] = _make_bins(value, key, unit)

    return bins


def generate_plot_coords_and_title(bins):
    plot_coords = []
    plot_title = []
    for key, value in bins.items():
        if value.size == 2:
            plot_title.append(
                f"{key}=({value.values[0]:.3g}, {value.values[1]:.3g}) [{value.unit}]"
            )
        else:
            plot_coords.append(key)
    if not plot_title:  # powder
        for key, value in bins.items():
            plot_title.append(
                f"{key}=({value.values[0]:.3g}, {value.values[-11]:.3g}) [{value.unit}]"
            )
    plot_title_str = "\n".join(
        ", ".join(plot_title[i : i + 2]) for i in range(0, len(plot_title), 2)
    )
    return plot_coords, plot_title_str


calculate_ei = {
    "time_on_source": time_on_source,
    "vi": vi_from_one_monitor,
    "ei": ei,
    "time_on_sample": time_on_sample,
    "ki": ki,
}


calculate_qe = {
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

calculate_solid_angle = {
    "detector_geometry": detector_geometry,
    "pixel_normal_vector": pixel_normal_vector,
    "pixel_solid_angle": calculate_rectangle_solid_angle,
}


def mdnorm(events, grid: dict, norm_factors, rrm=0):

    bins = generate_bins(grid)
    bins_tuple = (
        bins["qx"].rename(qx="h"),
        bins["qy"].rename(qy="k"),
        bins["qz"].rename(qz="l"),
        bins["en"].rename(en="energy_transfer"),
    )

    plot_coords, plot_title = generate_plot_coords_and_title(bins)

    ei = norm_factors.coords["ei"]["rrm", rrm]
    qm = norm_factors.coords["q_m"]["rrm", rrm]
    qM = norm_factors.coords["q_M"]["rrm", rrm]
    kf_m = norm_factors.coords["kf_m_mag"]["rrm", rrm]
    kf_M = norm_factors.coords["kf_M_mag"]["rrm", rrm]
    incident_flux = norm_factors.coords["monitor_counts"]["rrm", rrm]

    # convert from cross section to S(Q,E) by multiple ki/kf
    sqe_coords = {k: events.coords[k] for k in bins.keys()}
    sqe = sc.DataArray(
        data=events.data * sc.sqrt(ei / events.coords["ef"]), coords=sqe_coords
    )
    sqe_hist = sc.bin(sqe, **bins).hist()

    start = (
        sc.concat(
            [qm.fields.x.copy(), qm.fields.y.copy(), qm.fields.z.copy(), kf_m],
            dim="q-e",
        )
        .transpose(["pixel_id", "q-e"])
        .rename_dims(pixel_id="pixel")
    )
    stop = (
        sc.concat(
            [qM.fields.x.copy(), qM.fields.y.copy(), qM.fields.z.copy(), kf_M],
            dim="q-e",
        )
        .transpose(["pixel_id", "q-e"])
        .rename_dims(pixel_id="pixel")
    )

    norm = compute_q_de_norm(
        trajectory_start=start,
        trajectory_stop=stop,
        solid_angle=norm_factors.coords["d_omega"].rename_dims(pixel_id="pixel"),
        grid=bins_tuple,
        incident_energy=ei,
        incident_flux=incident_flux,
    )
    norm = norm.rename(h="qx", k="qy", l="qz", energy_transfer="en")

    data = (sqe_hist.squeeze().transpose()) / norm.squeeze()
    # fix a bug in the printing of steradian
    sc.units.aliases["counts/meV/sr"] = sc.units.Unit("counts/meV/sr")

    return data, plot_coords, plot_title
