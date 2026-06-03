import numpy as np
import scipp as sc
import scipp.constants as consts

# Utility Functions


def energy_to_momentum(en):
    k = sc.sqrt(2 * consts.m_n * en) / consts.hbar
    return k.to(unit="1/Å")


def energy_to_speed(en):
    v = sc.sqrt(2 * en / consts.m_n)
    return v.to(unit="m/s")


def speed_to_energy(v):
    en = 0.5 * consts.m_n * v**2
    return en.to(unit="meV")


def distance_between(position_1, position_2):
    d = position_1.to(unit="m") - position_2.to(unit="m")
    return sc.norm(d)


def speed_between(position0, position1, time0, time1):
    return sc.norm((position0 - position1)) / sc.abs((time0 - time1))


def sample_to_detectors(sample_position, detector_positions):
    return distance_between(sample_position, detector_positions)


# ------------------------------------------------------------
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
            "toa_min": t_gain,
            "toa_max": t_loss,
            "sample_position": monitor_data.coords["sample_position"],
            "time_on_sample": time_on_sample,
            "detector_positions": detecor_positions,
            "ei": monitor_data.coords["ei"],
            "vec_ki": monitor_data.coords["vec_ki"],
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


# -----------------------------------------------------
# Calculate Ei
# -----------------------------------------------------


def source_to_monitor(source_position, presample_monitor_position):
    return distance_between(source_position, presample_monitor_position)


def unit_vec_ki(presample_monitor_position, sample_position):
    """ """
    d = sample_position - presample_monitor_position
    unit_vec_ki = d / sc.norm(d)
    return unit_vec_ki


def vi_from_one_monitor(source_to_monitor, time_on_monitor):
    """ """
    vi = source_to_monitor / time_on_monitor
    return vi


def ei(vi):
    ei = 0.5 * consts.m_n * vi**2
    return sc.to_unit(ei, "meV")


def mag_ki(vi):
    """ """
    mag_ki = (consts.m_n * vi) / consts.hbar
    return sc.to_unit(mag_ki, "1/Å")


def monitor_to_sample(sample_position, presample_monitor_position):
    return distance_between(sample_position, presample_monitor_position)


def time_on_sample(monitor_to_sample, time_on_monitor, vi):
    """ """
    time_on_sample = time_on_monitor + monitor_to_sample / vi
    return time_on_sample


def vec_ki(mag_ki, unit_vec_ki):
    """ """
    return mag_ki * unit_vec_ki


calculate_ei = {
    "source_to_monitor": source_to_monitor,
    "vi": vi_from_one_monitor,
    "ei": ei,
    "monitor_to_sample": monitor_to_sample,
    "time_on_sample": time_on_sample,
    "unit_vec_ki": unit_vec_ki,
    "mag_ki": mag_ki,
    "vec_ki": vec_ki,
}


# -----------------------------------------------------
# Calculate detector trajectory endpoints
# -----------------------------------------------------
def vf_gain(sample_position, detector_positions, time_on_sample, toa_min):
    return speed_between(sample_position, detector_positions, time_on_sample, toa_min)


def vf_loss(sample_position, detector_positions, time_on_sample, toa_max):
    return speed_between(sample_position, detector_positions, time_on_sample, toa_max)


def ef_gain(vf_gain):
    return speed_to_energy(vf_gain)


def ef_loss(vf_loss):
    return speed_to_energy(vf_loss)


def en_gain_ratio(ei, ef_gain):
    return 1 - ef_gain / ei


def en_loss_ratio(ei, ef_loss):
    return 1 - ef_loss / ei


def kf_m(energy_loss_ratio, ei):
    ef = (1 - energy_loss_ratio) * ei
    return energy_to_momentum(ef)


def kf_M(energy_gain_ratio, ei):
    ef = (1 - energy_gain_ratio) * ei
    return energy_to_momentum(ef)


calculate_ef = {
    "vf_gain": vf_gain,
    "vf_loss": vf_loss,
    "ef_gain": ef_gain,
    "ef_loss": ef_loss,
    "energy_gain_ratio": en_gain_ratio,
    "energy_loss_ratio": en_loss_ratio,
    "kf_m": kf_m,
    "kf_M": kf_M,
}


def unit_vec_kf(detector_positions, sample_position):
    """ """
    d = detector_positions - sample_position
    unit_kf = d / sc.norm(d)

    return unit_kf


def vec_kf_m(kf_m, unit_vec_kf):
    return kf_m * unit_vec_kf


def vec_kf_M(kf_M, unit_vec_kf):
    return kf_M * unit_vec_kf


def vec_q_m(vec_ki, vec_kf_m):
    return vec_ki - vec_kf_m


def vec_q_M(vec_ki, vec_kf_M):
    return vec_ki - vec_kf_M


def qx_m(vec_q_m):
    return vec_q_m.fields.x.copy()


def qy_m(vec_q_m):
    return vec_q_m.fields.y.copy()


def qz_m(vec_q_m):
    return vec_q_m.fields.z.copy()


def qx_M(vec_q_M):
    return vec_q_M.fields.x.copy()


def qy_M(vec_q_M):
    return vec_q_M.fields.y.copy()


def qz_M(vec_q_M):
    return vec_q_M.fields.z.copy()


calculate_trajectory_endpoints = {
    "unit_vec_kf": unit_vec_kf,
    "vec_kf_m": vec_kf_m,
    "vec_kf_M": vec_kf_M,
    "vec_q_m": vec_q_m,
    "vec_q_M": vec_q_M,
    "qx_m": qx_m,
    "qy_m": qy_m,
    "qz_m": qz_m,
    "qx_M": qx_M,
    "qy_M": qy_M,
    "qz_M": qz_M,
}

# -----------------------------------------------------
# Calculate Ef
# -----------------------------------------------------


def vec_vf(sample_to_detectors, tof, time_on_sample):
    """ """
    vf = sample_to_detectors / (tof - time_on_sample)
    return vf


def mag_kf(vf):
    """ """
    mag_kf = (consts.m_n * vf) / consts.hbar
    return sc.to_unit(mag_kf, "1/Å")


def ef(vf):
    ef = 0.5 * consts.m_n * vf**2
    return sc.to_unit(ef, "meV")


def vec_kf(mag_kf, unit_vec_kf):
    """ """
    return mag_kf * unit_vec_kf


# calculate_ef = {
#     "sample_to_detectors": sample_to_detectors,
#     "unit_vec_kf": unit_vec_kf,
#     "vf": vec_vf,
#     "ef": ef,
#     "mag_kf": mag_kf,
#     "vec_kf": vec_kf,
# }


# --------- Q and en --------


def qx(vec_Q):
    return vec_Q.fields.x.copy()


def qy(vec_Q):
    return vec_Q.fields.y.copy()


def qz(vec_Q):
    return vec_Q.fields.z.copy()


def mag_Q(vec_Q):
    """ """
    return sc.to_unit(sc.norm(vec_Q), "1/Å")


def energy_transfer(ei, ef):
    """ """
    en = ei - ef
    return sc.to_unit(en, "meV")


dgs_reduction = (
    calculate_ei
    | calculate_ef
    | {
        "mag_Q": mag_Q,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "en": energy_transfer,
    }
)
