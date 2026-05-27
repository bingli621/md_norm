import numpy as np
import scipp as sc
import scipp.constants as consts
from scipy.ndimage import label


# Utility Functions


# FIXME
def _calc_pulse_centroid(tof_monitor, threshold=0, to_s=1e-6):
    # Assumes all pulses are evenly spaced in xaxis - but isn't bin data??? confirm this
    # Confirm correct way to axis tof data - is xaxis correct?????
    mask = tof_monitor.Intensity != threshold
    labels, num_features = label(mask)
    weighted_sum = np.bincount(
        labels, weights=tof_monitor.xaxis * tof_monitor.Intensity
    )[1:]
    weight_total = np.bincount(labels, weights=tof_monitor.Intensity)[1:]

    coms = weighted_sum / weight_total * to_s
    # coms = sc.array(dims=["tof"], values=coms, unit="s")
    return coms[0]


# -----------------------------------------------------
# Calculate Ei
# -----------------------------------------------------


def source_to_monitor(source_position, monitor_position):
    """ """
    d_sm = (monitor_position).to(unit="m") - (source_position).to(unit="m")
    return sc.norm(d_sm)


def unit_vec_ki(monitor_position, sample_position):
    """ """
    d = sample_position - monitor_position
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


def monitor_to_sample(
    sample_position,
    monitor_position,
):
    d = sample_position.to(unit="m") - monitor_position.to(unit="m")
    return sc.norm(d)


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
# Calculate Ef
# -----------------------------------------------------


def sample_to_detectors(sample_position, detector_positions):
    d = detector_positions - sample_position
    return sc.norm(d)





def unit_vec_kf(detector_positions, sample_position):
    """ """
    d = detector_positions - sample_position
    unit_kf = d / sc.norm(d)

    return unit_kf


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


calculate_ef = {
    "sample_to_detectors": sample_to_detectors,
    "unit_vec_kf": unit_vec_kf,
    "vf": vec_vf,
    "ef": ef,
    "mag_kf": mag_kf,
    "vec_kf": vec_kf,
}


# --------- Q and en --------


def vec_Q(vec_ki, vec_kf):
    """ """
    return vec_ki - vec_kf


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
        "vec_Q": vec_Q,
        "mag_Q": mag_Q,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "en": energy_transfer,
    }
)
