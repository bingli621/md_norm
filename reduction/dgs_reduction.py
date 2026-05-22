import numpy as np
import mcstasscript as ms
import mcstastox as mx
import scipp as sc
from scipy.ndimage import label


# generate bin boxes


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


# Utility Functions


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
    coms = sc.array(dims=["tof"], values=coms, unit="s")
    return coms


# ----------------------- Sciline Functions --------------------------------


def source_to_monitor(source_position, monitor_position):
    """ """
    d_sm = (monitor_position).to(unit="m") - (source_position).to(unit="m")
    return sc.norm(d_sm)


# -------- ki ---------


def unit_vec_ki(monitor_position, sample_position):
    """ """
    d = sample_position - monitor_position
    unit_vec_ki = d / sc.norm(d)
    return unit_vec_ki


def vi_from_one_monitor(monitor_position, time_on_monitor):
    """ """
    vi = sc.norm(monitor_position) / time_on_monitor
    return vi


def mag_ki(vi):
    """ """
    mag_ki = (sc.constants.neutron_mass * vi) / sc.constants.hbar
    return sc.to_unit(mag_ki, "1/Å")


def vec_ki(mag_ki, unit_vec_ki):
    """ """
    return mag_ki * unit_vec_ki


# -------- kf ---------

def sample_to_detectors(sample_position, detector_positions):
    d = detector_positions - sample_position
    return sc.norm(d)


def unit_vec_kf(detector_positions, sample_position):
    """ """
    d = detector_positions - sample_position
    unit_kf = d / sc.norm(d)

    return unit_kf


def time_on_sample(sample_position, monitor_position, time_on_monitor, vi):
    """ """
    time_on_sample = time_on_monitor + sc.norm(sample_position - monitor_position) / vi
    return time_on_sample



def vec_vf(sample_to_detectors, tof, time_on_sample):
    """ """
    vf = sample_to_detectors / (tof - time_on_sample)
    return vf


def mag_kf(vf):
    """ """
    mag_kf = (sc.constants.neutron_mass * vf) / sc.constants.hbar
    return sc.to_unit(mag_kf, "1/Å")


def vec_kf(mag_kf, unit_vec_kf):
    """ """
    return mag_kf * unit_vec_kf


# --------- Q and en --------


def vec_Q(vec_ki, vec_kf):
    """ """
    return vec_ki - vec_kf


def mag_Q(vec_Q):
    """ """
    return sc.to_unit(sc.norm(vec_Q), "1/Å")


def energy_transfer(mag_kf, mag_ki):
    """ """
    en = (sc.constants.hbar**2 / (2 * sc.constants.neutron_mass)) * (
        mag_ki**2 - mag_kf**2
    )
    return sc.to_unit(en, "meV")


dgs_reduction = {
    "source_to_monitor": source_to_monitor,
    "sample_to_detectors": sample_to_detectors,
    "unit_vec_ki": unit_vec_ki,
    "vi": vi_from_one_monitor,
    "mag_ki": mag_ki,
    "vec_ki": vec_ki,
    "unit_vec_kf": unit_vec_kf,
    "vf": vec_vf,
    "mag_kf": mag_kf,
    "vec_kf": vec_kf,
    "vec_Q": vec_Q,
    "mag_Q": mag_Q,
    "en": energy_transfer,
    "time_on_sample": time_on_sample,
}
