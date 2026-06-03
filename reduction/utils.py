import scipp as sc
import scipp.constants as consts

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------


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


# -----------------------------------------------------
# Calculate (Q,E) from (pixel_id, toa)
# -----------------------------------------------------


def vec_vf(sample_to_detectors, toa, time_on_sample):
    """ """
    vf = sample_to_detectors / (toa - time_on_sample)
    return vf


def unit_vec_kf(detector_positions, sample_position):
    """ """
    d = detector_positions - sample_position
    unit_kf = d / sc.norm(d)

    return unit_kf


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


def vec_q(vec_ki, vec_kf):
    return vec_ki - vec_kf


def qx(vec_q):
    return vec_q.fields.x.copy()


def qy(vec_q):
    return vec_q.fields.y.copy()


def qz(vec_q):
    return vec_q.fields.z.copy()


def mag_q(vec_q):
    """ """
    return sc.to_unit(sc.norm(vec_q), "1/Å")


def energy_transfer(ei, ef):
    """ """
    en = ei - ef
    return sc.to_unit(en, "meV")


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


def kf_m(ef_loss):
    return energy_to_momentum(ef_loss)


def kf_M(ef_gain):
    return energy_to_momentum(ef_gain)


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
