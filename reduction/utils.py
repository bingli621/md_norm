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


def velocity_to_energy(v):
    en = speed_to_energy(sc.norm(v))
    return en.to(unit="meV")


def velocity_to_momentum(v):
    k = consts.m_n * v / consts.hbar
    return k.to(unit="1/Å")


def displacement_between(position0, position1):
    """Displacement from 0 to 1"""
    return position1.to(unit="m") - position0.to(unit="m")


def distance_between(position0, position1):
    """Distance from 0 to 1"""
    return sc.norm(displacement_between(position0, position1))


def velocity_between(position0, time0, position1, time1):
    """Velocity from 0 to 1"""
    return displacement_between(position0, position1) / (time1 - time0)


def speed_between(position0, time0, position1, time1):
    """Speed from 0 to 1"""
    return distance_between(position0, position1) / (time1 - time0)


# -----------------------------------------------------
# Calculate Ei
# -----------------------------------------------------


def time_on_source():
    """Default time on time is 0.0 ms"""
    t0 = sc.scalar(value=0.0, unit="ms")
    return t0.to(unit="s")


def vi_from_one_monitor(
    source_position, presample_monitor_position, time_on_monitor, time_on_source
):
    return velocity_between(
        source_position,
        time_on_source,
        presample_monitor_position,
        time_on_monitor,
    )


def ki(vi):
    return velocity_to_momentum(vi)


def ei(vi):
    return velocity_to_energy(vi)


def time_on_sample(presample_monitor_position, time_on_monitor, sample_position, vi):
    time_on_sample = time_on_monitor + distance_between(
        presample_monitor_position, sample_position
    ) / sc.norm(vi)
    return time_on_sample


# -----------------------------------------------------
# Calculate (Q,E) from (pixel_id, toa)
# -----------------------------------------------------


def vf(sample_position, time_on_sample, detector_positions, toa):
    return velocity_between(sample_position, time_on_sample, detector_positions, toa)


def kf(vf):
    return velocity_to_momentum(vf)


def ef(vf):
    return velocity_to_energy(vf)


def momentum_transfer(ki, kf):
    """Momentum transfer Q defined as ki - kf"""
    return ki - kf


def qx(q):
    return q.fields.x.copy()


def qy(q):
    return q.fields.y.copy()


def qz(q):
    return q.fields.z.copy()


def q_mag(q):
    """ """
    return sc.norm(q)


def energy_transfer(ei, ef):
    """Energy transfer en defined as Ei - Ef"""
    return ei - ef


# -----------------------------------------------------
# Calculate detector trajectory endpoints
# -----------------------------------------------------


def kf_unit_vec_from_positions(sample_position, detector_positions):
    d = displacement_between(sample_position, detector_positions)
    return d / sc.norm(d)


def ef_gain(sample_position, time_on_sample, detector_positions, toa_min):
    vf_gain = speed_between(
        sample_position, time_on_sample, detector_positions, toa_min
    )
    return speed_to_energy(vf_gain)


def ef_loss(sample_position, detector_positions, time_on_sample, toa_max):
    vf_loss = speed_between(
        sample_position, time_on_sample, detector_positions, toa_max
    )
    return speed_to_energy(vf_loss)


def en_gain_ratio(ei, ef_gain):
    return 1 - ef_gain / ei


def en_loss_ratio(ei, ef_loss):
    return 1 - ef_loss / ei


def kf_m_mag(ef_loss):
    return energy_to_momentum(ef_loss)


def kf_M_mag(ef_gain):
    return energy_to_momentum(ef_gain)


def kf_m(kf_m_mag, kf_unit_vec):
    return kf_m_mag * kf_unit_vec


def kf_M(kf_M_mag, kf_unit_vec):
    return kf_M_mag * kf_unit_vec


def q_m(ki, kf_m):
    return ki - kf_m


def q_M(ki, kf_M):
    return ki - kf_M
