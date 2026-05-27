import numpy as np
import scipp as sc
import scipp.constants as consts


# --------- generate bin boxes -------------
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


def momentum_to_energy():
    pass


mdnorm = {}
