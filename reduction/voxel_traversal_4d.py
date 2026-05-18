import numpy as np


def clip_segment_to_box_4d(p0, p1, edges, eps=1e-12):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0

    box_min = np.array([e[0] for e in edges])
    box_max = np.array([e[-1] for e in edges])

    t0, t1 = 0.0, 1.0

    for i in range(4):
        if abs(d[i]) < eps:
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return None
            continue

        inv = 1.0 / d[i]
        ta = (box_min[i] - p0[i]) * inv
        tb = (box_max[i] - p0[i]) * inv

        if ta > tb:
            ta, tb = tb, ta

        t0 = max(t0, ta)
        t1 = min(t1, tb)

        if t0 > t1:
            return None

    return t0, t1


def voxel_traversal_4d(p0, p1, edges, eps=1e-12):
    """
    Yields for each crossed voxel:
    (index_tuple, entry_point, exit_point)
    """

    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0

    edges = [np.asarray(e, float) for e in edges]

    clipped = clip_segment_to_box_4d(p0, p1, edges, eps)
    if clipped is None:
        return

    t_enter, t_exit = clipped

    # ---- Ensure starting voxel is included ----
    p_start = p0 + (t_enter + eps) * d

    # Initial voxel index
    idx = np.array(
        [np.searchsorted(edges[i], p_start[i], side="right") - 1 for i in range(4)],
        dtype=int,
    )

    step = np.sign(d).astype(int)

    # Compute next boundary crossings
    tNext = np.zeros(4)

    for i in range(4):
        if abs(d[i]) < eps:
            tNext[i] = np.inf
            continue

        if step[i] > 0:
            boundary = edges[i][idx[i] + 1]
        else:
            boundary = edges[i][idx[i]]

        tNext[i] = (boundary - p0[i]) / d[i]

    t = t_enter

    while (
        np.all(idx >= 0)
        and all(idx[i] < len(edges[i]) - 1 for i in range(4))
        and t <= t_exit + eps
    ):
        axis = np.argmin(tNext)
        t_voxel_exit = min(tNext[axis], t_exit)

        entry = p0 + t * d
        exit_ = p0 + t_voxel_exit * d

        yield tuple(idx), entry, exit_

        if tNext[axis] > t_exit:
            break

        # Step to next voxel
        idx[axis] += step[axis]
        t = tNext[axis]

        if idx[axis] < 0 or idx[axis] >= len(edges[axis]) - 1:
            break

        # Update next boundary crossing for this axis
        if step[axis] > 0:
            boundary = edges[axis][idx[axis] + 1]
        else:
            boundary = edges[axis][idx[axis]]

        tNext[axis] = (boundary - p0[axis]) / d[axis]


def main():
    edges = [
        np.array([0.0, 0.5, 2.0, 3.0]),
        np.array([0.0, 1.0, 1.5, 4.0]),
        np.array([0.0, 0.2, 0.8, 1.0]),
        np.array([0.0, 2.0, 5.0]),
    ]

    p0 = [-0.2, 0.3, 0.1, 0.5]
    p1 = [2.8, 3.2, 0.9, 4.8]

    for idx, p_in, p_out in voxel_traversal_4d(p0, p1, edges):
        print("voxel: ({},{},{},{})".format(*idx))
        print("  entry:", p_in)
        print("  exit :", p_out)


if __name__ == "__main__":
    main()
