import numpy as np


def trapezoid_rule_2d(u, axes, deltas):
    x, t = axes[0], axes[1]
    dx, dt = deltas[0], deltas[1]

    n, m = x.shape[0], t.shape[0]

    integral = 0.0

    for i in range(n):
        for j in range(m):
            if (i == 0 or i == n - 1) and (j == 0 or j == m - 1):
                weight = 1
            elif (i == 0 or i == n - 1) or (j == 0 or j == m - 1):
                weight = 2
            else:
                weight = 4

            integral += weight * u[i, j]

    integral *= (dx * dt) / 4

    return integral


def tintegrate2d(u, axes, intervals):
    # u(x,t)

    x, t = axes
    I_x, I_t = intervals

    n, m = u.shape[0], u.shape[1]

    dt = t[1] - t[0]
    dx = x[1] - x[0]

    u_integrated = np.zeros((n // I_x, m // I_t))

    slices_t = np.arange(0, m, I_t)
    slices_x = np.arange(0, n, I_x)

    for i in range(len(slices_x) - 1):
        for j in range(len(slices_t) - 1):
            u_integrated[i, j] = trapezoid_rule_2d(
                u[slices_x[i]: slices_x[i + 1], slices_t[j]: slices_t[j + 1]],
                [x[slices_x[i]: slices_x[i + 1]], t[slices_t[j]: slices_t[j + 1]]],
                [dx, dt]
            )

    return u_integrated


# -------------------------------------------------------------------------------------------------------------------- #

def trapezoid_rule_3d(u, axes, deltas):
    x, y, t = axes[0], axes[1], axes[2]
    dx, dy, dt = deltas[0], deltas[1], deltas[2]

    n, m, l = x.shape[0], y.shape[0], t.shape[0]

    integral = 0.0

    for i in range(n):
        for j in range(m):
            for k in range(l):
                if (i == 0 or i == n - 1) and (j == 0 or j == m - 1) and (k == 0 or k == l - 1):  # corners (8)
                    weight = 1
                elif ((i == 0 or i == n - 1) and (j == 0 or j == m - 1)) or (
                        (i == 0 or i == n - 1) and (k == 0 or k == l - 1)) or (
                        (j == 0 or j == m - 1) and (k == 0 or k == l - 1)):  # edges (12)
                    weight = 2
                elif (i == 0 or i == n - 1) or (j == 0 or j == m - 1) or (k == 0 or k == l - 1):  # faces (6)
                    weight = 4
                else:  # interior points
                    weight = 8

                integral += weight * u[i, j, k]

    integral *= (dx * dy * dt) / 8

    return integral


def tintegrate3d(u, axes, intervals):
    # u(x, y, t)

    x, y, t = axes
    I_x, I_y, I_t = intervals

    n, m, l = u.shape[0], u.shape[1], u.shape[2]

    dt = t[1] - t[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    u_integrated = np.zeros((n // I_x, m // I_y, l // I_t))

    slices_t = np.arange(0, m, I_t)
    slices_x = np.arange(0, n, I_x)
    slices_y = np.arange(0, n, I_y)

    for i in range(len(slices_x) - 1):
        for j in range(len(slices_y) - 1):
            for k in range(len(slices_t) - 1):
                u_integrated[i, j, k] = trapezoid_rule_3d(
                    u[slices_x[i]: slices_x[i + 1], slices_y[j]: slices_y[j + 1], slices_t[k]: slices_t[k + 1]],
                    [x[slices_x[i]: slices_x[i + 1]], y[slices_y[j]: slices_y[j + 1]], t[slices_t[k]: slices_t[k + 1]]],
                    [dx, dy, dt]
                )

    return u_integrated
