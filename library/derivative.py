import numpy as np

def centered_finite_difference(u, delta, order):
    """
    Takes dth derivative ... using 2nd order finite difference method (up to 3th order)

    u: ...
    delta: Grid spacing. Assumes uniform spacing
    order: Derivative order
    """

    n = u.size
    derivative = np.zeros(n, dtype=np.complex64)

    if order == 1:
        derivative = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * delta)

        derivative[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / delta
        derivative[n - 1] = (-3.0 / 2 * u[n - 1] + 2 * u[n - 2] - u[n - 3] / 2) / delta

        return derivative

    if order == 2:
        derivative = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (delta ** 2)

        derivative[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / (delta ** 2)
        derivative[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / (delta ** 2)

        return derivative

    if order == 3:
        derivative = (np.roll(u, -2, axis=0) * 0.5
                      - np.roll(u, -1, axis=0)
                      + np.roll(u, 1, axis=0)
                      - np.roll(u, 2, axis=0) * 0.5) / (delta ** 3)

        derivative[0] = (-2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]) / (delta ** 3)
        derivative[1] = (-2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]) / (delta ** 3)
        derivative[n - 1] = (2.5 * u[n - 1] - 9 * u[n - 2] + 12 * u[n - 3] - 7 * u[n - 4] + 1.5 * u[n - 5]) / (
                delta ** 3)
        derivative[n - 2] = (2.5 * u[n - 2] - 9 * u[n - 3] + 12 * u[n - 4] - 7 * u[n - 5] + 1.5 * u[n - 6]) / (
                delta ** 3)

        return derivative

    if order > 3:
        return centered_finite_difference(centered_finite_difference(u, delta, 3), delta, order - 3)