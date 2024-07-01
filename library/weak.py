import numpy as np

function_phi = lambda chi, tau: (1 - tau ** 2) * (1 - chi ** 2) ** 4
function_phi_t = lambda chi, tau: -2 * tau * (1 - chi ** 2) ** 4
function_phi_x = lambda chi, tau: - 8 * (1 - tau ** 2) * chi * (1 - chi ** 2) ** 3
function_phi_xx = lambda chi, tau: - 8 * (1 - tau ** 2) * (1 - chi ** 2) ** 2 * (1 - 7 * chi ** 2)


def weak_integrate_time(t, x, u, n, m, I_x, I_t):
    d_t = t[1] - t[0]
    d_x = x[1] - x[0]

    delta_t = t[I_t - 1] - t[0]
    delta_x = x[I_x - 1] - x[0]

    d_tau = 2 * d_t / delta_t
    d_chi = 2 * d_x / delta_x

    tau = (2 * t[:I_t] - t[I_t - 1] - t[0]) / delta_t
    chi = (2 * x[:I_x] - x[I_x - 1] - x[0]) / delta_x

    phi_t = np.zeros((I_x, I_t))

    for i in range(I_x):
        phi_t[i, :] = function_phi_t(chi[i], tau)

    u_integrated = np.zeros((n // I_x, m // I_t))

    slices_t = np.arange(0, m, I_t)
    slices_x = np.arange(0, n, I_x)

    for i in range(len(slices_x) - 1):
        for j in range(len(slices_t) - 1):
            u_integrated[i, j] = - np.sum(
                u[slices_x[i]: slices_x[i + 1], slices_t[j]: slices_t[j + 1]] * phi_t
            ) * delta_x * 0.5 * d_chi * d_tau

    return u_integrated
