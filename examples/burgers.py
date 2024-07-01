import scipy.io as sio
import matplotlib.pyplot as plt
import pysindy as ps
import numpy as np

from library.derivative import centered_finite_difference
from library.library import print_equation
from library.regression import stls
from library.trapezoidal_rule import tintegrate2d

if __name__ == '__main__':
    """
    Burgers equation:
    u_t = - uu_x + 0.1 * u_xx 
    """

    """
    Read Data
    """
    data = sio.loadmat('../data/burgers.mat')

    t = np.ravel(data["t"])
    x = np.ravel(data["x"])
    u = np.real(data["usol"])
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    """
    Intervals & Axes
    """
    intervals = [5, 5]
    axes = [x, t]

    """
    Compute u_t
    """

    n, m = u.shape

    u_t = np.zeros((n, m))

    for i in range(n):
        u_t[i, :] = centered_finite_difference(u[i, :], delta=dt, order=1)

    u_t_integrated = tintegrate2d(u_t, axes, intervals)

    ni, mi = u_t_integrated.shape

    u_t_integrated = np.reshape(u_t_integrated, (ni * mi), order='F')

    """
    Crete Theta matrix
    """
    polynomial_order = 2
    derivative_order = 2

    Theta_integrated = np.zeros((ni * mi, (polynomial_order + 1) * (derivative_order + 1)))

    u_x = np.zeros((n, m))

    for d in range(derivative_order + 1):
        if d > 0:
            for i in range(m):
                u_x[:, i] = centered_finite_difference(u[:, i], delta=dx, order=d)
        else:
            u_x = np.ones((n, m))

        for p in range(polynomial_order + 1):
            u_x_integrated = tintegrate2d(np.multiply(u_x, np.power(u, p)), axes, intervals)

            Theta_integrated[:, d * (polynomial_order + 1) + p] = np.reshape(u_x_integrated, (ni * mi), order='F')

            print(f'derivative: {d}, polimomial: {p}')

    """
    Find Xi
    """
    threshold = 0.08

    alpha = 1e-5

    Xi = stls(Theta_integrated, u_t_integrated, alpha, threshold, iterations=100)

    print(Xi)

    terms = np.array(['1', 'u', 'u^2', 'u_x', 'uu_x', 'u^2u_x', 'u_xx', 'uu_xx', 'u^2u_xx'])

    print_equation(Xi, terms)

    # ---------------------------------------------------------------------------------------------------------------- #

    """
    Plot Data
    """

    data = sio.loadmat('../data/burgers.mat')
    t = np.ravel(data['t'])
    x = np.ravel(data['x'])
    u = np.real(data['usol'])
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    # Plot u and u_dot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(t, x, u)
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x', fontsize=16)
    plt.title(r'$u(x, t)$', fontsize=16)

    u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(t, x, u_dot)
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x', fontsize=16)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.title(r'$\dot{u}(x, t)$', fontsize=16)
    plt.savefig('../images/burgers.png', dpi=300)
    plt.show()

    u = u.reshape(len(x), len(t), 1)
    u_dot = u_dot.reshape(len(x), len(t), 1)

    normalize_columns = True

    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=3,
        spatial_grid=x,
        is_uniform=True,
    )

    print('STLSQ model: ')
    optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=normalize_columns)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()

    print('SR3 model, L0 norm: ')
    optimizer = ps.SR3(
        threshold=2,
        max_iter=10000,
        tol=1e-15,
        nu=1e2,
        thresholder="l0",
        normalize_columns=normalize_columns,
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()

    print('SR3 model, L1 norm: ')
    optimizer = ps.SR3(
        threshold=0.5, max_iter=10000, tol=1e-15,
        thresholder="l1", normalize_columns=normalize_columns
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()

    print('SSR model: ')
    optimizer = ps.SSR(normalize_columns=normalize_columns, kappa=1)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()

    print('SSR (metric = model residual) model: ')
    optimizer = ps.SSR(criteria="model_residual",
                       normalize_columns=normalize_columns,
                       kappa=1)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()

    print('FROLs model: ')
    optimizer = ps.FROLS(normalize_columns=normalize_columns, kappa=1e-3)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()
