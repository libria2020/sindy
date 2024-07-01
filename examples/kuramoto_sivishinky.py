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
    Kuramoto Sivishinky equation:
    u_t = - uu_x - u_xx - u_xxxx
    """

    """
    Read Data
    """
    data = sio.loadmat('../data/kuramoto_sivishinky.mat')

    t = np.ravel(data["tt"])
    x = np.ravel(data["x"])
    u = np.real(data["uu"])
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    """
    Intervals & Axes
    """
    intervals = [26, 10]
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
    polinomial_order = 2
    derivative_oreder = 4

    Theta_integrated = np.zeros((ni * mi, (polinomial_order + 1) * (derivative_oreder + 1)))

    u_x = np.zeros((n, m))

    for d in range(derivative_oreder + 1):
        if d > 0:
            for i in range(m):
                u_x[:, i] = centered_finite_difference(u[:, i], delta=dx, order=d)
        else:
            u_x = np.ones((n, m))

        for p in range(polinomial_order + 1):
            u_x_integrated = tintegrate2d(np.multiply(u_x, np.power(u, p)), axes, intervals)

            Theta_integrated[:, d * (polinomial_order + 1) + p] = np.reshape(u_x_integrated, (ni * mi), order='F')

            print(f'derivative: {d}, polimomial: {p}')

    """
    Find Xi
    """
    threshold = 0.1

    alpha = 1e-5

    Xi = stls(Theta_integrated, u_t_integrated, alpha, threshold, iterations=100)

    print(Xi)

    terms = np.array(
        ['1', 'u', 'u^2', 'u_x', 'uu_x', 'u^2u_x', 'u_xx', 'uu_xx', 'u^2u_xx', 'u_xxx', 'uu_xxx', 'u^2u_xxx', 'u_xxxx',
         'uu_xxxx', 'u^2u_xxxx'])
    
    print_equation(Xi, terms)

    # ---------------------------------------------------------------------------------------------------------------- #

    """
    Plot Data
    """

    data = sio.loadmat('../data/kuramoto_sivishinky.mat')
    t = np.ravel(data['tt'])
    x = np.ravel(data['x'])
    u = data['uu']
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
    plt.savefig('../images/ks.png', dpi=300)
    plt.show()

    u = u.reshape(len(x), len(t), 1)
    u_dot = u_dot.reshape(len(x), len(t), 1)

    train = range(0, int(len(t) * 0.6))
    test = [i for i in np.arange(len(t)) if i not in train]
    u_train = u[:, train, :]
    u_test = u[:, test, :]
    u_dot_train = u_dot[:, train, :]
    u_dot_test = u_dot[:, test, 0]
    t_train = t[train]
    t_test = t[test]

    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=x,
        include_bias=True,
        is_uniform=True,
        periodic=True
    )

    print('STLSQ model: ')
    optimizer = ps.STLSQ(threshold=10, alpha=1e-5, normalize_columns=True)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    model.print()
    u_dot_stlsq = model.predict(u_test)

    print('SR3 model, L0 norm: ')
    optimizer = ps.SR3(
        threshold=7,
        max_iter=10000,
        tol=1e-15,
        nu=1e2,
        thresholder="l0",
        normalize_columns=True,
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    model.print()

    print('SR3 model, L1 norm: ')
    optimizer = ps.SR3(
        threshold=1, max_iter=10000, tol=1e-15, thresholder="l1", normalize_columns=True
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    model.print()

    print('SSR model: ')
    optimizer = ps.SSR(normalize_columns=True, kappa=1e1)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    model.print()

    print('SSR (metric = model residual) model: ')
    optimizer = ps.SSR(criteria="model_residual", normalize_columns=True, kappa=1e1)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    model.print()

    print('FROLs model: ')
    optimizer = ps.FROLS(normalize_columns=True, kappa=1e-4)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    model.print()

    # Make fancy plot comparing derivative
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.pcolormesh(t_test, x, u_dot_test,
                   cmap='seismic', vmin=-1.5, vmax=1.5)
    plt.colorbar()
    plt.xlabel('t', fontsize=20)
    plt.ylabel('x', fontsize=20)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(1, 3, 2)
    u_dot_stlsq = np.reshape(u_dot_stlsq, (len(x), len(t_test)))
    plt.pcolormesh(t_test, x, u_dot_stlsq,
                   cmap='seismic', vmin=-1.5, vmax=1.5)
    plt.colorbar()
    plt.xlabel('t', fontsize=20)
    plt.ylabel('x', fontsize=20)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(1, 3, 3)
    plt.pcolormesh(t_test, x, u_dot_stlsq - u_dot_test,
                   cmap='seismic', vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.xlabel('t', fontsize=20)
    plt.ylabel('x', fontsize=20)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()