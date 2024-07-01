import scipy.io as sio
import matplotlib.pyplot as plt
import pysindy as ps
import numpy as np

from library.derivative import centered_finite_difference
from library.library import print_equation
from library.regression import stls
from library.trapezoidal_rule import tintegrate3d

if __name__ == '__main__':
    """
    Reaction Diffusion equation:
    b = 1
    
    u_t = 0.1 * u_xx + 0.1 * u_yy + u - u^2 * u - v^2 * u + b * u^2 * v + b * v^2 * v
    
    v_t = 0.1 * v_xx + 0.1 * v_yy + v - u^2 * v - v^2 * v - b * u^2 * u - b * v^2 * u
    """

    """
    Read Data
    """
    data = sio.loadmat('../data/reaction_diffusion.mat')

    t = np.ravel(data["t"])
    x = np.ravel(data["x"])
    y = np.ravel(data["y"])
    u = np.real(data["u_sol"])
    v = np.real(data["v_sol"])
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    """
    Intervals & Axes
    """
    intervals = [16, 16, 10]
    axes = [x, y, t]

    """
    Compute u_t
    """

    n, m, l = u.shape

    u_t = np.zeros((n, m, l))
    v_t = np.zeros((n, m, l))

    for i in range(n):
        for j in range(m):
            u_t[i, j, :] = centered_finite_difference(u[i, j, :], delta=dt, order=1)
            v_t[i, j, :] = centered_finite_difference(v[i, j, :], delta=dt, order=1)

    u_t_integrated = tintegrate3d(u_t, axes, intervals)
    v_t_integrated = tintegrate3d(v_t, axes, intervals)

    ni, mi, li = u_t_integrated.shape

    u_t_integrated = np.reshape(u_t_integrated, (ni * mi * li), order='F')
    v_t_integrated = np.reshape(v_t_integrated, (ni * mi * li), order='F')

    """
    Crete Theta matrix
    """
    polynomial_order = 3
    derivative_order = 3

    columns = 2 * (polynomial_order + 1) * (derivative_order + 1) + 2 * (polynomial_order) * (
            derivative_order + 1) + 2 * polynomial_order - 1

    Theta_integrated = np.zeros((ni * mi * li, columns))

    index = 0

    # 1 u u^2 u^3
    # u_x uu_x u^2u_x u^3u_x
    # u_xx uu_xx u^2u_xx u^3u_xx
    # u_xxx uu_xxx u^2u_xxx u^3u_xxx
    derivatives = np.zeros((n, m, l))
    for d in range(derivative_order + 1):
        if d > 0:
            for i in range(n):
                for k in range(l):
                    derivatives[i, :, k] = centered_finite_difference(u[i, :, k], delta=dx, order=d)
        else:
            derivatives = np.ones((n, m, l))

        for p in range(polynomial_order + 1):
            product = np.multiply(derivatives, np.power(u, p))
            product_integrated = tintegrate3d(product, axes, intervals)

            Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

            index = index + 1

    # u_y uu_y u^2u_y u^3u_y
    # u_yy uu_yy u^2u_yy u^3u_yy
    # u_yyy uu_yyy u^2u_yyy u^3u_yyy
    derivatives = np.zeros((n, m, l))
    for d in range(1, derivative_order + 1):
        for j in range(m):
            for k in range(l):
                derivatives[:, j, k] = centered_finite_difference(u[:, j, k], delta=dy, order=d)

        for p in range(polynomial_order + 1):
            product = np.multiply(derivatives, np.power(u, p))
            product_integrated = tintegrate3d(product, axes, intervals)

            Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

            index = index + 1

    # ---------------------------------------------------------------------------------------------------------------- #

    # 1 v v^2 v^3
    # v_x vv_x v^2v_x v^3v_x
    # v_xx vv_xx v^2v_xx v^3v_xx
    # v_xxx v_xxx v^2v_xxx v^3v_xxx
    derivatives = np.zeros((n, m, l))
    for d in range(derivative_order + 1):
        if d > 0:
            for i in range(n):
                for k in range(l):
                    derivatives[i, :, k] = centered_finite_difference(v[i, :, k], delta=dx, order=d)
        else:
            derivatives = np.ones((n, m, l))

        for p in range(polynomial_order + 1):
            product = np.multiply(derivatives, np.power(v, p))
            product_integrated = tintegrate3d(product, axes, intervals)

            Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

            index = index + 1

    # v_y vv_y v^2v_y v^3v_y
    # v_yy vv_yy v^2v_yy v^3v_yy
    # v_yyy vv_yyy v^2v_yyy v^3v_yyy
    derivatives = np.zeros((n, m, l))
    for d in range(1, derivative_order + 1):
        for j in range(m):
            for k in range(l):
                derivatives[:, j, k] = centered_finite_difference(v[:, j, k], delta=dy, order=d)

        for p in range(polynomial_order + 1):
            product = np.multiply(derivatives, np.power(v, p))
            product_integrated = tintegrate3d(product, axes, intervals)

            Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

            index = index + 1

    # ---------------------------------------------------------------------------------------------------------------- #

    # uv uv^2 uv^3
    for p in range(1, polynomial_order + 1):
        product = np.multiply(u, np.power(v, p))
        product_integrated = tintegrate3d(product, axes, intervals)

        Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

        index = index + 1

    # ---------------------------------------------------------------------------------------------------------------- #

    # vu^2 vu^3
    for p in range(2, polynomial_order + 1):
        product = np.multiply(v, np.power(u, p))
        product_integrated = tintegrate3d(product, axes, intervals)

        Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

        index = index + 1

    # ---------------------------------------------------------------------------------------------------------------- #

    """
    Find Xi
    """
    threshold = 0.05

    alpha = 1e-5

    Xi = stls(Theta_integrated, u_t_integrated, alpha, threshold, iterations=100)
    # u_t = 0.1 * u_xx + 0.1 * u_yy + u - u^2 * u - v^2 * u + b * u^2 * v + b * v^2 * v

    print(Xi)

    Xi = stls(Theta_integrated, v_t_integrated, alpha, threshold, iterations=100)
    # v_t = 0.1 * v_xx + 0.1 * v_yy + v - u^2 * v - v^2 * v - b * u^2 * u - b * v^2 * u

    print(Xi)

    # ---------------------------------------------------------------------------------------------------------------- #

    u_sol = u
    v_sol = v

    # Compute u_t from generated solution
    u = np.zeros((n, n, len(t), 2))
    u[:, :, :, 0] = u_sol
    u[:, :, :, 1] = v_sol
    u_dot = ps.FiniteDifference(axis=2)._differentiate(u, dt)

    X, Y = np.meshgrid(x, y)

    # Choose 60 % of data for training because data is big...
    # can only randomly subsample if you are passing u_dot to model.fit!!!
    train = np.random.choice(len(t), int(len(t) * 0.6), replace=False)
    test = [i for i in np.arange(len(t)) if i not in train]
    u_train = u[:, :, train, :]
    u_test = u[:, :, test, :]
    u_dot_train = u_dot[:, :, train, :]
    u_dot_test = u_dot[:, :, test, :]
    t_train = t[train]
    t_test = t[test]
    spatial_grid = np.asarray([X, Y]).T

    # Odd polynomial terms in (u, v), up to second order derivatives in (u, v)
    library_functions = [
        lambda x: x,
        lambda x: x * x * x,
        lambda x, y: x * y * y,
        lambda x, y: x * x * y,
    ]
    library_function_names = [
        lambda x: x,
        lambda x: x + x + x,
        lambda x, y: x + y + y,
        lambda x, y: x + x + y,
    ]
    pde_lib = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=spatial_grid,
        include_bias=True,
        is_uniform=True,
        periodic=True
    )
    print('STLSQ model: ')
    optimizer = ps.STLSQ(threshold=50, alpha=1e-5,
                         normalize_columns=True, max_iter=200)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, x_dot=u_dot_train)
    model.print()
    u_dot_stlsq = model.predict(u_test)

    print('SR3 model, L0 norm: ')
    optimizer = ps.SR3(
        threshold=60,
        max_iter=1000,
        tol=1e-10,
        nu=1,
        thresholder="l0",
        normalize_columns=True,
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, x_dot=u_dot_train)
    model.print()
    u_dot_sr3 = model.predict(u_test)

    print('SR3 model, L1 norm: ')
    optimizer = ps.SR3(
        threshold=40,
        max_iter=1000,
        tol=1e-10,
        nu=1e2,
        thresholder="l1",
        normalize_columns=True,
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, x_dot=u_dot_train)
    model.print()

    print('Constrained SR3 model, L0 norm: ')
    feature_names = np.asarray(model.get_feature_names())
    n_features = len(feature_names)
    n_targets = u_train.shape[-1]
    constraint_rhs = np.zeros(2)
    constraint_lhs = np.zeros((2, n_targets * n_features))

    # (u_xx coefficient) - (u_yy coefficient) = 0
    constraint_lhs[0, 11] = 1
    constraint_lhs[0, 15] = -1
    # (v_xx coefficient) - (v_yy coefficient) = 0
    constraint_lhs[1, n_features + 11] = 1
    constraint_lhs[1, n_features + 15] = -1
    optimizer = ps.ConstrainedSR3(
        threshold=.05,
        max_iter=400,
        tol=1e-10,
        nu=1,
        thresholder="l0",
        normalize_columns=False,
        constraint_rhs=constraint_rhs,
        constraint_lhs=constraint_lhs,
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, x_dot=u_dot_train)
    model.print()
    u_dot_constrained_sr3 = model.predict(u_test)


"""
Reaction Diffusion equation:
b = 1

u_t = 0.1 * u_xx + 0.1 * u_yy + u - u^2 * u - v^2 * u + b * u^2 * v + b * v^2 * v

v_t = 0.1 * v_xx + 0.1 * v_yy + v - u^2 * v - v^2 * v - b * u^2 * u - b * v^2 * u

1 u u^2 u^3 u_x uu_x
u^2u_x u^3u_x u_xx uu_xx u^2u_xx u^3u_xx 
u_xxx uu_xxx u^2u_xxx u^3u_xxx u_y uu_y 
u^2u_y u^3u_y u_yy uu_yy u^2u_yy u^3u_yy 
u_yyy uu_yyy u^2u_yyy u^3u_yyy  1 v 
v^2 v^3 v_x vv_x v^2v_x v^3v_x 
v_xx vv_xx v^2v_xx v^3v_xx v_xxx v_xxx 
v^2v_xxx v^3v_xxx v_y vv_y v^2v_y v^3v_y 
v_yy vv_yy v^2v_yy v^3v_yy v_yyy vv_yyy 
v^2v_yyy v^3v_yyy uv uv^2 uv^3 vu^2 
vu^3
    
    
[ 0.          1.02374973  0.         -1.02606264  0.          0. u u^3
  0.          0.          0.10012269  0.          0.          0. uxx 
  0.          0.          0.          0.          0.          0.
  0.          0.          0.0996732   0.          0.          0. uyy
  0.          0.          0.          0.          0.          0.
  0.          0.99997384  0.          0.          0.          0. v^3
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.         -1.02587205  0.          0.99841463 vu^2 uv^2
  0.        ]
[ 0.          0.          0.         -0.99992231  0.          0. u^3
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          1.02301339 v
  0.         -1.02544355  0.          0.          0.          0. v^3
  0.09961426  0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0. v_xx
  0.10002761  0.          0.          0.          0.          0. v_yy
  0.          0.          0.         -0.99863319  0.         -1.02498195 uv^2 vu^2 
  0.        ]
"""