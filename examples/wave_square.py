import imageio
import os
import yaml

from pathlib import Path
from scipy.interpolate import griddata, Rbf
from tqdm import tqdm

from easydict import EasyDict as edict
import numpy as np
import matplotlib.pyplot as plt

from library.derivative import centered_finite_difference
from library.io import read_mesh, read_data, read_data_flat_sample
from library.library import print_equation
from library.regression import stls
from library.trapezoidal_rule import tintegrate3d


def build_rhs_integrated(u, axes, intervals):
    x, y, t = axes

    n, m, l = u.shape[0], u.shape[1], u.shape[2]

    dt = t[1] - t[0]

    u_tt = np.zeros((n, m, l))

    for i in range(n):
        for j in range(m):
            u_tt[i, j, :] = centered_finite_difference(u[i, j, :], delta=dt, order=2)

    u_tt_integrated = tintegrate3d(u_tt, axes, intervals)

    ni, mi, li = u_tt_integrated.shape

    u_tt_integrated = np.reshape(u_tt_integrated, (ni * mi * li), order='F')

    return u_tt_integrated


def build_theta_integrated(u, polynomial_order, derivative_order, axes, intervals, verbose=False):
    x, y, t = axes

    n, m, l = u.shape[0], u.shape[1], u.shape[2]

    dt = t[1] - t[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    ni, mi, li = np.array(u.shape) // intervals

    columns = (polynomial_order + 1) * (derivative_order + 1) + (polynomial_order + 1) * derivative_order

    if verbose: print(f'Number of coulumns: {columns}')

    index = 0

    Theta_integrated = np.zeros((ni * mi * li, columns))

    derivatives = np.zeros((n, m, l))
    for d in range(derivative_order + 1):
        if d > 0:
            for i in range(n):
                for k in range(l):
                    derivatives[i, :, k] = centered_finite_difference(u[i, :, k], delta=dx, order=d)
        else:
            derivatives = np.ones((n, m, l))

        for p in range(polynomial_order + 1):
            if verbose: print(f'{index}: `x`: derivative: {d}, polimomial: {p}')

            product = np.multiply(derivatives, np.power(u, p))
            product_integrated = tintegrate3d(product, axes, intervals)

            Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

            index = index + 1

    derivatives = np.zeros((n, m, l))
    for d in range(1, derivative_order + 1):
        if d > 0:
            for j in range(m):
                for k in range(l):
                    derivatives[:, j, k] = centered_finite_difference(u[:, j, k], delta=dy, order=d)
        else:
            derivatives = np.ones((n, m, l))

        for p in range(polynomial_order + 1):
            if verbose: print(f'{index}: `x`: derivative: {d}, polimomial: {p}')

            product = np.multiply(derivatives, np.power(u, p))
            product_integrated = tintegrate3d(product, axes, intervals)

            Theta_integrated[:, index] = np.reshape(product_integrated, (ni * mi * li), order='F')

            index = index + 1

    return Theta_integrated


def interpolate(geometry, u_sampled, X, Y, method):
    u_grid = griddata(geometry, u_sampled, (X, Y), method=method)

    return u_grid


def interpolate_rbf(u_sampled, geometry, X, Y, dimensions, function):
    """
    Interpolate data
    """
    NX, NY, NT = dimensions

    x_sample, y_sample = geometry[:, 0], geometry[:, 1]

    u_grid = np.zeros((NX, NY, NT))

    for i in tqdm(range(NT)):
        rbf_interpolator = Rbf(x_sample, y_sample, u_sampled[:, i], function=function)
        u_grid[:, :, i] = rbf_interpolator(X, Y)

    return u_grid


def solve_system(u_grid, axes, intervals, polynomial_order=2, derivative_order=2, threshold=0.45, alpha=1e-5):
    """
    Compute u_t
    """
    u_tt_integrated = build_rhs_integrated(u_grid, axes, intervals)

    """
    Crete Theta matrix
    """

    Theta_integrated = build_theta_integrated(u_grid, polynomial_order, derivative_order, axes, intervals)
    # norms = np.linalg.norm(Theta_integrated, axis=0)
    # Theta_integrated = Theta_integrated / norms

    """
    Find Xi
    """

    Xi = stls(Theta_integrated, u_tt_integrated, alpha, threshold, iterations=100)
    # Xi = Xi * norms

    return Xi


def create_interpolation_plot(u_sampled, geometry, time, x, y, X, Y, dimensions, function, filename):
    x_sample, y_sample = geometry[:, 0], geometry[:, 1]

    rbf_interpolator = Rbf(x_sample, y_sample, u_sampled, function=function)
    u_grid = rbf_interpolator(X, Y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_sample, y_sample, u_sampled, color='darkred', s=10, label='Sample Data')
    ax.plot_surface(X, Y, u_grid, cmap='viridis', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_zlim([-4, 4])

    ax.legend()
    ax.set_title(f'Radial Basis Function Interpolation\nTime: {time: 03f}')

    plt.savefig(filename)
    plt.close(fig)


if __name__ == '__main__':
    """
    Wave equation:
    u_tt = u_xx + u_yy

    u(x, y, t)
    """

    root = "../data/rectangular/"

    """
    Parameters
    """

    path = Path(os.path.join(root, 'config.yaml'))
    with path.open('r') as f:
        configuration = edict(yaml.safe_load(f))

    NT = configuration.NT
    NX = configuration.NX
    NY = configuration.NY

    T_range = configuration.T
    X_range = configuration.X
    Y_range = configuration.Y

    dt = (T_range[1] - T_range[0]) / NT
    dx = (X_range[1] - X_range[0]) / NX
    dy = (Y_range[1] - Y_range[0]) / NY

    x = np.linspace(X_range[0], X_range[1], NX + 1)
    y = np.linspace(Y_range[0], Y_range[1], NY + 1)
    t = np.linspace(T_range[0], T_range[1], NT + 1)

    terms = np.array(['1', 'u', 'u^2', 'u_x', 'uu_x', 'u^2u_x', 'u_xx', 'uu_xx', 'u^2u_xx',
                      'u_y', 'uu_y', 'u^2u_y', 'u_yy', 'uu_yy', 'u^2u_yy'])

    """
    Read Data u: (x, y, t)
    """

    X, Y = read_mesh(os.path.join(root, 'u.h5'), NX + 1, NY + 1)
    u = read_data(os.path.join(root, 'u.h5'), NX + 1, NY + 1, NT + 1)

    """
    Intervals & Axes
    """
    intervals = [16, 16, 10]
    axes = [x, y, t]

    # ---------------------------------------------------------------------------------------------------------------- #

    """
    Solve System
    """
    Xi = solve_system(u, axes, intervals, polynomial_order=2, derivative_order=2)

    print(f"All points: {(NX + 1) * (NY + 1)}")
    print(Xi)
    print_equation(Xi, terms)
    print()

    # # ---------------------------------------------------------------------------------------------------------------- #
    # # ---------------------------------------------------------------------------------------------------------------- #
    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # Sample u
    #
    # u: (x, y, t)
    # """
    # percentual = 75
    # SAMPLE = (NX + 1) * (NY + 1) * percentual // 100
    # u_sampled, geometry, topology = read_data_flat_sample(root, 'u.h5', (NX + 1) * (NY + 1), NT + 1, SAMPLE)
    #
    # """
    # Interpolate u
    # """
    # methods = ['nearest', 'linear', 'cubic']
    #
    # for method in methods:
    #     u_grid = interpolate(geometry, u_sampled, X, Y, method)
    #     Xi = solve_system(u_grid, axes, intervals, polynomial_order=2, derivative_order=2)
    #
    #     print(f"Interpolation method: {method}. Sampled points: {SAMPLE}")
    #     print(Xi)
    #     print_equation(Xi, terms)
    #     print()
    #
    # # ---------------------------------------------------------------------------------------------------------------- #
    # # ---------------------------------------------------------------------------------------------------------------- #
    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # Sample u
    #
    # u: (x, y, t)
    # """
    # percentual = 75
    # SAMPLE = (NX + 1) * (NY + 1) * percentual // 100
    # u_sampled, geometry, topology = read_data_flat_sample(root, 'u.h5', (NX + 1) * (NY + 1), NT + 1, SAMPLE)
    #
    # dimensions = (NX + 1, NY + 1, NT + 1)
    #
    # """
    # Interpolate u
    # """
    # functions = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
    #
    # for function in functions:
    #     u_grid = interpolate_rbf(u_sampled, geometry, X, Y, dimensions, function)
    #     Xi = solve_system(u_grid, axes, intervals, polynomial_order=2, derivative_order=2)
    #
    #     print(f"Interpolation method: {function}. Sampled points: {SAMPLE}")
    #     print(Xi)
    #     print_equation(Xi, terms)
    #     print()
    #
    # # ---------------------------------------------------------------------------------------------------------------- #
    # # ---------------------------------------------------------------------------------------------------------------- #
    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # Plot Interpolated Data
    # """
    #
    # # Directory to save images
    # image_dir = '../images/rbf_images_square'
    # os.makedirs(image_dir, exist_ok=True)
    #
    # # Generate and save images for each time step
    # filenames = []
    # for i in tqdm(range(len(t))):
    #     filename = os.path.join(image_dir, f'rbf_{t[i]:03f}.png')
    #     create_interpolation_plot(u_sampled[:, i], geometry, t[i], x, y, X, Y, dimensions, 'multiquadric', filename)
    #     filenames.append(filename)
    #
    # # Create a GIF from the saved images
    # with imageio.get_writer(os.path.join('../images', 'rbf_interpolation_square_shift.gif'), mode='I',
    #                         duration=0.1) as writer:
    #     for filename in tqdm(filenames):
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    #
    # # Cleanup images
    # for filename in filenames:
    #     os.remove(filename)
    #
    # print("GIF created: rbf_interpolation_square.gif")
