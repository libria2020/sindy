import numpy as np

def trapezoid_rule_nd(u, axes, h):
    d = len(axes)

    bounds = [(axis[0], axis[-1]) for axis in axes]

    mesh = np.meshgrid(*axes, indexing='ij')
    combinations = np.stack(mesh, axis=-1)
    points = combinations.reshape(-1, len(axes))

    indices = [np.arange(len(axis)) for axis in axes]
    index_mesh = np.meshgrid(*indices, indexing='ij')
    index_combinations = np.stack(index_mesh, axis=-1)
    index_points = index_combinations.reshape(-1, len(axes))

    integral = 0.0

    for index, point in zip(index_points, points):
        weight = 1
        for i in range(d):
            if point[i] == bounds[i][0] or point[i] == bounds[i][1]:
                weight *= 1
            else:
                weight *= 2

        integral += weight * u[tuple(index)]

    volume = np.prod(h) / 2 ** d
    integral *= volume

    return integral

def tintegrate(u, axes, intervals):
    # axes: t, x, y, z
    # intervals: I_t, I_x, I_y, I_z

    dimensions = np.asarray(u.shape)
    number_of_dimensions = u.ndim

    # Initialize the integrated values array
    integrated_values = np.zeros(tuple(int(np.ceil(dimension / interval)) for dimension, interval in zip(dimensions, intervals)))

    # Create slices for each dimension
    slices = [np.arange(0, dimension, interval) for dimension, interval in zip(dimensions, intervals)]

    # Computing delta for each axis
    deltas = [axis[1] - axis[0] for axis in axes]

    # Create meshgrid for each array
    mesh = np.meshgrid(*slices, indexing='ij')
    # Combine the meshgrids into one array
    combinations = np.stack(mesh, axis=-1)
    # Reshape the combinations array to have shape (N, M), where N is the total number of combinations
    # and M is the total number of arrays in the original list
    combinations = combinations.reshape(-1, len(slices))

    # Create indices for each array
    indices = [np.arange(arr.shape[0]) for arr in slices]
    # Create meshgrid for indices
    index_mesh = np.meshgrid(*indices, indexing='ij')
    # Combine the meshgrids into one array
    index_combinations = np.stack(index_mesh, axis=-1)
    # Reshape the index combinations array to have shape (N, M), where N is the total number of combinations
    # and M is the total number of arrays in the original list
    index_combinations = index_combinations.reshape(-1, len(slices))

    for indexes, combination in zip(index_combinations, combinations):
        initial = combination
        final = combination + intervals

        if (final > dimensions).any():
            final[final > dimensions] = dimensions[final > dimensions]

        slice_slices = [slice(start, end) for start, end in zip(initial, final)]
        sliced_axes = [axis[slice_slices[i]] for i, axis in enumerate(axes)]
        sliced_u = u[tuple(slice_slices)]

        integrated_values[tuple(indexes)] = trapezoid_rule_nd(sliced_u, sliced_axes, deltas)

    return integrated_values


# if __name__ == "__main__":
#     u = np.random.randint(10, size=(450, 37))
#
#     t = np.linspace(0, 10, 450)
#     x = np.linspace(0, 10, 37)
#
#
#     axes = [t, x]
#     intervals = [100, 10]
#
#     u_integrated = tintegrate(u, axes, intervals)
#
#     print()


# -------------------------------------------------------------------------------------------------------------------- #

def integrate2D(u, n, m, I_x, I_t, dx, dt):
    # u(x,t)
    u_integrated = np.zeros((n // I_x, m // I_t))

    slices_t = np.arange(0, m, I_t)
    slices_x = np.arange(0, n, I_x)

    for i in range(len(slices_x) - 1):
        for j in range(len(slices_t) - 1):
            u_integrated[i, j] = np.sum(u[slices_x[i]: slices_x[i + 1], slices_t[j]: slices_t[j + 1]]) * dx * dt

    return u_integrated


def integrate3D(u, n, m, l, I_x, I_y, I_t, dx, dy, dt):
    # u(x,y,t)

    u_integrated = np.zeros((n // I_x, m // I_y, l // I_t))

    slices_x = np.arange(0, n, I_x)
    slices_y = np.arange(0, m, I_y)
    slices_t = np.arange(0, l, I_t)

    for i in range(len(slices_x) - 1):
        for j in range(len(slices_y) - 1):
            for k in range(len(slices_t) - 1):
                u_integrated[i, j, k] = np.sum(
                    u[slices_x[i]: slices_x[i + 1], slices_y[j]: slices_y[j + 1], slices_t[k]: slices_t[k + 1]]
                ) * dx * dy * dt

    return u_integrated


def random_domain_integration(domain, x, t, dx, dt, I_x, I_t, n_points=100, seed=123):
    np.random.seed(seed)

    t_point = np.random.choice(t, size=n_points, replace=(len(t) < n_points))
    x_point = np.random.choice(x, size=n_points, replace=(len(x) < n_points))

    integrated_subdomains = []

    for x_value, t_value in zip(x_point, t_point):
        i = np.argmax(x == x_value)
        j = np.argmax(t == t_value)

        subdomain = domain[i: i + I_x, j: j + I_t]

        integrated_subdomains.append(np.sum(subdomain) * dx * dt)

    return integrated_subdomains


def gauss_integral(geometry, topology, u, NT):
    L = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    w = np.array([[1 / 3], [1 / 3], [1 / 3]])

    u_gauss_integral = np.zeros((len(topology), NT + 1))

    for index, element in enumerate(topology):
        N1, N2, N3 = element[0], element[1], element[2]
        u_node = np.array([u[N1], u[N2], u[N3]])

        x1, y1 = geometry[N1][0], geometry[N1][1]
        x2, y2 = geometry[N2][0], geometry[N2][1]
        x3, y3 = geometry[N3][0], geometry[N3][1]

        matrix = np.array([[1, 1, 1], [x1, x2, x3], [y1, y2, y3]])
        area = 0.5 * np.linalg.det(matrix)

        approximate_integral = (w.T @ (L @ u_node)) * area
        u_gauss_integral[index] = approximate_integral.squeeze()

    return u_gauss_integral


def u_flatten(geometry, topology, X, Y, u, NT):
    u_flat = np.zeros((len(topology), NT + 1))

    index = 0
    for point in geometry:
        row = np.where(X == point[0])[1][0]
        col = np.where(Y == point[1])[0][0]
        u_flat[index] = u[row, col]
        index += 1

    return u_flat


def weak_integrate_time(u, w_x, n, m, I_x, I_t, delta_x, delta_t, d_chi, d_tau):
    u_integrated = np.zeros((n // I_x, m // I_t))

    slices_t = np.arange(0, m, I_t)
    slices_x = np.arange(0, n, I_x)

    for i in range(len(slices_x) - 1):
        for j in range(len(slices_t) - 1):
            u_integrated[i, j] = - np.sum(
                u[slices_x[i]: slices_x[i + 1], slices_t[j]: slices_t[j + 1]] * w_x
            ) * delta_x * 0.5 * d_chi * d_tau

    return u_integrated

# TODO: in progress
# def build_linear_system(u, dt, dx, polinomial_order = 2, derivative_oreder = 2, integration=None):
#     """
#     Required:
#         :param u:
#         :param dt:
#         :param dx:
#
#     Optional:
#         :param polinomial_order:
#         :param derivative_oreder:
#         :param integration:
#             None:
#             trapezoidal:
#             gauss:
#             weak:
#
#     :return:
#     """
#
#     """
#     Compute u_t
#     """
#
#     shape = u.shape
#
#     u_t = np.zeros(shape)
#
#     for i in range(n):
#         u_t[i, :] = centered_finite_difference(u[i, :], delta=dt, order=1)
#
#     """
#     Crete Theta matrix
#     """
#
#     dimensions = u.ndim
#
#     if dimensions > 2:
#         for dim in range(dimensions - 1):
#             um = np.moveaxis(u, dim, 0)
#
#             for d in range(1, derivative_oreder + 1):
#                 for j in range(m):
#                     for k in range(l):
#                         u_x[:, i] = centered_finite_difference(u[:, i], delta=dx, order=d)
#
#
#                 for p in range(polinomial_order + 1):
#                     u_x_integrated = integrate2D(np.multiply(u_x, np.power(u, p)), n, m, I_x, I_t, dx, dt)
#
#                     Theta_integrated[:, d * (polinomial_order + 1) + p] = np.reshape(u_x_integrated, (ni * mi),
#                                                                                      order='F')
#
#     return None


def print_equation(Xi, terms):
    idx = Xi != 0
    selected_term = terms[idx]
    coefficients = Xi[idx]

    print('u_t = ', end='')
    for i, (c, t) in enumerate(zip(coefficients, selected_term)):
        if i != len(coefficients) - 1:
            print(f'{c: .3f} * ' + t + ' + ', end='')
        else:
            print(f'{c: .3f} * ' + t)