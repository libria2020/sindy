import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from library.derivative import centered_finite_difference
from library.library import tintegrate
from library.regression import stls

from scipy.interpolate import interp1d


def interpolate_to_time(target_time, source_time, source_data):
    interpolator = interp1d(source_time, source_data, kind='linear', bounds_error=False, fill_value='extrapolate')
    return interpolator(target_time)


def interpolate_to_time_4d(target_time, source_time, source_data):
    # Initialize an empty array to store the interpolated data
    interpolated_data = np.empty((len(target_time),) + source_data.shape[1:], dtype=source_data.dtype)

    # Loop over each pixel (30x30x30) and perform interpolation along the time dimension
    for i in range(source_data.shape[1]):
        for j in range(source_data.shape[2]):
            for k in range(source_data.shape[3]):
                # Extract the 1D data for the current pixel
                source_data_1d = source_data[:, i, j, k]

                # Interpolate along the time dimension for the current pixel
                interpolator = interp1d(source_time, source_data_1d, axis=0, kind='linear', bounds_error=False,
                                        fill_value='extrapolate')
                interpolated_data[:, i, j, k] = interpolator(target_time)

    return interpolated_data


if __name__ == '__main__':
    """
    f_t = - F/(m * alpha) * f_v
        = - Fx//(m * alpha) * f_vx - Fy//(m * alpha) * f_vy - Fz//(m * alpha) * f_vz
                    
    Fx = q * (Ex + vy * Bz − vz * By)
    Fy = q * (Ey + vz * Bx − vx * Bz)
    Fz = q * (Ez + vx * By − vy * Bx)
    
    alpha = (1 + v * U / U^2)
    
    """

    """
    Read Data
    """

    data = sio.loadmat('../data/des-2016-12-11-15-20-15-out.h5.mat')

    # Velocity Distribution Function
    vdf = data['fpicart']
    tvdf = data['time_vdf'].squeeze()

    dtvdf = tvdf[1] - tvdf[0]  # [ns]

    # Magnetic Field [nT]
    B_vector = data['b_field'].squeeze()
    Bx = B_vector[:, 0]
    By = B_vector[:, 1]
    Bz = B_vector[:, 2]
    B = B_vector[:, 3]

    tB = data['time_b'].squeeze()  # [ns]
    dtB = tB[1] - tB[0]

    # Electric Field [mV/m]
    E_vector = data['e_field'].squeeze()

    Ex = E_vector[:, 0]
    Ey = E_vector[:, 1]
    Ez = E_vector[:, 2]

    tE = data['time_e'].squeeze()  # [ns]
    dtE = tE[1] - tE[0]

    # Velocity [km/s]
    vx = data["grid_x"]
    vy = data["grid_y"]
    vz = data["grid_z"]

    # Bulk velocity [km/s]
    bulk_velocity_e = data['e_bulkv']
    tbve = data['time_ebulkv'].squeeze()  # [ns]

    dtbve = tbve[1] - tbve[0]

    bvxe = bulk_velocity_e[:, 0]
    bvye = bulk_velocity_e[:, 1]
    bvze = bulk_velocity_e[:, 2]

    """
    Reshape Arrays
    """
    t = tbve * 10e-9
    Bxr = interpolate_to_time(t, tB * 10e-9, Bx)
    Byr = interpolate_to_time(t, tB * 10e-9, By)
    Bzr = interpolate_to_time(t, tB * 10e-9, Bz)

    Exr = interpolate_to_time(t, tE * 10e-9, Ex)
    Eyr = interpolate_to_time(t, tE * 10e9, Ey)
    Ezr = interpolate_to_time(t, tE * 10e-9, Ez)

    bvxer = interpolate_to_time(t, tbve * 10e-9, bvxe)
    bvyer = interpolate_to_time(t, tbve * 10e-9, bvye)
    bvzer = interpolate_to_time(t, tbve * 10e-9, bvze)

    """
    Intervals & Axes
    """
    intervals = [100, 3, 3, 3]
    vxx = vx[:, 0, 0]
    axes = [tbve, vxx, vxx, vxx]

    """
    Compute u_t
    """
    u = vdf
    # dt = dtbve
    dt = t[1] - t[0]

    _, n, m, l = u.shape

    u_t = np.zeros((_, n, m, l))

    print("f_t...")
    for i in range(n):
        for j in range(m):
            for k in range(l):
                u_t[:, i, j, k] = centered_finite_difference(u[:, i, j, k], delta=dt, order=1)

    u_t = tintegrate(u_t, axes, intervals)

    dim = u_t.shape

    rhs = np.reshape(u_t, (np.prod(dim)), order='F')


    """
    Crete Theta matrix

    f_t = - F/(m * alpha) * f_v
        = - Fx//(m * alpha) * f_vx - Fy//(m * alpha) * f_vy - Fz//(m * alpha) * f_vz
                    
    Fx = q * (Ex + vy * Bz − vz * By)
    Fy = q * (Ey + vz * Bx − vx * Bz)
    Fz = q * (Ez + vx * By − vy * Bx)
    
    alpha = (1 + v * U / U^2)
    
    """

    q = - 1.6e-19
    mass = 9.11e-31

    vxx = vx[np.newaxis, :, :, :]
    vyy = vy[np.newaxis, :, :, :]
    vzz = vz[np.newaxis, :, :, :]

    bvxxe = bvxer[:, np.newaxis, np.newaxis, np.newaxis]
    bvyye = bvyer[:, np.newaxis, np.newaxis, np.newaxis]
    bvzze = bvzer[:, np.newaxis, np.newaxis, np.newaxis]

    dot_product = vxx * bvxxe + vyy * bvyye + vzz * bvzze
    vector_norm_squared = bvxer ** 2 + bvyer ** 2 + bvzer ** 2

    alpha = 1 + dot_product / vector_norm_squared[:, np.newaxis, np.newaxis, np.newaxis]

    # Fx = q * (Ex + vy * Bz - vz * By)
    # Fy = q * (Ey + vz * Bx - vx * Bz)
    # Fz = q * (Ez + vx * By - vy * Bx)

    FEx = q * Exr[:, np.newaxis, np.newaxis, np.newaxis]  # Fx = q * Ex
    FEy = q * Eyr[:, np.newaxis, np.newaxis, np.newaxis]  # Fy = q * Ey
    FEz = q * Ezr[:, np.newaxis, np.newaxis, np.newaxis]  # Fz = q * Ez

    FBx1 = q * vyy * Bzr[:, np.newaxis, np.newaxis, np.newaxis]  # Fx =   q * vy * Bz
    FBx2 = - q * vzz * Byr[:, np.newaxis, np.newaxis, np.newaxis]  # Fx = - q * vz * By

    FBy1 = q * vzz * Bxr[:, np.newaxis, np.newaxis, np.newaxis]  # Fy =   q * vz * Bx
    FBy2 = - q * vxx * Bzr[:, np.newaxis, np.newaxis, np.newaxis]  # Fy = - q * vx * Bz

    FBz1 = q * vxx * Byr[:, np.newaxis, np.newaxis, np.newaxis]  # Fz = - q * vx * By
    FBz2 = - q * vyy * Bxr[:, np.newaxis, np.newaxis, np.newaxis]  # Fz = - q * vy * Bx

    print("f_vx...")
    u_vx = np.zeros((_, n, m, l))
    for i in range(_):
        for j in range(m):
            for k in range(l):
                u_vx[i, :, j, k] = centered_finite_difference(u[i, :, j, k], delta=dt, order=1)

    print("f_vy...")
    u_vy = np.zeros((_, n, m, l))
    for i in range(_):
        for j in range(n):
            for k in range(l):
                u_vy[i, j, :, k] = centered_finite_difference(u[i, j, :, k], delta=dt, order=1)

    print("f_vz...")
    u_vz = np.zeros((_, n, m, l))
    for i in range(_):
        for j in range(n):
            for k in range(m):
                u_vz[i, j, k, :] = centered_finite_difference(u[i, j, k, :], delta=dt, order=1)

    print("Theta...")
    Theta = np.zeros((np.prod(dim), 9))

    print('0...')
    term = (FEx / (mass * alpha)) * u_vx
    term = tintegrate(term, axes, intervals)
    Theta[:, 0] = np.reshape(term, (np.prod(dim)), order='F')

    print('1...')
    term = (FEy / (mass * alpha)) * u_vy
    term = tintegrate(term, axes, intervals)
    Theta[:, 1] = np.reshape(term, (np.prod(dim)), order='F')

    print('2...')
    term = (FEz / (mass * alpha)) * u_vy
    term = tintegrate(term, axes, intervals)
    Theta[:, 2] = np.reshape(term, (np.prod(dim)), order='F')

    # --- --- ---

    print('3...')
    term = (FBx1 / (mass * alpha)) * u_vx
    term = tintegrate(term, axes, intervals)
    Theta[:, 3] = np.reshape(term, (np.prod(dim)), order='F')

    print('4...')
    term = (FBx2 / (mass * alpha)) * u_vx
    term = tintegrate(term, axes, intervals)
    Theta[:, 4] = np.reshape(term, (np.prod(dim)), order='F')

    # ---

    print('5...')
    term = (FBy1 / (mass * alpha)) * u_vy
    term = tintegrate(term, axes, intervals)
    Theta[:, 5] = np.reshape(term, (np.prod(dim)), order='F')

    print('6...')
    term = (FBy2 / (mass * alpha)) * u_vy
    term = tintegrate(term, axes, intervals)
    Theta[:, 6] = np.reshape(term, (np.prod(dim)), order='F')

    # ---

    print('7...')
    term = (FBz1 / (mass * alpha)) * u_vz
    term = tintegrate(term, axes, intervals)
    Theta[:, 7] = np.reshape(term, (np.prod(dim)), order='F')

    print('8...')
    term = (FBz2 / (mass * alpha)) * u_vz
    term = tintegrate(term, axes, intervals)
    Theta[:, 8] = np.reshape(term, (np.prod(dim)), order='F')

    norms = np.linalg.norm(Theta, axis=0)
    Theta = Theta / norms

    """
    Find Xi
    """
    threshold = 1e-7

    alpha = 1e-5

    print("Xi...")
    Xi = stls(Theta, rhs, alpha, threshold, iterations=100)
    Xi = Xi * norms

    print(Xi)
