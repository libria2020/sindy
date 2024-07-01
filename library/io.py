import h5py
import numpy as np
import os

"""
Read Data in regular grid (array-like)
"""


def read_data(path, n, m, l):
    # u(x, y, t)
    u = np.zeros((n, m, l))

    with h5py.File(path, 'r') as hdf5_file:
        mesh = hdf5_file['Mesh']['mesh']['geometry'][()]

        index = np.lexsort((mesh[:, 0], mesh[:, 1]))

        key = list(hdf5_file['Function'].keys())[0]
        dataset = hdf5_file['Function'][key]
        for t, group in enumerate(dataset):
            function = dataset[group][()]
            function = function[index].reshape((n, m))
            u[:, :, t] = function

    return u


def read_mesh(path, n, m):
    with h5py.File(path, 'r') as hdf5_file:
        mesh = hdf5_file['Mesh']['mesh']['geometry'][()]

        index = np.lexsort((mesh[:, 0], mesh[:, 1]))
        meshgrid = mesh[index]
        X = meshgrid[:, 0].reshape((n, m))
        Y = meshgrid[:, 1].reshape((n, m))

    return X, Y


"""
Read Data in regular grid (flat) and randomly sample a number of points
"""


def read_data_flat_sample(root, file, POINTS, NT, SAMPLE, seed=58796541):
    u = np.zeros((POINTS, NT))

    np.random.seed(seed)
    values = sorted(np.random.choice(np.arange(POINTS), SAMPLE, replace=False))

    with h5py.File(os.path.join(root, file), 'r') as hdf5_file:
        geometry = hdf5_file['Mesh']['mesh']['geometry'][()]
        topology = hdf5_file['Mesh']['mesh']['topology'][()]

        key = list(hdf5_file['Function'].keys())[0]
        dataset = hdf5_file['Function'][key]
        for t, group in enumerate(dataset):
            u[:, t] = dataset[group][()].squeeze()

    return u[values], geometry[values], topology[values]


"""
Read Data in a non regular grid (flat)
"""


def read_mesh_flat(root):
    with h5py.File(os.path.join(root, 'u.h5'), 'r') as hdf5_file:
        geometry = hdf5_file['Mesh']['mesh']['geometry'][()]
        topology = hdf5_file['Mesh']['mesh']['topology'][()]

    return geometry, topology


def read_data_flat(root, file, POINTS, NT):
    u = np.zeros((POINTS, NT + 1))

    with h5py.File(os.path.join(root, file), 'r') as hdf5_file:
        key = list(hdf5_file['Function'].keys())[0]
        dataset = hdf5_file['Function'][key]
        for t, group in enumerate(dataset):
            u[:, t] = dataset[group][()].squeeze()

    return u
