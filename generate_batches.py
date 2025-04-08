"""Script for generating batches of hierarchical B-Rep graphs for inputting into neural network.
Batches stored in separate hdf5 file."""

import h5py
import random
import numpy as np


EPSILON = 1e-6


def get_sparse_tensor(adj_matrix, default_value=0.):
    """Get sparse tensor of dense tensor."""
    idx = np.where(np.not_equal(adj_matrix, default_value))
    values = adj_matrix[idx]
    shape = np.shape(adj_matrix)

    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)

    return idx, values, shape


def normalize_data(data):
    """Normalize data."""
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    data_norm = (data - data_min) / (data_max - data_min + EPSILON)
    return data_norm


def normalize_surface_labels(data, num_surface_types=11):
    """Normalize the surface labels."""
    data_norm = data / (num_surface_types + EPSILON)
    return data_norm


def disjoint_adj_sparse(m1_idx, m1_values, m1_shape, m2_idx, m2_values, m2_shape):
    """Get disjoint adjacency tensor from two sparse tensors."""
    m3_shape = [m1_shape[0] + m2_shape[0], m1_shape[1] + m2_shape[1]]
    m3_values = np.concatenate((m1_values, m2_values))

    m2_idx[:, 0] = m2_idx[:, 0] + m1_shape[0]
    m2_idx[:, 1] = m2_idx[:, 1] + m1_shape[1]
    m3_idx = np.concatenate((m1_idx, m2_idx), axis=0)

    return m3_idx, m3_values, m3_shape


def extract_data_from_h5_group(h5_group, normalize=True):
    """get data from a h5 group in hdf5 file."""
    V_1 = np.array(h5_group.get("V_1"))
    V_2 = np.array(h5_group.get("V_2"))
    surface_labels = V_1[:, -1].reshape(-1, 1)
    V_1 = V_1[:, :-1]
    labels = np.array(h5_group.get("labels"))

    if normalize:
        V_1 = normalize_data(V_1)
        V_2 = normalize_data(V_2)
        surface_labels = normalize_surface_labels(surface_labels)

    V_1 = np.concatenate((V_1, surface_labels), axis=1)

    A_1_idx = np.array(h5_group.get("A_1_idx"))
    A_1_values = np.array(h5_group.get("A_1_values"))
    A_1_shape = np.array(h5_group.get("A_1_shape"))

    E_1_idx = np.array(h5_group.get("E_1_idx"))
    E_1_values = np.array(h5_group.get("E_1_values"))
    E_1_shape = np.array(h5_group.get("E_1_shape"))

    E_2_idx = np.array(h5_group.get("E_2_idx"))
    E_2_values = np.array(h5_group.get("E_2_values"))
    E_2_shape = np.array(h5_group.get("E_2_shape"))

    E_3_idx = np.array(h5_group.get("E_3_idx"))
    E_3_values = np.array(h5_group.get("E_3_values"))
    E_3_shape = np.array(h5_group.get("E_3_shape"))

    A_2_idx = np.array(h5_group.get("A_2_idx"))
    A_2_values = np.array(h5_group.get("A_2_values"))
    A_2_shape = np.array(h5_group.get("A_2_shape"))

    A_3_idx = np.array(h5_group.get("A_3_idx"))
    A_3_values = np.array(h5_group.get("A_3_values"))
    A_3_shape = np.array(h5_group.get("A_3_shape"))

    return V_1, V_2, A_1_idx, A_1_values, A_1_shape, E_1_idx, E_1_values, E_1_shape, E_2_idx, E_2_values, E_2_shape,\
           E_3_idx, E_3_values, E_3_shape, A_2_idx, A_2_values, A_2_shape, A_3_idx, A_3_values, A_3_shape, labels


def generate_h5_batch_file(file_path, batch_path, vertices_per_batch=10000):
    """Generate hdf5 file for batches of subset of dataset.

    :param file_path: Path of hdf5 file storing the hierarchical B-Rep graphs of subset.
    :param batch_path: Path of hdf5 file for batches.
    :param vertices_per_batch: Max number of vertices in batch for all hierarchical B-Rep graphs.
    :return: None
    """
    hf = h5py.File(file_path, 'r')
    batch_counter = 0
    batch_num = 0
    vertex_count = 0

    V_1_batch = V_2_batch = I_1_batch = names = labels_batch = None
    E_1_batch_idx = E_1_batch_values = E_1_batch_shape = None
    E_2_batch_idx = E_2_batch_values = E_2_batch_shape = None
    E_3_batch_idx = E_3_batch_values = E_3_batch_shape = None
    A_1_batch_idx = A_1_batch_values = A_1_batch_shape = None
    A_2_batch_idx = A_2_batch_values = A_2_batch_shape = None
    A_3_batch_idx = A_3_batch_values = A_3_batch_shape = None

    keys = list(hf.keys())
    random.shuffle(keys)

    # Loop over groups in h5 file
    for key in keys:
        group = hf.get(key)
        V_1, V_2, A_1_idx, A_1_values, A_1_shape, E_1_idx, E_1_values, E_1_shape, E_2_idx, E_2_values, E_2_shape, \
        E_3_idx, E_3_values, E_3_shape, A_2_idx, A_2_values, A_2_shape, A_3_idx, A_3_values, A_3_shape, labels\
            = extract_data_from_h5_group(group, normalize=True)

        vertex_count += V_1.shape[0] + V_2.shape[0]

        # If the first graph in a new batch exceeds the vertices per batch limit
        if batch_counter == 0 and vertex_count >= vertices_per_batch:
            print(f"Generated batch num {batch_num}")
            I_1_batch = np.zeros(V_1.shape[0])
            names = np.array([[key]], dtype='S')

            batch = [names, V_1, A_1_idx, A_1_values, A_1_shape, E_1_idx, E_1_values, E_1_shape, E_2_idx, E_2_values,
                     E_2_shape, E_3_idx, E_3_values, E_3_shape, V_2, A_2_idx, A_2_values, A_2_shape, A_3_idx, A_3_values,
                     A_3_shape, I_1_batch, labels]

            write_batch_to_file(batch_num, batch, batch_path)

            # Reset batch
            vertex_count = 0
            batch_counter = 0
            batch_num += 1
            names = I_1_batch = None

        # Check if adding the new graph will make the batch graph too. If so add the batch graph to the file.
        if batch_counter > 0 and vertex_count >= vertices_per_batch:
            print(f"Generated batch num {batch_num}")
            batch = [names, V_1_batch, A_1_batch_idx, A_1_batch_values, A_1_batch_shape, E_1_batch_idx,
                     E_1_batch_values, E_1_batch_shape, E_2_batch_idx, E_2_batch_values, E_2_batch_shape,
                     E_3_batch_idx, E_3_batch_values, E_3_batch_shape, V_2_batch, A_2_batch_idx,
                     A_2_batch_values, A_2_batch_shape, A_3_batch_idx, A_3_batch_values, A_3_batch_shape, I_1_batch,
                     labels_batch]

            write_batch_to_file(batch_num, batch, batch_path)

            # Reset batch
            vertex_count = 0
            batch_counter = 0
            batch_num += 1
            V_1_batch = V_2_batch = I_1_batch = names = labels_batch = None
            E_1_batch_idx = E_1_batch_values = E_1_batch_shape = None
            E_2_batch_idx = E_2_batch_values = E_2_batch_shape = None
            E_3_batch_idx = E_3_batch_values = E_3_batch_shape = None
            A_1_batch_idx = A_1_batch_values = A_1_batch_shape = None
            A_2_batch_idx = A_2_batch_values = A_2_batch_shape = None
            A_3_batch_idx = A_3_batch_values = A_3_batch_shape = None

        # If limit is not reached add new graph to batch.
        else:
            # If there is no graphs in current batch
            if batch_counter == 0:
                V_1_batch = V_1
                V_2_batch = V_2
                I_1_batch = np.zeros(V_1.shape[0])
                names = np.array([[key]], dtype='S')
                labels_batch = labels

                E_1_batch_idx, E_1_batch_values, E_1_batch_shape = E_1_idx, E_1_values, E_1_shape
                E_2_batch_idx, E_2_batch_values, E_2_batch_shape = E_2_idx, E_2_values, E_2_shape
                E_3_batch_idx, E_3_batch_values, E_3_batch_shape = E_3_idx, E_3_values, E_3_shape
                A_1_batch_idx, A_1_batch_values, A_1_batch_shape = A_1_idx, A_1_values, A_1_shape
                A_2_batch_idx, A_2_batch_values, A_2_batch_shape = A_2_idx, A_2_values, A_2_shape
                A_3_batch_idx, A_3_batch_values, A_3_batch_shape = A_3_idx, A_3_values, A_3_shape

                batch_counter += 1

            # If there are graphs in current batch
            else:
                V_1_batch = np.append(V_1_batch, V_1, axis=0)
                V_2_batch = np.append(V_2_batch, V_2, axis=0)
                I_1 = np.full(V_1.shape[0], batch_counter)
                I_1_batch = np.append(I_1_batch, I_1, axis=0)
                names = np.append(names, np.array([[key]], dtype='S'), axis=0)
                labels_batch = np.append(labels_batch, labels, axis=0)

                E_1_batch_idx, E_1_batch_values, E_1_batch_shape = \
                    disjoint_adj_sparse(E_1_batch_idx, E_1_batch_values, E_1_batch_shape, E_1_idx, E_1_values, E_1_shape)
                E_2_batch_idx, E_2_batch_values, E_2_batch_shape = \
                    disjoint_adj_sparse(E_2_batch_idx, E_2_batch_values, E_2_batch_shape, E_2_idx, E_2_values, E_2_shape)
                E_3_batch_idx, E_3_batch_values, E_3_batch_shape = \
                    disjoint_adj_sparse(E_3_batch_idx, E_3_batch_values, E_3_batch_shape, E_3_idx, E_3_values, E_3_shape)
                A_1_batch_idx, A_1_batch_values, A_1_batch_shape = \
                    disjoint_adj_sparse(A_1_batch_idx, A_1_batch_values, A_1_batch_shape, A_1_idx, A_1_values, A_1_shape)
                A_2_batch_idx, A_2_batch_values, A_2_batch_shape = \
                    disjoint_adj_sparse(A_2_batch_idx, A_2_batch_values, A_2_batch_shape, A_2_idx, A_2_values, A_2_shape)
                A_3_batch_idx, A_3_batch_values, A_3_batch_shape = \
                    disjoint_adj_sparse(A_3_batch_idx, A_3_batch_values, A_3_batch_shape, A_3_idx, A_3_values, A_3_shape)

                batch_counter += 1

    batch = [names, V_1_batch, A_1_batch_idx, A_1_batch_values, A_1_batch_shape, E_1_batch_idx,
             E_1_batch_values, E_1_batch_shape, E_2_batch_idx, E_2_batch_values, E_2_batch_shape,
             E_3_batch_idx, E_3_batch_values, E_3_batch_shape, V_2_batch, A_2_batch_idx,
             A_2_batch_values, A_2_batch_shape, A_3_batch_idx, A_3_batch_values, A_3_batch_shape, I_1_batch,
             labels_batch]

    if batch[0] is not None:
        write_batch_to_file(batch_num, batch, batch_path)

    hf.close()


def write_batch_to_file(batch_num, batch, file_path):
    """Writes batch graph to h5 file.

    :param batch_num: Index of batch.
    :param batch: List containing batch graph information.
    :param file_path: File path of h5 file.
    :return: None
    """
    hf = h5py.File(file_path, 'a')

    batch_group = hf.create_group(str(batch_num))

    batch_group.create_dataset("names", data=batch[0], compression="gzip", compression_opts=9)
    batch_group.create_dataset("V_1", data=batch[1])
    batch_group.create_dataset("A_1_idx", data=batch[2])
    batch_group.create_dataset("A_1_values", data=batch[3])
    batch_group.create_dataset("A_1_shape", data=batch[4])
    batch_group.create_dataset("E_1_idx", data=batch[5])
    batch_group.create_dataset("E_1_values", data=batch[6])
    batch_group.create_dataset("E_1_shape", data=batch[7])
    batch_group.create_dataset("E_2_idx", data=batch[8])
    batch_group.create_dataset("E_2_values", data=batch[9])
    batch_group.create_dataset("E_2_shape", data=batch[10])
    batch_group.create_dataset("E_3_idx", data=batch[11])
    batch_group.create_dataset("E_3_values", data=batch[12])
    batch_group.create_dataset("E_3_shape", data=batch[13])
    batch_group.create_dataset("V_2", data=batch[14])
    batch_group.create_dataset("A_2_idx", data=batch[15])
    batch_group.create_dataset("A_2_values", data=batch[16])
    batch_group.create_dataset("A_2_shape", data=batch[17])
    batch_group.create_dataset("A_3_idx", data=batch[18])
    batch_group.create_dataset("A_3_values", data=batch[19])
    batch_group.create_dataset("A_3_shape", data=batch[20])
    batch_group.create_dataset("I_1", data=batch[21])
    batch_group.create_dataset("labels", data=batch[22])

    hf.close()


if __name__ == "__main__":
    split = "test"

    read_file_path = f"./data/test_sub.h5"
    batch_h5_path = f"./data/{split}_sub_batch.h5"
    generate_h5_batch_file(read_file_path, batch_h5_path, vertices_per_batch=10000)
