import h5py
import numpy as np
from scipy.sparse import coo_matrix


def get_sparse_tensor(adj_matrix, default_value=0.):
    """Convert a dense adjacency matrix to a sparse tensor representation."""
    idx = np.where(np.not_equal(adj_matrix, default_value))
    values = adj_matrix[idx]
    shape = np.shape(adj_matrix)

    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)

    return idx, values, shape


def get_mat_idxs(adj_matrix, default_value=0.):
    """Get the indices of non-default values in the adjacency matrix."""
    idx = np.where(np.not_equal(adj_matrix, default_value))
    idx = np.transpose(idx).astype(np.int32)

    return idx


def get_sub_graphs(idxs):
    """Get sub-graphs from adjacency matrix indices."""
    def graph_traverse(adj_node):
        # Recursive function to traverse the graph.
        if adj_node not in sub_graph:
            found_nodes.add(adj_node)
            sub_graph.append(adj_node)
            n_adj_nodes = edge_dict[adj_node]
            for n_adj_node in n_adj_nodes:
                graph_traverse(n_adj_node)

    # Initialize edge dictionary and sets for found nodes and sub-graphs.
    edge_dict = {}
    found_nodes = set()
    sub_graphs = []

    for idx in idxs:
        # Populate edge dictionary.
        if idx[0] not in edge_dict:
            edge_dict[idx[0]] = {idx[1].item()}
        else:
            edge_dict[idx[0]].add(idx[1].item())

    for key, values in edge_dict.items():
        # Traverse and find sub-graphs.
        if key in found_nodes:
            continue
        else:
            sub_graph = [key]
            found_nodes.add(key)
            for value in values:
                found_nodes.add(value)
                graph_traverse(value)

        sub_graphs.append(sub_graph)

    return sub_graphs


def reduce_adj_mat(adj_mat, to_del):
    """Reduce the adjacency matrix by deleting specified rows and columns."""
    sub_mat = np.delete(adj_mat, to_del, axis=0)
    sub_mat = np.delete(sub_mat, to_del, axis=1)
    return sub_mat


def reduce_vert_feats(verts, to_del):
    """Reduce the vertex features by deleting specified rows."""
    sub_verts = np.delete(verts, to_del, axis=0)
    return sub_verts


def reduce_inc_mat(inc_mat, to_del):
    """Reduce the incidence matrix by deleting specified columns and rows."""
    sub_inc_mat = np.delete(inc_mat, to_del, axis=1)
    facet_to_del = np.where(~sub_inc_mat.any(axis=1))[0]
    sub_inc_mat = np.delete(sub_inc_mat, facet_to_del, axis=0)

    return sub_inc_mat, facet_to_del


def write_batch_to_file(batch_num, name, V_1, A_1, V_2, A_2, A_3, labels, idxs):
    """Write a batch of data to an HDF5 file."""
    A_1_idx, A_1_values, A_1_shape = get_sparse_tensor(A_1)
    A_2_idx, A_2_values, A_2_shape = get_sparse_tensor(A_2)
    A_3_idx, A_3_values, A_3_shape = get_sparse_tensor(A_3)
    hf = h5py.File(write_file_path, 'a')

    batch_group = hf.create_group(str(batch_num))

    batch_group.create_dataset("name", data=name)
    batch_group.create_dataset("V_1", data=V_1)
    batch_group.create_dataset("A_1_idx", data=A_1_idx)
    batch_group.create_dataset("A_1_values", data=A_1_values)
    batch_group.create_dataset("A_1_shape", data=A_1_shape)
    batch_group.create_dataset("V_2", data=V_2)
    batch_group.create_dataset("A_2_idx", data=A_2_idx)
    batch_group.create_dataset("A_2_values", data=A_2_values)
    batch_group.create_dataset("A_2_shape", data=A_2_shape)
    batch_group.create_dataset("A_3_idx", data=A_3_idx)
    batch_group.create_dataset("A_3_values", data=A_3_values)
    batch_group.create_dataset("A_3_shape", data=A_3_shape)
    batch_group.create_dataset("idxs", data=idxs)
    batch_group.create_dataset("labels", data=labels)

    hf.close()

    batch_num += 1


def partition_graph_by_concave_mat(batch_num, name, face_verts, face_mat_idx, face_mat_values, face_mat_shape,
                                   facet_verts, facet_mat_idx, facet_mat_values, facet_mat_shape, inc_mat_idx,
                                   inc_mat_values, inc_mat_shape, labels):
    """Partition the hierarchical graph to have a subgraph of only concave feature faces."""
    face_mat_row = face_mat_idx[:, 0]
    face_mat_column = face_mat_idx[:, 1]
    face_mat = coo_matrix((face_mat_values, (face_mat_row, face_mat_column)), shape=face_mat_shape).toarray()

    # Each sub graph is a set of concave faces. These will belong to concave features such as holes or pockets.
    sub_graphs = get_sub_graphs(face_mat_idx)

    inc_mat_row = inc_mat_idx[:, 0]
    inc_mat_column = inc_mat_idx[:, 1]
    inc_mat = coo_matrix((inc_mat_values, (inc_mat_row, inc_mat_column)), shape=inc_mat_shape).toarray()

    facet_mat_row = facet_mat_idx[:, 0]
    facet_mat_column = facet_mat_idx[:, 1]
    facet_mat = coo_matrix((facet_mat_values, (facet_mat_row, facet_mat_column)), shape=facet_mat_shape).toarray()

    full_idxs = np.arange(face_mat_shape[0])
    sub_graph_faces = []

    # Loop over sub-graphs and extract features for each sub-graph.
    for sub_graph in sub_graphs:
        idxs = np.array(sub_graph)
        to_del = np.delete(full_idxs, idxs)
        face_mat_partition = reduce_adj_mat(face_mat, to_del)
        face_vert_partition = reduce_vert_feats(face_verts, to_del)
        inc_mat_partition, facet_to_delete = reduce_inc_mat(inc_mat, to_del)
        facet_mat_partition = reduce_adj_mat(facet_mat, facet_to_delete)
        facet_vert_partition = reduce_vert_feats(facet_verts, facet_to_delete)
        labels_partition = reduce_vert_feats(labels, to_del)
        sub_graph_faces.extend(sub_graph)

        write_batch_to_file(batch_num, name, face_vert_partition, face_mat_partition, facet_vert_partition, facet_mat_partition,
                            inc_mat_partition, labels_partition, idxs)
        batch_num += 1

    return list(set(sub_graph_faces)), batch_num


def partition_graph_by_convex_mat(batch_num, name, face_verts, face_mat_idx, face_mat_values, face_mat_shape,
                                  facet_verts, facet_mat_idx, facet_mat_values, facet_mat_shape, inc_mat_idx,
                                  inc_mat_values, inc_mat_shape, labels, concave_faces):
    """Partition the hierarchical graph to get a subgraph of only convex feature faces."""
    face_mat_row = face_mat_idx[:, 0]
    face_mat_column = face_mat_idx[:, 1]
    face_mat = coo_matrix((face_mat_values, (face_mat_row, face_mat_column)), shape=face_mat_shape).toarray()

    inc_mat_row = inc_mat_idx[:, 0]
    inc_mat_column = inc_mat_idx[:, 1]
    inc_mat = coo_matrix((inc_mat_values, (inc_mat_row, inc_mat_column)), shape=inc_mat_shape).toarray()

    facet_mat_row = facet_mat_idx[:, 0]
    facet_mat_column = facet_mat_idx[:, 1]
    facet_mat = coo_matrix((facet_mat_values, (facet_mat_row, facet_mat_column)), shape=facet_mat_shape).toarray()

    # Remove concave faces from the full set of faces to get convex faces.
    full_idxs = np.arange(face_mat_shape[0])
    convex_faces = np.delete(full_idxs, concave_faces)
    face_mat_partition = reduce_adj_mat(face_mat, concave_faces)
    face_vert_partition = reduce_vert_feats(face_verts, concave_faces)
    inc_mat_partition, facet_to_delete = reduce_inc_mat(inc_mat, concave_faces)
    facet_mat_partition = reduce_adj_mat(facet_mat, facet_to_delete)
    facet_vert_partition = reduce_vert_feats(facet_verts, facet_to_delete)
    labels_partition = reduce_vert_feats(labels, concave_faces)

    write_batch_to_file(batch_num, name, face_vert_partition, face_mat_partition, facet_vert_partition,
                        facet_mat_partition, inc_mat_partition, labels_partition, convex_faces)

    batch_num += 1
    return batch_num


def read_h5_file(file_path):
    """Read hierarchical graphs stored in HDF5 file and process its contents."""
    hf = h5py.File(file_path, 'r')
    keys = list(hf.keys())
    batch_num = 0
    count = 0

    # Loop over groups in h5 file
    for key in keys:
        group = hf.get(key)
        V_1 = np.array(group.get("V_1"))
        E_1_idx = np.array(group.get("E_1_idx"))
        E_1_values = np.array(group.get("E_1_values"))
        E_1_shape = np.array(group.get("E_1_shape"))

        E_2_idx = np.array(group.get("E_2_idx"))
        E_2_values = np.array(group.get("E_2_values"))
        E_2_shape = np.array(group.get("E_2_shape"))

        E_3_idx = np.array(group.get("E_3_idx"))
        E_3_values = np.array(group.get("E_3_values"))

        V_2 = np.array(group.get("V_2"))
        A_2_idx = np.array(group.get("A_2_idx"))
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))

        A_3_idx = np.array(group.get("A_3_idx"))
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))

        labels = np.array(group.get("labels"))

        concave_faces, batch_num = partition_graph_by_concave_mat(batch_num, key, V_1, E_2_idx, E_2_values, E_2_shape,
                                                                  V_2, A_2_idx, A_2_values, A_2_shape, A_3_idx,
                                                                  A_3_values, A_3_shape, labels)

        E_not_concave_idx = np.concatenate((E_1_idx, E_3_idx), axis=0)
        E_not_concave_values = np.concatenate((E_1_values, E_3_values))
        batch_num = partition_graph_by_convex_mat(batch_num, key, V_1, E_not_concave_idx, E_not_concave_values,
                                                  E_1_shape, V_2, A_2_idx, A_2_values, A_2_shape, A_3_idx, A_3_values,
                                                  A_3_shape, labels, concave_faces)

        if count % 100 == 0:
            print(count)
        count += 1


if __name__ == "__main__":
    import os

    read_file_path = os.path.join("data", "test.h5")
    write_file_path = os.path.join("data", "test_sub.h5")
    read_h5_file(read_file_path)
