import h5py
import numpy as np


def get_sparse_tensor(adj_matrix, default_value=0.):
    idx = np.where(np.not_equal(adj_matrix, default_value))
    values = adj_matrix[idx]
    shape = np.shape(adj_matrix)

    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)

    return idx, values, shape


def get_face_features(faces):
    faces_list = []
    label_list = []

    for face_tag, face in faces.items():
        face_list = [face.surface_area, face.centroid[0], face.centroid[1], face.centroid[2],
                     face.face_type]
        faces_list.append(face_list)
        label_list.append(face.label)

    return np.array(faces_list, dtype=np.float32), np.array(label_list, dtype=np.float32)


def get_facet_features(facets):
    facets_list = []

    for facet_tag, facet in facets.items():
        facet_list = [facet.normal[0], facet.normal[1], facet.normal[2], facet.d_co]
        facets_list.append(facet_list)

    return np.array(facets_list, dtype=np.float32)


def get_face_adj(edges, faces):
    brep_adj = np.zeros((len(faces), len(faces)))
    convex_adj = np.zeros((len(faces), len(faces)))
    concave_adj = np.zeros((len(faces), len(faces)))
    other_adj = np.zeros((len(faces), len(faces)))

    for edge in edges.values():
        a = edge.face_tags[0]
        b = edge.face_tags[1]

        brep_adj[a, b] = 1
        brep_adj[b, a] = 1

        if edge.convexity == 0:
            convex_adj[a, b] = 1
            convex_adj[b, a] = 1
        elif edge.convexity == 1:
            concave_adj[a, b] = 1
            concave_adj[b, a] = 1
        elif edge.convexity == 2:
            other_adj[a, b] = 1
            other_adj[b, a] = 1

    return brep_adj, convex_adj, concave_adj, other_adj


def get_facet_adj(facets, facet_edges):
    facet_adj = np.zeros((len(facets), len(facets)))
    facet_indices = sorted(list(facets.keys()))

    for edge in facet_edges.values():
        try:
            a = facet_indices.index(edge[0])
            b = facet_indices.index(edge[1])
            facet_adj[a, b] = 1
            facet_adj[b, a] = 1
        except:
            continue

    return facet_adj


def get_face_facet_links(facets, faces):
    projection = np.zeros((len(faces), len(facets)))
    facet_indices = sorted(list(facets.keys()))

    for key, facet in facets.items():
        a = facet.face_tag
        b = facet_indices.index(key)
        projection[a, b] = 1

    return np.transpose(projection)


def write_h5_file(h5_path, graph_name, work_faces, work_facets, work_face_edges, work_facet_edges, save_sparse=True):
    hf = h5py.File(h5_path, 'a')
    hier_group = hf.create_group(str(graph_name))

    V_1, labels = get_face_features(work_faces)
    V_2 = get_facet_features(work_facets)
    A_1, E_1, E_2, E_3 = get_face_adj(work_face_edges, work_faces)
    A_2 = get_facet_adj(work_facets, work_facet_edges)
    A_3 = get_face_facet_links(work_facets, work_faces)

    hier_group.create_dataset("V_1", data=V_1, compression="lzf")
    hier_group.create_dataset("V_2", data=V_2, compression="lzf")
    hier_group.create_dataset("labels", data=labels, compression="lzf")

    if not save_sparse:
        hier_group.create_dataset("A_1", data=A_1, compression="lzf")
        hier_group.create_dataset("E_1", data=E_1, compression="lzf")
        hier_group.create_dataset("E_2", data=E_2, compression="lzf")
        hier_group.create_dataset("E_3", data=E_3, compression="lzf")
        hier_group.create_dataset("A_2", data=A_2, compression="lzf")
        hier_group.create_dataset("A_3", data=A_3, compression="lzf")

    else:
        A_1_idx, A_1_values, A_1_shape = get_sparse_tensor(A_1)
        E_1_idx, E_1_values, E_1_shape = get_sparse_tensor(E_1)
        E_2_idx, E_2_values, E_2_shape = get_sparse_tensor(E_2)
        E_3_idx, E_3_values, E_3_shape = get_sparse_tensor(E_3)
        A_2_idx, A_2_values, A_2_shape = get_sparse_tensor(A_2)
        A_3_idx, A_3_values, A_3_shape = get_sparse_tensor(A_3)

        hier_group.create_dataset("A_1_idx", data=A_1_idx, compression="lzf")
        hier_group.create_dataset("A_1_values", data=A_1_values, compression="lzf")
        hier_group.create_dataset("A_1_shape", data=A_1_shape, compression="lzf")
        hier_group.create_dataset("E_1_idx", data=E_1_idx, compression="lzf")
        hier_group.create_dataset("E_1_values", data=E_1_values, compression="lzf")
        hier_group.create_dataset("E_1_shape", data=E_1_shape, compression="lzf")
        hier_group.create_dataset("E_2_idx", data=E_2_idx, compression="lzf")
        hier_group.create_dataset("E_2_values", data=E_2_values, compression="lzf")
        hier_group.create_dataset("E_2_shape", data=E_2_shape, compression="lzf")
        hier_group.create_dataset("E_3_idx", data=E_3_idx, compression="lzf")
        hier_group.create_dataset("E_3_values", data=E_3_values, compression="lzf")
        hier_group.create_dataset("E_3_shape", data=E_3_shape, compression="lzf")
        hier_group.create_dataset("A_2_idx", data=A_2_idx, compression="lzf")
        hier_group.create_dataset("A_2_values", data=A_2_values, compression="lzf")
        hier_group.create_dataset("A_2_shape", data=A_2_shape, compression="lzf")
        hier_group.create_dataset("A_3_idx", data=A_3_idx, compression="lzf")
        hier_group.create_dataset("A_3_values", data=A_3_values, compression="lzf")
        hier_group.create_dataset("A_3_shape", data=A_3_shape, compression="lzf")

    hf.close()
