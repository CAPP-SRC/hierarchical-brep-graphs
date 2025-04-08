import numpy as np

from collections import defaultdict
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.TopologyUtils import TopologyExplorer


class WorkFacet:
    """Stores information about each facet in mesh."""
    def __init__(self, facet_tag, face_tag, node_tags):
        self.facet_tag = facet_tag
        self.face_tag = face_tag
        self.node_tags = node_tags
        self.node_coords = []
        self.normal = None
        self.d_co = None
        self.centroid = None
        self.occ_face = None
        self.occ_hash_face = None

    def get_normal(self):
        vec1 = self.node_coords[1] - self.node_coords[0]
        vec2 = self.node_coords[2] - self.node_coords[1]
        norm = np.cross(vec1, vec2)
        self.normal = norm / np.linalg.norm(norm)

    def get_d_coefficient(self):
        self.d_co = -(self.normal[0] * self.node_coords[0][0] + self.normal[1] * self.node_coords[0][1]
                      + self.normal[2] * self.node_coords[0][2])

    def get_centroid(self):
        x = (self.node_coords[0][0] + self.node_coords[1][0] + self.node_coords[1][0]) / 3
        y = (self.node_coords[0][1] + self.node_coords[1][1] + self.node_coords[1][1]) / 3
        z = (self.node_coords[0][2] + self.node_coords[1][2] + self.node_coords[1][2]) / 3

        self.centroid = [x, y, z]


def triangulation_from_face(face, face_tag, work_facets, work_nodes, facet_face_link):
    """Triangulate a B-Rep face and get information on its facets."""
    aLoc = TopLoc_Location()
    aTriangulation = BRep_Tool().Triangulation(face, aLoc)
    aTrsf = aLoc.Transformation()

    aNodes = aTriangulation.Nodes()
    aTriangles = aTriangulation.Triangles()

    node_link = {}

    for i in range(1, aTriangulation.NbNodes() + 1):
        node = aNodes.Value(i)
        node.Transform(aTrsf)
        node_tag = len(work_nodes)
        work_nodes[node_tag] = np.array([node.X(), node.Y(), node.Z()])
        node_link[i] = node_tag

    for i in range(1, aTriangulation.NbTriangles() + 1):
        node_1, node_2, node_3 = aTriangles.Value(i).Get()
        node_tags = [node_link[node_1], node_link[node_2], node_link[node_3]]
        node_tags.sort()

        wf = WorkFacet(len(work_facets), face_tag, node_tags)
        facet_face_link[wf.facet_tag] = face_tag

        for node in wf.node_tags:
            wf.node_coords.append(work_nodes[node])

        wf.get_normal()
        wf.get_d_coefficient()
        wf.get_centroid()
        work_facets[wf.facet_tag] = wf

    return work_facets, work_nodes, facet_face_link


def group_nodes(work_nodes):
    new_node_link = {}
    node_groups = defaultdict(list)
    for key, val in sorted(work_nodes.items()):
        node_groups[tuple(val)].append(key)

    for nodes in node_groups.values():
        new_node_link[nodes[0]] = nodes[0]

        for i in range(1, len(nodes)):
            new_node_link[nodes[i]] = nodes[0]

    return new_node_link


def replace_nodes_of_facets(work_facets, node_link):
    for facet in work_facets.values():
        for i in range(len(facet.node_tags)):
            facet.node_tags[i] = node_link[facet.node_tags[i]]

    return work_facets


def get_edge_dicts(facets):
    edge_dict = {}
    edge_facet_dict = {}

    for facet in facets.values():
        edge_1 = tuple(sorted((facet.node_tags[0], facet.node_tags[1])))
        edge_2 = tuple(sorted((facet.node_tags[0], facet.node_tags[2])))
        edge_3 = tuple(sorted((facet.node_tags[1], facet.node_tags[2])))

        edge_1_tag = len(edge_dict)
        edge_2_tag = edge_1_tag + 1
        edge_3_tag = edge_2_tag + 1

        edge_dict[edge_1_tag] = edge_1
        edge_dict[edge_2_tag] = edge_2
        edge_dict[edge_3_tag] = edge_3

        edge_facet_dict[edge_1_tag] = facet.facet_tag
        edge_facet_dict[edge_2_tag] = facet.facet_tag
        edge_facet_dict[edge_3_tag] = facet.facet_tag

    return edge_dict, edge_facet_dict


def sort_edges_to_facets(edge_dict, edges_to_facets_dict):
    new_edge_to_facets = {}

    edge_groups = defaultdict(list)
    for key, val in sorted(edge_dict.items()):
        edge_groups[val].append(key)

    for group in edge_groups.values():
        new_edge_to_facets[group[0]] = [edges_to_facets_dict[group[0]]]

        for i in range(1, len(group)):
            new_edge_to_facets[group[0]].append(edges_to_facets_dict[group[i]])

    return new_edge_to_facets


def get_face_facet_links(facets, faces):
    projection = np.zeros((len(faces), len(facets)))

    facet_indices = sorted(list(facets.keys()))

    for key, facet in facets.items():
        a = facet.face_tag
        b = facet_indices.index(key)
        projection[a, b] = 1

    embedding = np.transpose(projection)

    return embedding, projection


def get_mesh_information(shape):
    face_dict = {}
    facets_to_faces = {}
    facets = {}
    nodes = {}
    face_tag = 0

    topo = TopologyExplorer(shape)
    faces = topo.faces()

    for face in faces:
        face_dict[face_tag] = face
        facets, nodes, facets_to_faces = triangulation_from_face(face, face_tag, facets, nodes, facets_to_faces)
        face_tag += 1

    node_link = group_nodes(nodes)
    facets = replace_nodes_of_facets(facets, node_link)
    edge_dict, edge_facet_dict = get_edge_dicts(facets)
    edge_to_facets = sort_edges_to_facets(edge_dict, edge_facet_dict)

    return facets, edge_to_facets, facets_to_faces, nodes
