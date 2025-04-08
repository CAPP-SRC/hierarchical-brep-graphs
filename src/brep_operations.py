import numpy as np

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone, GeomAbs_Sphere, \
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, \
    GeomAbs_OffsetSurface, GeomAbs_OtherSurface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopoDS import topods_Face
from OCC.Core.gp import gp_Vec
from OCC.Core._BRepGProp import brepgprop_SurfaceProperties


class WorkFace:
    def __init__(self, index, face):
        self.index = index
        self.hash = hash(face)
        self.face = face
        self.surface_area = None
        self.centroid = None
        self.face_type = None
        self.label = None


class WorkEdge:
    def __init__(self, index, edge):
        self.index = index
        self.hash = hash(edge)
        self.edge = edge
        self.faces = []
        self.hash_faces = []
        self.face_tags = []
        # Convex = 0, Concave = 1, Other = 2
        self.convexity = None


def get_brep_information(shape, label_map):
    topo = TopologyExplorer(shape)
    work_faces, faces = get_faces(topo, label_map)
    work_edges = get_edges(topo, faces)

    return work_faces, work_edges, faces


def ask_point_uv2(xyz, face):
    """
    This is a general function which gives the uv coordinates from the xyz coordinates.
    The uv value is not normalised.
    """
    gpPnt = gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    surface = BRep_Tool().Surface(face)

    sas = ShapeAnalysis_Surface(surface)
    gpPnt2D = sas.ValueOfUV(gpPnt, 0.01)
    uv = list(gpPnt2D.Coord())

    return uv


def ask_point_normal_face(uv, face):
    """
    Ask the normal vector of a point given the uv coordinate of the point on a face
    """
    face_ds = topods_Face(face)
    surface = BRep_Tool().Surface(face_ds)
    props = GeomLProp_SLProps(surface, uv[0], uv[1], 1, 1e-6)

    gpDir = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        gpDir.Reverse()

    return gpDir.Coord()


def ask_edge_midpnt_tangent(edge):
    """
    Ask the midpoint of an edge and the tangent at the midpoint
    """
    result = BRep_Tool.Curve(edge)  # result[0] is the handle of curve;result[1] is the umin; result[2] is umax
    tmid = (result[1] + result[2]) / 2
    p = gp_Pnt(0, 0, 0)
    v1 = gp_Vec(0, 0, 0)
    result[0].D1(tmid, p, v1)  # handle.GetObject() gives Geom_Curve type, p:gp_Pnt, v1:gp_Vec

    return [p.Coord(), v1.Coord()]


def edge_dihedral(edge, faces):
    """
    Calculate the dihedral angle of an edge
    """
    [midPnt, tangent] = ask_edge_midpnt_tangent(edge)
    uv0 = ask_point_uv2(midPnt, faces[0])
    uv1 = ask_point_uv2(midPnt, faces[1])
    n0 = ask_point_normal_face(uv0, faces[0])
    n1 = ask_point_normal_face(uv1, faces[1])

    if edge.Orientation() == TopAbs_FORWARD:
        cp = np.cross(n0, n1)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    else:
        cp = np.cross(n1, n0)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    return s


def get_edges(topo, occ_faces):
    work_edges = {}

    edges = topo.edges()
    for edge in edges:
        faces = list(topo.faces_from_edge(edge))

        we = WorkEdge(len(work_edges), edge)

        if len(faces) > 1:
            s = edge_dihedral(edge, faces)
        else:
            s = 0

        if s == 1:
            # Convex
            edge_convexity = 0
        elif s == -1:
            # Concave
            edge_convexity = 1
        else:
            # Smooth (s==0) or other
            edge_convexity = 2

        we.convexity = edge_convexity
        we.faces = faces

        for face in faces:
            we.hash_faces.append(hash(face))
            we.face_tags.append(occ_faces.index(face))

        if len(faces) == 1:
            we.hash_faces.append(hash(faces[0]))
            we.face_tags.append(occ_faces.index(faces[0]))

        work_edges[we.hash] = we

    return work_edges


def ask_surface_area(f):
    props = GProp_GProps()

    brepgprop_SurfaceProperties(f, props)
    area = props.Mass()
    return area


def recognise_face_type(face):
    """Get surface type of B-Rep face"""
    #   BRepAdaptor to get the face surface, GetType() to get the type of geometrical surface type
    surf = BRepAdaptor_Surface(face, True)
    surf_type = surf.GetType()
    a = 0
    if surf_type == GeomAbs_Plane:
        a = 1
    elif surf_type == GeomAbs_Cylinder:
        a = 2
    elif surf_type == GeomAbs_Torus:
        a = 3
    elif surf_type == GeomAbs_Sphere:
        a = 4
    elif surf_type == GeomAbs_Cone:
        a = 5
    elif surf_type == GeomAbs_BezierSurface:
        a = 6
    elif surf_type == GeomAbs_BSplineSurface:
        a = 7
    elif surf_type == GeomAbs_SurfaceOfRevolution:
        a = 8
    elif surf_type == GeomAbs_OffsetSurface:
        a = 9
    elif surf_type == GeomAbs_SurfaceOfExtrusion:
        a = 10
    elif surf_type == GeomAbs_OtherSurface:
        a = 11

    return a


def ask_face_centroid(face):
    """Get centroid of B-Rep face."""
    mass_props = GProp_GProps()
    brepgprop.SurfaceProperties(face, mass_props)
    gPt = mass_props.CentreOfMass()

    return gPt.Coord()


def get_faces(topo, label_map):
    work_faces = {}
    faces = list(topo.faces())

    for face in faces:
        wf = WorkFace(len(work_faces), face)
        wf.face_type = recognise_face_type(face)
        wf.surface_area = ask_surface_area(face)
        wf.centroid = ask_face_centroid(face)
        wf.label = label_map[face]

        work_faces[wf.hash] = wf

    return work_faces, faces