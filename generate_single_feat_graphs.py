"""Script to generate hierarchical B-Rep graphs of STEP file CAD models and store them in a hdf5 file.
This is run for each subset in the dataset.

Reads the label of the CAD model from the name of the file. i.e. 0-2.step had label 0 and is number 2 of the CAD models
with this label.
"""

from OCC.Extend.DataExchange import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

from src.mesh_operations import get_mesh_information
from src.brep_operations import get_brep_information
from src.write_hier_graphs_single import write_h5_file


def check_face_in_map(topo, label_map):
    """Function to check loaded faces match face label map."""
    failure = False
    faces = list(topo.faces())

    for face in faces:
        if face not in label_map:
            failure = True
            break

    return failure


def read_step_with_labels(filename):
    """Reads STEP file with labels on each B-Rep face."""
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    id_map = {}
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())

    for face in faces:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            nameid = name
            id_map[face] = nameid

    return shape, id_map, topo


def triangulate_shape(shape, linear_deflection=0.9, angular_deflection=0.5):
    """Triangulate the shape into a faceted mesh."""
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    assert mesh.IsDone()


def create_hier_graphs(h5_path, step_path, shape_name):
    try:
        shape, label_map, topo = read_step_with_labels(step_path)
        failure_test = check_face_in_map(topo, label_map)

        if failure_test:
            print("Issue with face map")
        else:
            triangulate_shape(shape)
            work_faces, work_edges, faces = get_brep_information(shape, label_map)
            facet_dict, edge_facet_link, facet_face_link, node_dict = get_mesh_information(shape)
            write_h5_file(h5_path, shape_name, work_faces, facet_dict, work_edges, edge_facet_link)

    except Exception as error:
        print(error)


if __name__ == '__main__':
    """Make sure you CAD files are labelled: featurelabel_modelnumber.step i.e. 1_4.step where 1 is the feature class 
    label and 4 is the CAD model number."""
    import glob
    import os

    split = "train"
    h5_path = f"./data/{split}.h5"
    step_files = glob.glob(f"./data/{split}/*.step")

    for i, step_file in enumerate(step_files):
        if i % 500 == 0:
            print(f"Count: {i}")
        step_id = os.path.basename(step_file)[:-len(".step")]
        create_hier_graphs(h5_path, step_file, step_id)
