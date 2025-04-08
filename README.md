# Hierarchical B-Rep Graphs
This repo contains scripts for generating hierarchical B-Rep graphs of CAD models, and generating batches for inputting into neural network.

The MFCAD++ dataset can be downloaded from here: https://pure.qub.ac.uk/en/datasets/mfcad-dataset-dataset-for-paper-hierarchical-cadnet-learning-from

## Requirements
- Python >= 3.7.9
- pythonocc-core >= 7.4.1 (more info here: https://github.com/tpaviot/pythonocc-core)
- occt >= 7.4.0 (more info here: https://github.com/tpaviot/pythonocc-core)
- h5py >= 1.10.6
- Numpy >= 1.19.2

## Instructions
### Multiple Machining Features
- This code requires the CAD models to be in a STEP file format (.step).
- Split the CAD model dataset into "train", "val" and "test" directories.
- Run *generate_hier_graphs.py* for each dataset split changing the split variable. This generates the hierarchical B-Rep graphs for each CAD model.
- Once the hierarchical B-Rep graphs have been generated, run *generate_batches.py* to create the batches for inputting data into neural network.
- Use batches to train Hierarchical CADNet using code in this repo: https://gitlab.com/qub_femg/machine-learning/hierarchical-cadnet.

### Single Machining Features
- This is for CAD models where there is one class label per CAD model rather than per B-Rep face.
- Split the CAD model dataset into "train", "val" and "test" directories.
- The code is set up that the STEP file is named "classlabel-filenumber.step" where "classlabel" is the class label of the CAD model and "filenumber" is the given number in the dataset of that model for the given label. i.e. "1-10.step" where the CAD model would have a class label of 1 and it would be the 10th CAD model with this label.
- Run *generate_single_feat_graphs.py* for each dataset split changing the split variable. This generates the hierarchical B-Rep graphs for each CAD model.
- Once the hierarchical B-Rep graphs have been generated, run *generate_batches.py* to create the batches for inputting data into neural network.
- Use batches to train Hierarchical CADNet using code in this repo: https://gitlab.com/qub_femg/machine-learning/hierarchical-cadnet.

### Subgraph Construction
In section 6.4 of the Hierarchical CADNet paper, a method of decomposing the hierarchical B-Rep graphs into subgraphs using the edge convexity information. This approach
is similar to those found in attribute adjacency graphs (AAG). This was used as a method of feature engineering to simplify the input data and was shown to improve results on a number of the complex CAD models tested in the paper.

To generate your own subgraphs use the following process:
- Run *generate_hier_graphs.py* for the required dataset split or individual CAD model (STEP file).
- Run *generate_sub_graph.py* to generate the subgraphs for your CAD models. Make sure to change the read and write file paths for your own files.
- Once the subgraphs have been generated, run *generate_batches_sub.py* to create the batches for inputting data into neural network.

## Citation
Please cite this work if used in your research:

    @article{hierarchicalcadnet2022,
      Author = {Andrew R. Colligan, Trevor. T. Robinson, Declan C. Nolan, Yang Hua, Weijuan Cao},
      Journal = {Computer-Aided Design},
      Title = {Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition},
      Year = {2022}
      Volume = {147}
      URL = {https://www.sciencedirect.com/science/article/abs/pii/S0010448522000240}
    }

## Funding
This project was funded through DfE funding.
