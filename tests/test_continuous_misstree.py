from models.tree_classifier_wrapper import create_tree_classifier
from models.treefarms_wrapper import construct_tree_rset, DEFAULT_TF_CONFIG
import pandas as pd
import numpy as np
from preprocessing.binarize import bin_by_quantile

def test_basic_set_finding():
    """
    Test whether our prediction vector extraction method
    always matches the preds of TreeClassifier
    """
    data = pd.DataFrame(np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]),
                        columns=["a", "b", "c", "y"])
    X_binned, bin_name_to_original = bin_by_quantile(data.drop(columns=['y']), n_bins=2)
    y = data['y']

    model = construct_tree_rset(X_binned, y, DEFAULT_TF_CONFIG, tree_type='ContinuousMissTree')

def test_for_cts_overlap():
    data = pd.DataFrame(np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]),
                        columns=["a <= 1", "a <= 2", "z <= 1", "y"])
    X = data.drop(columns=['y'])
    y = data['y']

    model = construct_tree_rset(X, y, DEFAULT_TF_CONFIG, tree_type='ContinuousMissTree')

def test_feature_names_version():
    data = pd.DataFrame(np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]),
                        columns=["a <= 1", "a <= 2", "z <= 1", "y"])
    X = data.drop(columns=['y'])
    y = data['y']

    model = construct_tree_rset(X, y, DEFAULT_TF_CONFIG, tree_type='ContinuousMissTree')
    tree = model.model_list[0].source
    individual_tree = create_tree_classifier(tree, tree_type='ContinuousMissTree', feature_names = ["a <= 1", "a <= 2", "z <= 1"])
    assert(individual_tree == individual_tree)