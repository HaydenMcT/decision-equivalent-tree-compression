from models.treefarms_wrapper import create_tree_classifier
from sympy import to_dnf, simplify_logic, And, Symbol

header = ['x <= 2', 'x <= 1']
tree_json_A = {
        "feature": 0,
        "name": "x <= 2",
        "relation": "==",
        "reference": "true",
        "true": {
            "prediction": 1,
            "name": "Prediction"
        },
        "false": {
            "prediction": 0,
            "name": "Prediction"
        }
    }

tree_json_AB = {
        "feature": 0,
        "name": "x <= 2",
        "relation": "==",
        "reference": "true",
        "true": {
            "feature": 1,
            "name": "x <= 1",
            "relation": "==",
            "reference": "true",
            "true": {
                "prediction": 1,
                "name": "Prediction"
            },
            "false": {
                "prediction": 0,
                "name": "Prediction"
            }
        },
        "false": {
            "prediction": 0,
            "name": "Prediction"
        }
    }

tree_json_A_XOR_B = {
        "feature": 0,
        "relation": "==",
        "reference": "true",
        "name": "x <= 2",
        "true": {
            "feature": 1,
            "name": "x <= 1",
            "relation": "==",
            "reference": "true",
            "true": {
                "prediction": 0,
                "name": "Prediction"
            },
            "false": {
                "prediction": 1,
                "name": "Prediction"
            }
        },
        "false": {
            "feature": 1,
            "name": "x <= 1",
            "relation": "==",
            "reference": "true",
            "true": {
                "prediction": 1,
                "name": "Prediction"
            },
            "false": {
                "prediction": 0,
                "name": "Prediction"
            }
        }
    }
    
tree_json_A_AND_NOT_B = {
        "feature": 1,
        "relation": "==",
        "reference": "true",
        "name": "x <= 1",
        "true": {
            "prediction": 0,
            "name": "Prediction"
        },
        "false": {
            "feature": 0,
            "name": "x <= 2",
            "relation": "==",
            "reference": "true",
            "true": {
                "prediction": 1,
                "name": "Prediction"
            },
            "false": {
                "prediction": 0,
                "name": "Prediction"
            }
        }
    }

def check_identity(tree_json):
    tree = create_tree_classifier(tree_json, tree_type='DNF')
    assert(tree.diff(tree) == False)
    assert(tree.sim(tree) == True)

def test_identity():
    check_identity(tree_json_A)
    check_identity(tree_json_AB)
    check_identity(tree_json_A_XOR_B)
    check_identity(tree_json_A_AND_NOT_B)

def test_overlap():
    tree_A = create_tree_classifier(tree_json_A, tree_type='DNF', header = header)
    tree_AB = create_tree_classifier(tree_json_AB, tree_type='DNF', header = header)
    assert(tree_A.diff(tree_AB) == to_dnf(And(~Symbol('`x <= 1`'), Symbol('`x <= 2`'))))

def test_sim_diff_nonoverlap():
    tree_A = create_tree_classifier(tree_json_A, tree_type='DNF')
    tree_AB = create_tree_classifier(tree_json_AB, tree_type='DNF')
    # The sim and diff between two trees should always be non-overlapping
    for t1_json in [tree_json_A, tree_json_AB, tree_json_A_XOR_B, tree_json_A_AND_NOT_B]:
        for t2_json in [tree_json_A, tree_json_AB, tree_json_A_XOR_B, tree_json_A_AND_NOT_B]:
            t1 = create_tree_classifier(t1_json, tree_type='DNF', header = header)
            t2 = create_tree_classifier(t2_json, tree_type='DNF', header = header)
            assert simplify_logic(
                And(t1.sim(t2), t1.diff(t2))
            ) == False

def test_dontcare():
    """
    tree_xor encodes (x <= 2 XOR x <= 1) = (x <= 2 & ~x <= 1) | (x <= 1 & ~x <= 2)
    tree_and encodes x <= 2 & ~x <= 1
    These trees should differ only when x <= 2 = 0, x <= 1 = 1; 
    But this case is impossible when considering that x <= 1 implies x <= 2
    Meaning the two trees should be identical when we initialize the 
    DNF Trees to consider dontcare conditions from having multiple <= 
    splits on the same variable. 
    """
    tree_xor = create_tree_classifier(tree_json_A_XOR_B, tree_type='DNF', header = header)
    tree_and = create_tree_classifier(tree_json_A_AND_NOT_B, tree_type='DNF', header = header)

    dontcare = to_dnf(And(Symbol('`x <= 1`'), ~Symbol('`x <= 2`')))

    assert tree_xor.diff(tree_and) != False, tree_xor.diff(tree_and)
    assert tree_xor.diff(tree_and, external_dontcare=dontcare) == False, tree_xor.diff(tree_and, external_dontcare=dontcare)
    
    assert tree_xor.sim(tree_and) != True, tree_xor.sim(tree_and)
    assert tree_xor.sim(tree_and, external_dontcare=dontcare) == True, tree_xor.sim(tree_and, external_dontcare=dontcare)

def test_dontcare_stored():
    """
    tree_xor encodes (x <= 2 XOR x <= 1) = (x <= 2 & ~x <= 1) | (x <= 1 & ~x <= 2)
    tree_and encodes x <= 2 & ~x <= 1
    These trees should differ only when x <= 2 = 0, x <= 1 = 1; 
    as such, adding this case as a dontcare condition should make the
    two trees identical
    """
    bin_to_original_dict = {
        "x <= 2": "x",
        "x <= 1": "x"
    }
    tree_xor = create_tree_classifier(tree_json_A_XOR_B, tree_type='DNF', bin_to_original_dict=bin_to_original_dict, separator_str=" <= ", header = header)
    tree_and = create_tree_classifier(tree_json_A_AND_NOT_B, tree_type='DNF', bin_to_original_dict=bin_to_original_dict, separator_str=" <= ", header = header)

    assert tree_xor.diff(tree_and) == False, tree_xor.diff(tree_and)
    assert tree_xor.sim(tree_and) == True, tree_xor.sim(tree_and)

def test_dontcare_cross_conditions():
    """
    Tree 1 returns True iff x <= 1, and Tree 2 returns True iff x <= 2
    Describing their difference only requires specifying x<=2 and ~x<=1, 
    with no other conditions, if we're considering dontcare conditions
    imposed by continuous features. This test checks whether these
    dontcare conditions imposed by continuous features are handled with
    DNFTrees directly, without requiring an external dontcare condition.
    """
    tree_json_1 = {
        "feature": 0,
        "name": "x <= 1",
        "relation": "==",
        "reference": "true",
        "true": {
            "prediction": 1,
            "name": "Prediction"
        },
        "false": {
            "prediction": 0,
            "name": "Prediction"
        }
    }
    tree_json_2 = {
        "feature": 1,
        "name": "x <= 2",
        "relation": "==",
        "reference": "true",
        "true": {
            "prediction": 1,
            "name": "Prediction"
        },
        "false": {
            "prediction": 0,
            "name": "Prediction"
        }
    }
    bin_to_original_dict = {
        "x <= 2": "x",
        "x <= 1": "x"
    }
    tree_1 = create_tree_classifier(tree_json_1, tree_type='DNF', bin_to_original_dict=bin_to_original_dict, separator_str=" <= ", header = header)
    tree_2 = create_tree_classifier(tree_json_2, tree_type='DNF', bin_to_original_dict=bin_to_original_dict, separator_str=" <= ", header = header)
    assert tree_1.diff(tree_2) == to_dnf(And(Symbol('`x <= 2`'), ~Symbol('`x <= 1`'))), tree_1.diff(tree_2)
