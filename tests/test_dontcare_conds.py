import numpy as np
import pandas as pd
from preprocessing.data import load_data, prepare_binarized_data
from sympy import And, simplify_logic, Symbol, Or
from utils.dontcare_utils import get_dontcare_from_bin_map

def test_tautology_from_dontcare():
    binarization_config = {
        "missing_values": np.nan,
        "columns_to_drop": None,
        "num_quantiles": 2,
        "gosdt_guesses": False,
        "keep_missingness_after_binarization": False
    }
    X_all = pd.DataFrame({
        f"x{i}": np.random.rand(100) for i in range(10)
    })
    y_all = pd.DataFrame({
        "y": np.random.randint(0, 2, 100)
    })
    X_train, y_train, X_test, y_test, bin_to_original_dict = prepare_binarized_data(
        X_all,
        y_all, 
        **binarization_config
    )

    dontcare_cond = get_dontcare_from_bin_map(
        bin_to_original_dict,
        separator_str="_<=_",
        used_vars=[X_train.columns[0], X_train.columns[1]],
        validation_X=X_train
    )

    # For the DNF (x1 < 1) & ~(x1 < 2), reasonable dontcare conditions
    # should simplify to False

    """
    For the DNF (x1 < 1) & ~(x1 < 2), we have truth table
    x1 < 1  | x1 < 2    | (x1 < 1) & ~(x1 < 2)
    0       | 0         | 0
    1       | 0         | 1
    0       | 1         | 0
    1       | 1         | 0
    A reasonable dontcare will tell us to ignore the case where 
    (x1 < 1) = True, (x1 < 2) = False, eliminating the second row,
    which makes this False
    """
    dnf = And(Symbol(f'`{X_train.columns[0]}`'), ~Symbol(f'`{X_train.columns[1]}`'))
    reduced = simplify_logic(
        dnf, 
        dontcare=dontcare_cond,
        force=True
    )
    assert reduced == False, "Error: Failed to evaluate OR case to True"

    """
    For the DNF ~(x1 < 1) | (x1 < 2), we have truth table
    x1 < 1  | x1 < 2    | ~(x1 < 1) OR (x1 < 2)
    0       | 0         | 1
    1       | 0         | 0
    0       | 1         | 1
    1       | 1         | 1
    A reasonable dontcare will tell us to ignore the case where 
    (x1 < 1) = False, (x1 < 2) = True, eliminating the second row,
    which makes this a tautology
    """
    dnf = Or(~Symbol(f'`{X_train.columns[0]}`'), Symbol(f'`{X_train.columns[1]}`'))
    reduced = simplify_logic(
        dnf, 
        dontcare=dontcare_cond,
        force=True
    )

    assert reduced == True, "Error: Failed to evaluate OR case to True"
