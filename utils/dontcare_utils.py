from sympy import And, Symbol, Or, Not
import numpy as np
import warnings
# import pandas as pd

class ViableRanges:
    """
    A class that represents, for some set of known thresholded values, 
    the set of values the other thresholds could possibly take.
    For example, if we know a > 5 and a < 10, the threshold a < 3 is outside the viable
    range and must always be false.
    """
    def __init__(self, known_ftrs: dict = {}, feature_space: list = None, original_name_getter=None):
        """
        Args:
            known_ftrs: a dict corresponding to boolean assignments of variables, 
                of the form
                { <binarized_feature_name>: <boolean value of feature>, 
                    ... 
                }
            original_name_getter: a function that returns the name of the
                original feature and the corresponding threshold. 
                Defaults to splitting on " <= "
            feature_space (optional): ordered list of names in the original feature space. 
                                  Needed only if we want to generate samples in the range.
        Note: Dontcares are currently implemented assuming <= binning.
        """
        if original_name_getter is None:
            original_name_getter = lambda x: (x.split(" <= ")[0], float(x.split(" <= ")[1]))
        self.original_name_getter = original_name_getter

        # A dictionary mapping from an original variable to two values:
        # 1) The precomputed max false threshold
        # 2) The precomputed min true threshold
        self.og_to_range_vals = {
            self.original_name_getter(k)[0]: {"max_false": -np.inf, "min_true": np.inf} for k in known_ftrs
        }
        self.update_ranges(known_ftrs)
        self.feature_space = feature_space

    def deduce_threshold_state(
        self,
        binarized_variable_name: str
    ):
        """
        Infer the state of (X <= target) given known thresholded boolean values.

        Args:
            binarized_variable_name: the name of the binarized variable of interest
                (e.g., x1 <= 25)

        Returns:
            True / False if it must be one of them, or None if indeterminate.
        """
        variable_name, threshold_value = self.original_name_getter(binarized_variable_name)

        if variable_name not in self.og_to_range_vals:
            return None

        max_false = self.og_to_range_vals[variable_name]["max_false"]
        min_true = self.og_to_range_vals[variable_name]["min_true"]

        # E.g., We know X <= 3 = False, and are asking about X < 2
        if threshold_value <= max_false:
            return False
        # E.g., We know X <= 8 = True, and are asking about X < 10
        if threshold_value >= min_true:
            return True
        return None
    
    def update(self, binarized_variable_name: str, value: bool):
        '''
        Update the viable ranges based on a new known feature value
        Return the old range, so it can be restored later
        '''
        variable_name, threshold_value = self.original_name_getter(binarized_variable_name)
        old_range = None
        if variable_name not in self.og_to_range_vals:
            self.og_to_range_vals[variable_name] = {"max_false": -np.inf, "min_true": np.inf}
        else:
            old_range = self.og_to_range_vals[variable_name].copy()
        if value and self.og_to_range_vals[variable_name]["min_true"] > threshold_value:
            self.og_to_range_vals[variable_name]["min_true"] = threshold_value
        if not value and self.og_to_range_vals[variable_name]["max_false"] < threshold_value:
            self.og_to_range_vals[variable_name]["max_false"] = threshold_value
        return variable_name, old_range
    
    def restore_old_range(self, variable_name, old_range):
        if old_range is None:
            del self.og_to_range_vals[variable_name]
        else:
            self.og_to_range_vals[variable_name] = old_range

    def update_ranges(self, known_ftrs: dict):
        for k in known_ftrs:
            cur_var, cur_thresh = self.original_name_getter(k)

            if known_ftrs[k] == True and self.og_to_range_vals[cur_var]["min_true"] > cur_thresh:
                self.og_to_range_vals[cur_var]["min_true"] = cur_thresh

            if known_ftrs[k] == False and self.og_to_range_vals[cur_var]["max_false"] < cur_thresh:
                self.og_to_range_vals[cur_var]["max_false"] = cur_thresh

    def generate_sample(self) -> np.array:
        '''
        Generate a sample that is consistent with the viable ranges (not randomly sampled).
        For now, each feature is false unless it is forced to be true.
        Works very scalably for continuous features with <= thresholds; doesn't work if 
        there are more complex dependencies.
        '''
        if self.feature_space is None:
            raise Exception("Must provide feature space to generate sample")
        sample = []
        for var in self.feature_space:
            val = self.deduce_threshold_state(var)
            if val is None: # not determined
                val = False
            sample.append(val)
        return np.array(sample)     


def get_vars_used_by_source(source):
    """
    Extracts the used variables from a tree in json format. 
    If the json format of the tree does not include 'name"
    fields for the feature name in each split, uses feature_x
    as the name, replacing x with the index of the feature
    """
    def _get_used_vars(cur_leaf):
        if "prediction" in cur_leaf:
            return []
        else:
            if 'name' in cur_leaf:
                return [cur_leaf['name']] + _get_used_vars(cur_leaf['true']) + _get_used_vars(cur_leaf['false'])
            else:
                return [f"feature_{cur_leaf['feature']}"] + _get_used_vars(cur_leaf['true']) + _get_used_vars(cur_leaf['false'])
    return _get_used_vars(source)

def add_names(source, header):
    """
    Adds 'name' fields to each split in a tree json,
    using the provided header to map feature indices
    to feature names. Modifies the tree json in place.
    Needed so that get_vars_used_by_source can 
    access feature names from treefarms
    Args:
        source : dict -- The tree in json format, as output
            by treefarms
        header : list<str> -- The list of feature names, where
            header[i] is the name of feature i
    """
    if header is None:
        raise Exception("Must provide header to add_names")
    def _add_names(cur_leaf):
        if "prediction" not in cur_leaf:
            cur_leaf['name'] = header[cur_leaf['feature']]
            _add_names(cur_leaf['true'])
            _add_names(cur_leaf['false'])
    _add_names(source)

def get_dontcare_from_bin_map(
    bin_to_original_dict, 
    separator_str="_<=_", 
    used_vars=None,
    validation_X=None
):
    """
    Extracts a logical dontcare condition for a given binarization strategy, 
    under the assumption that variables are binarized as, eg, x < 0, x < 0.5, x < 1...
    Args:
        bin_to_original_dict : dict -- A mapping from each column name in 
            the binarized dataset to its source column name.
        separator_str : str -- The string that, in binarized column names,
            separates the original column name from the threshold value
        used_vars : list<str> -- The list of binarized variables used in the
            DNF of interest. This is used to avoid adding unnecessary terms to
            our dontcare condition. Defaults to include all variables
        validation_X : pd.DataFrame -- Binarized data to validate our bin 
            strategy against, eg, check whether a < 2 is always True when
            a < 1. If None, no validation is performed
    Returns:
        dontcare_cond : Or -- A logical formula expressing each truth assignment
            that can never happen due to our binarization strategy
    """
    # Construct a mapping from original var to list of binned vars
    og_var_to_bin_list = {}
    for k in bin_to_original_dict:
        if bin_to_original_dict[k] not in og_var_to_bin_list:
            og_var_to_bin_list[bin_to_original_dict[k]] = [k]
        else:
            og_var_to_bin_list[bin_to_original_dict[k]] = og_var_to_bin_list[bin_to_original_dict[k]] + [k]

    dontcare_ands = []
    for og_var in og_var_to_bin_list:
        # Grab each binned var for this variable, and their corresponding
        # threshold values
        cur_splits = og_var_to_bin_list[og_var]
        thresh_vals = [float(c.split(separator_str)[1]) for c in cur_splits]

        # Sort in ascending order by threshold
        split_order = np.argsort(thresh_vals)

        # For each binarized variable
        cur_splits_array = np.array(cur_splits)
        for idx, binned_var in enumerate(cur_splits_array[split_order]):
            # Consider each variable with a threshold guaranteed to be met if
            # the one for binned_var is met. Eg, binned_var = a < 1, guaranteed_var = a < 2
            for guaranteed_var in cur_splits_array[split_order][idx+1:]:
                # If both vars of interest are in our target DNF, add to the dontcare
                # condition that (a < 0) and ~(a < 1) is not possible
                if (used_vars is None) or (binned_var in used_vars and guaranteed_var in used_vars):
                    dontcare_ands.append(
                        And(Symbol(f'`{binned_var}`'), Not(Symbol(f'`{guaranteed_var}`')))
                    )

                    if validation_X is not None:
                        assert (validation_X[validation_X[used_vars] == 1][guaranteed_var] == 1).all()
    dontcare_cond = Or(*dontcare_ands)
    return dontcare_cond
