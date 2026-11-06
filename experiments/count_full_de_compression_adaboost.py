import pandas as pd
from models.ensemble_wrapper import AdaBoostWrapper
from preprocessing.data import load_data
import argparse
from sklearn.model_selection import KFold
import numpy as np
import time
from utils.tree_to_dict import unbinned_sklearn_tree_to_dict
from pystreed import STreeDClassifier
from sklearn.tree import DecisionTreeClassifier
from models.treefarms_wrapper import TreeFarmsWrapper
from utils.dontcare_utils import ViableRanges

# utils to run equivalence checking (Section 4.1 from paper)

class ContinuousViableRanges(ViableRanges):
    """
    A class like ViableRanges, but updated to generate samples in a continuous 
    feature space rather than a discretized one. Expects that variable_names
    for generation are just integers and splits are of the form 
    '<feature_index> <= <threshold>'.
    """
    def generate_sample(self) -> np.array:
        if self.feature_space is None:
            raise Exception("Must provide feature space to generate sample")
        sample = np.zeros(len(self.feature_space))
        for var, r in self.og_to_range_vals.items():
            # viable values satisfy max_false < x <= min_true
            if np.isfinite(r["min_true"]):
                sample[int(var)] = r["min_true"]
            elif np.isfinite(r["max_false"]):
                sample[int(var)] = r["max_false"] + 1.0
        return sample

def findMismatch(S: ViableRanges, T: dict, T_prime: dict):
    # same as in compress_rset, except here queries are features rather than names, of the form
    # '<feature_index> <= <threshold>'
    if 'prediction' in T:
        if 'prediction' in T_prime:
            if T['prediction'] != T_prime['prediction']:
                return S.generate_sample()
            else:
                return None
        return findMismatch(S, T_prime, T) # note the swapped order; T is a leaf so we iterate over T' the rest of the DFS
    else:
        query = T['feature']
        # check if known
        known_val = S.deduce_threshold_state(query)
        if known_val is not False: # if true is possible: 
            cached_range_before_update = S.update(T['feature'], True)
            mismatch = findMismatch(S, T['true'], T_prime)
            S.restore_old_range(*cached_range_before_update)
            if mismatch is not None:
                return mismatch
        if known_val is not True: # if false is possible:
            cached_range_before_update = S.update(T['feature'], False)
            mismatch = findMismatch(S, T['false'], T_prime)
            S.restore_old_range(*cached_range_before_update)
            if mismatch is not None:
                return mismatch
        return None
    
def WitnessDE(T, T_prime, feature_space=[]):
    # same as in compress Rset but for continuous viable ranges
    S = ContinuousViableRanges(feature_space=feature_space)
    return findMismatch(S, T, T_prime)

'''
Maps from collection to a compressed set of trees, each tree is a representative of an equivalence class of trees in the original collection, 
where two trees are in the same equivalence class if they are identical on all inputs in the original feature space.
'''
def compress_rset(collection: TreeFarmsWrapper, feature_space = []):
    X_diff = []
    predmap = {}
    for tree in collection:
        collision = False
        preds = ''
        i = -1
        while i < len(X_diff):
            if i > -1:
                x = X_diff[i]
                preds = preds + str(tree.classify(x)[0])
            i += 1
            if preds in predmap:
                collision = True
                colliding_tree = predmap.pop(preds)
                colliding_tree_preds = preds
                while i < len(X_diff):
                    x = X_diff[i]
                    colliding_tree_pred = str(colliding_tree.classify(x)[0])
                    tree_pred = str(tree.classify(x)[0])
                    preds += tree_pred
                    colliding_tree_preds += colliding_tree_pred
                    if colliding_tree_pred != tree_pred:
                        # resolved collision, insert both trees with their respective preds and break
                        predmap[colliding_tree_preds] = colliding_tree
                        predmap[preds] = tree
                        i = len(X_diff) # break to next tree
                    else:
                        i += 1
                if colliding_tree_preds == preds:
                    distinguishing_point = WitnessDE(tree.source, colliding_tree.source, feature_space=feature_space)
                    if distinguishing_point is not None:
                        X_diff.append(distinguishing_point)
                        tree_pred = tree.classify(distinguishing_point)[0]
                        colliding_tree_pred = colliding_tree.classify(distinguishing_point)[0]
                        # assert tree_pred != colliding_tree_pred, "WitnessDE returned a point that does not distinguish the two trees"
                        predmap[preds + str(colliding_tree_pred)] = colliding_tree
                        predmap[preds + str(tree_pred)] = tree
                    else:
                        # insert just one tree, since they're from the same DE class
                        predmap[colliding_tree_preds] = colliding_tree
        if not collision:
            predmap[preds] = tree
    return predmap

original_name_getter = lambda x: (x.split(" <= ")[0], float(x.split(" <= ")[1]))
class Implicant:
    """
    A class that represents, for some set of known thresholded values, 
    the set of values the other thresholds could possibly take.
    For example, if we know a > 5 and a < 10, the threshold a < 3 is outside the viable
    range, so we know enough to say the entire implicant is false.
    (Largely redundant to ViableRanges; kept because it is needed for the sklearn conversion below)
    """
    def __init__(self, required_ftrs: dict):
        """
        Args:
            required_ftrs: a dict corresponding to boolean assignments of variables,
                of the form
                { <binarized_feature_name>: <boolean value of feature>, 
                    ... 
                }.
                Iff all these assignments are true for some sample, then this implicant holds.
            """
        self.required_ftrs = required_ftrs

        # A dictionary mapping from an original variable to two values:
        # 1) The precomputed max false threshold
        # 2) The precomputed min true threshold
        self.og_to_range_vals = {
        }
        for k in self.required_ftrs:
            cur_var, cur_thresh = original_name_getter(k)

            if cur_var not in self.og_to_range_vals:
                self.og_to_range_vals[cur_var] = {"max_false": -np.inf, "min_true": np.inf}
            if self.required_ftrs[k] == True and self.og_to_range_vals[cur_var]["min_true"] > cur_thresh:
                self.og_to_range_vals[cur_var]["min_true"] = cur_thresh

            if self.required_ftrs[k] == False and self.og_to_range_vals[cur_var]["max_false"] < cur_thresh:
                self.og_to_range_vals[cur_var]["max_false"] = cur_thresh
        
        # track relevant splits
        # Relevant splits is a list of (continuous feature, threshold) tuples, each of which
        # corresponds to values we'd still need to query to verify whether required_ftrs will
        # be satisfied. It is initialized to match the og_to_range_vals, but is updated separately.
        # For example, if we already know a < 5, then for a variable with min_true = a < 5, the
        # og_to_range_vals values should stay the same, but we no longer need to verify a < 5
        # to determine the value of the split. Once relevant_splits is empty, the Implicant
        # is confirmed satisfied.
        self.relevant_splits = set()
        for cts_ftr in self.og_to_range_vals:
            min_true = self.og_to_range_vals[cts_ftr]['min_true']
            if min_true != np.inf: #don't need to confirm outer extrema
                self.relevant_splits.add((cts_ftr, min_true))
            max_false = self.og_to_range_vals[cts_ftr]['max_false']
            if max_false != -np.inf:
                self.relevant_splits.add((cts_ftr, max_false))
            
    def complete(self):
        return len(self.relevant_splits) == 0

    def update_with_known_value(self, binarized_variable_name, is_feature_negated=False) -> bool :
        # return false if the implicant is guaranteed to be contradicted  by the given feature's value (that is, if the given feature value is opposite the value expected by the implicant) 
        # return true if implicant guaranteed true after incorporating the given feature's value
        # else return None
        variable_name, threshold_value = original_name_getter(binarized_variable_name)

        if variable_name not in self.og_to_range_vals:
            return None  
        max_false = self.og_to_range_vals[variable_name]["max_false"]
        min_true = self.og_to_range_vals[variable_name]["min_true"]

        if is_feature_negated:
            if threshold_value >= min_true:
                return False
            elif max_false <= threshold_value:
                self.og_to_range_vals[variable_name]["max_false"] = threshold_value
                # no longer relevant to query the old or new threshold further,
                # since the truth values of both are known
                if (variable_name, max_false) in self.relevant_splits:
                    self.relevant_splits.remove((variable_name, max_false))
                    if self.complete():
                        return True
        else:
            if threshold_value <= max_false:
                return False
            elif min_true >= threshold_value:
                self.og_to_range_vals[variable_name]["min_true"] = threshold_value
                # no longer relevant to query the old or new threshold further,
                # since the truth values of both are known
                if (variable_name, min_true) in self.relevant_splits:
                    self.relevant_splits.remove((variable_name, min_true))
                    if self.complete():
                        return True
        return None
    def query_value(self, variable_name, threshold_value):
        # returns True if variable_name <= threshold_value is known true
        # False if variable_name <= threshold value is known false
        # else None
        if variable_name not in self.og_to_range_vals:
            return None  
        max_false = self.og_to_range_vals[variable_name]["max_false"]
        min_true = self.og_to_range_vals[variable_name]["min_true"]
        if threshold_value >= min_true:
            return True
        elif threshold_value <= max_false:
            return False
        else:
            return None

def nobin_sklearn_tree_to_leaf_implicants(tree) -> dict:
    true_implicants = []
    false_implicants = []
    def _recurse_to_leaf_implicants(node, known):
        if 'prediction' in node:
            prediction = node['prediction']
            if prediction == 0:
                false_implicants.append(Implicant(known))
            else:
                true_implicants.append(Implicant(known))
        else:
            feature = node['feature']
            new_known = known.copy()
            #covers weird rounding case
            if feature in new_known:
                if new_known[feature] == False:
                    _recurse_to_leaf_implicants(node['false'], new_known)
                else:
                    _recurse_to_leaf_implicants(node['true'], new_known)
            else:
                new_known[feature] = False
                _recurse_to_leaf_implicants(node['false'], new_known)
                new_known = known.copy()
                new_known[feature] = True
                _recurse_to_leaf_implicants(node['true'], new_known)

    _recurse_to_leaf_implicants(unbinned_sklearn_tree_to_dict(tree), {})
    return true_implicants, false_implicants

DEFAULT_CONFIG = {"num_quantiles": 0, "gosdt_guesses": True, "keep_missingness_after_binarization": True}
CONFIGS = {
    'compas_complete.csv': DEFAULT_CONFIG,
    'wine_quality.csv': DEFAULT_CONFIG,
    'wisconsin.csv': DEFAULT_CONFIG,
    'coupon_full.csv': DEFAULT_CONFIG,
    'fico_complete.csv': DEFAULT_CONFIG,
    'netherlands.csv': DEFAULT_CONFIG,
    'spiral.csv': DEFAULT_CONFIG,
    'tic-tac-toe.csv': DEFAULT_CONFIG,
    'iris_virginica.csv': DEFAULT_CONFIG,
    'iris_versicolor.csv': DEFAULT_CONFIG,
    'iris_setosa.csv': DEFAULT_CONFIG,
    'broward_general_2y.csv': DEFAULT_CONFIG,
}



def get_vars_used_by_source(source, ftr_names):
    dict_of_vals = {}
    def _get_used_vars(cur_leaf):
        if "prediction" in cur_leaf:
            return []
        else:
            feature, thresh = cur_leaf['feature'].split(' <= ')
            if feature in dict_of_vals:
                dict_of_vals[feature].add(float(thresh))
            else:
                dict_of_vals[feature] = set([float(thresh)])
            _get_used_vars(cur_leaf['true'])
            _get_used_vars(cur_leaf['false'])
    _get_used_vars(source)
    for key in dict_of_vals:
        dict_of_vals[key] = sorted(dict_of_vals[key])
    return dict_of_vals

def compress_with_streed(tree, feature_names, depth = 5):
    X_streed = []
    y_streed = []

    # fast streed version:
    used_features = get_vars_used_by_source(unbinned_sklearn_tree_to_dict(tree), feature_names)
    pos_leaves, neg_leaves = nobin_sklearn_tree_to_leaf_implicants(tree)
    for pos in pos_leaves:
        row = []
        for ftr in used_features:
            for value in used_features[ftr]:
                feature_value = pos.query_value(ftr, value)
                if feature_value is None:
                    row.append(-1)
                elif feature_value:
                    row.append(1)
                else:
                    row.append(0)
        X_streed.append(row)
        y_streed.append(1)
    for neg in neg_leaves:
        row = []
        for ftr in used_features:
            for value in used_features[ftr]:
                feature_value = neg.query_value(ftr, value)
                if feature_value is None:
                    row.append(-1)
                elif feature_value:
                    row.append(1)
                else:
                    row.append(0)
        X_streed.append(row)
        y_streed.append(0)

    X_streed = np.array(X_streed)
    y_streed = np.array(y_streed)

    # settle parameters
    depth = depth#min(X.shape[0]-1, 20) # anything deeper than this is guaranteed less sparse than existing tree
    complexity_penalty = 1/((X_streed.shape[0]-1)*X_streed.shape[0]+1) # guarantee any tree, even a leaf at root, that makes 
                                                        # at least one error is suboptimal relative to the original tree

    acc = np.nan
    compression_time = np.nan
    num_splits = np.nan
    try: 
        model = STreeDClassifier("cost-complex-accuracy", max_depth = depth, cost_complexity=complexity_penalty, use_terminal_solver=False, time_limit=300)
        start = time.perf_counter()
        model.fit(X_streed, y_streed)
        end = time.perf_counter()
        num_splits = model.get_n_leaves() - 1
        compression_time = end - start
        # also assert accuracy, etc
        acc = model.score(X_streed,y_streed)
    except Exception as e:
        print(f"error encountered in Streed! Message: {e}")
    return num_splits, acc, compression_time

BASE_CONFIG = {
    'estimator': DecisionTreeClassifier(max_depth=3),
    'n_estimators': 100,
    'random_state': 42
}
DEPTHS = [1, 2, 3, 4, 5]
TREE_TYPES = ['ContinuousMissTreeUnbinned']

def experiment(dataset='fico_full.csv', balance=False, depth=3, just_bag = False, tree_type='MissTree', guess_n_est=50, guess_max_depth=3):
    X, y = load_data(dataset, balance=balance)
    binarization_config = CONFIGS[dataset]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = kf.split(X)
    result_rows = []
    for train_index, test_index in folds:
        result_row = {'dataset': dataset, 
                      'balanced': balance,
                      'depth': depth,
                      'tree_type': tree_type
                      }
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        result_row['num_features'] = X_train.shape[1]
        result_row['num_samples'] = X_train.shape[0]

        # Integer feature names
        X_train.columns = [str(s) for s in range(X_train.shape[1])]
        X_test.columns = [str(s) for s in range(X_train.shape[1])]

        # # Integer feature names
        X_train.columns = range(X_train.shape[1])
        X_test.columns = range(X_train.shape[1])

        config = BASE_CONFIG.copy()
        config['estimator'] = DecisionTreeClassifier(max_depth=depth)
        start = time.perf_counter()
        model = AdaBoostWrapper(adaboost_config = config, tree_type = tree_type, feature_names = X_train.columns)
        equiv_check_time = model.fit(X_train, y_train, return_compress_time=True)
        result_row['training_time'] = time.perf_counter() - start
        result_row['tree_wrapper_time'] = equiv_check_time

        # Get number of duplicate trees:
        index_map = {id(tree): idx for idx, tree in enumerate(model.trees)}
        start = time.perf_counter()
        unique_trees = list(compress_rset(model.trees, feature_space=X_train.columns.tolist()).values())
        result_row['pairwise_equivalence_check_time'] = time.perf_counter() - start + equiv_check_time
        result_row['num_trees'] = len(model.trees) # should just match n_est
        result_row['num_unique_trees'] = len(unique_trees)
        result_row['imperfect_acc_counter'] = 0
        result_row['compression_time'] = 0
        result_row['num_timeouts'] = 0
        result_row['streed_compute_time'] = 0
        # compress unique trees:
        old_sparsities = []
        skipped_sparsities = []
        for tree_est in model.model.estimators_:
            old_sparsities.append(tree_est.tree_.node_count)
        result_row['pre_PE_node_count'] = sum(old_sparsities)
        new_sparsities = []
        for tree_est in unique_trees:
            start = time.perf_counter()
            sparsity, acc, runtime = compress_with_streed(model.model.estimators_[index_map[id(tree_est)]].tree_, X_train.columns.tolist())
            result_row['compression_time'] += time.perf_counter() - start
            result_row['num_timeouts'] += (runtime >= 300)
            result_row['streed_compute_time'] += runtime
            if not np.isnan(sparsity):
                new_sparsities.append(sparsity*2 + 1)
                result_row['imperfect_acc_counter'] += (acc < 1.0)
            else:
                skipped_sparsities.append(tree_est.nodes())

        result_row['sparsified_node_count'] = sum(new_sparsities)
        result_row['skipped_node_count'] = sum(skipped_sparsities)
        result_row['num_skipped_trees'] = len(skipped_sparsities)

        result_row['train_acc'] = (model.predict_default(X_train) == y_train).mean()
        result_row['test_acc'] = (model.predict_default(X_test) == y_test).mean()
        result_row['total_compression_time'] = result_row['compression_time'] + result_row['pairwise_equivalence_check_time']
        result_row['total_runtime'] = result_row['training_time'] + result_row['total_compression_time']

        result_rows.append(result_row)
        print(result_row)
    
    results_df = pd.DataFrame(result_rows)
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='Index of the configuration to run')
    parser.add_argument('--rerun', action='store_true', help='Whether to rerun even if results file exists')
    parser.add_argument('--softrerun', action='store_true', help='rerun if results file exists but skipped node count isn\'t zero')
    args = parser.parse_args()
    index_to_run = args.index
    rerun = args.rerun
    softrerun = args.softrerun
    if not rerun:
        import os
        if os.path.exists(f'experiments/ada_results/compress_ada_nobin/{index_to_run}.csv'):
            print(f'Results for index {index_to_run} already exist. Potentially Skipping...')
            if softrerun:
                # read as csv, check if skipped_node_count is zero for all rows
                df_existing = pd.read_csv(f'experiments/ada_results/compress_ada_nobin/{index_to_run}.csv')
                if (df_existing['num_skipped_trees'] > 0).any():
                    print('Some skipped_node_count > 0, rerunning as per soft rerun flag.')
                else:
                    print("skipping because soft rerun, but no skipped trees.")
                    exit(0)
            else:
                exit(0)
    datasets = list(CONFIGS.keys())
    dataset = datasets[index_to_run % len(datasets)]
    idx = index_to_run // len(datasets)
    depth = DEPTHS[idx % len(DEPTHS)]
    idx = idx // len(DEPTHS)
    tree_type = TREE_TYPES[idx % len(TREE_TYPES)]
    print(f'Running experiment for dataset: {dataset}')
    results_df = experiment(dataset=dataset, depth=depth, tree_type=tree_type)
    results_df.to_csv(f'experiments/ada_results/compress_ada_nobin/{index_to_run}.csv', index=False)