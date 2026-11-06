from preprocessing.data import prepare_binarized_data_kfolds, load_data
from models.tree_classifier_wrapper import DNFTree, TreeClassifierWrapper, ConservativeTree
from models.treefarms_wrapper import TreeFarmsWrapper, construct_tree_rset
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import time

from utils.dontcare_utils import ViableRanges
"""
Returns None if T(x) = T_prime(x) for all x in S, and otherwise returns an x in S such that T(x) != T_prime(x)
Assumes T and T_prime are decision trees in the same feature space, represented as json objects in our standard representational form

# Test case A: assert true for T and T_prime identical

"""
def findMismatch(S: ViableRanges, T: dict, T_prime: dict):
    if 'prediction' in T:
        if 'prediction' in T_prime:
            if T['prediction'] != T_prime['prediction']:
                return S.generate_sample() #todo replace with sampling from S
            else:
                return None
        return findMismatch(S, T_prime, T) # note the swapped order; T is a leaf so we iterate over T' the rest of the DFS
    else:
        query = T['name']
        # check if known
        known_val = S.deduce_threshold_state(query)
        if known_val is not False: # if true is possible: 
            cached_range_before_update = S.update(T['name'], True)
            mismatch = findMismatch(S, T['true'], T_prime)
            S.restore_old_range(*cached_range_before_update)
            if mismatch is not None:
                return mismatch
        if known_val is not True: # if false is possible:
            cached_range_before_update = S.update(T['name'], False)
            mismatch = findMismatch(S, T['false'], T_prime)
            S.restore_old_range(*cached_range_before_update)
            if mismatch is not None:
                return mismatch
        return None
    
def __hash__(self):
    return self.hash_val

def WitnessDE(T, T_prime, feature_space=[]):
    S = ViableRanges(feature_space=feature_space)
    return findMismatch(S, T, T_prime)

'''
Maps from collection to a compressed set of trees, each tree is a representative of an equivalence class of trees in the original collection, 
where two trees are in the same equivalence class if they are identical on all inputs in the original feature space.
'''
def compress_rset(collection: TreeFarmsWrapper, feature_space = []):
    X_diff = []
    predmap = {}
    for tree in collection: #not sure best mapping to use; could also do indices, etc; and not sure how to wrap.
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

TF_CONFIG = {
    "depth_budget": 4,
    "rashomon_bound_adder": 0.02,
    "rashomon_ignore_trivial_extensions": True,
    "regularization": 0.01,
    "verbose": False
}

DEFAULT_CONFIG = {"num_quantiles": 0, "gosdt_guesses": True, "keep_missingness_after_binarization": False}
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

NAN_HANDLING_METHOD = {'ours': DNFTree, 'path-based': TreeClassifierWrapper, 'used-features': ConservativeTree}

RSET_NAN_HANDLING_METHOD = {'ours': 'DNF', 'path-based': 'Default', 'used-features': 'Conservative'}

def get_largest_vbl_complexity(model_list):
    # Implementation for getting the largest VBL complexity (split-node count, distinct-split count)
    max_nodes, max_unique = 0, 0
    for model in model_list or []:
        max_nodes = max(max_nodes, len(model.used_vars))
        max_unique = max(max_unique, len(set(model.used_vars)))
    return max_nodes, max_unique

def experiment(dataset='fico_full.csv', tree_type='DNF', balance=False, hashing_type=None, n_samples_for_hash=0, tf_config=TF_CONFIG):
    X, y = load_data(dataset, balance=balance)
    binarization_config = CONFIGS[dataset]

    folds, binarization_info = prepare_binarized_data_kfolds(X, y, **binarization_config)
    result_rows = []
    for i in range(5):
        result_row = {'dataset': dataset, 'tree_type': tree_type}
        result_row['balanced'] = balance
        result_row['lambda'] = tf_config['regularization']
        result_row['depth'] = tf_config['depth_budget']
        result_row['eps'] = tf_config['rashomon_bound_adder']
        result_row['rashomon_ignore_trivial_extensions'] = tf_config['rashomon_ignore_trivial_extensions']
        result_row['n_samples_for_hash'] = n_samples_for_hash
        result_row['hashing_type'] = hashing_type
        result_row['fold'] = i
        X_train, y_train, X_test, y_test = folds[i]

        # rename columns to be integers
        # X_train.columns = range(X_train.shape[1])
        # X_test.columns = range(X_test.shape[1])

        # Train model:

        misstree = construct_tree_rset(X_train.reset_index(drop=True),
                                       y_train.reset_index(drop=True),
                                       tf_config=tf_config,
                                       tree_type=tree_type,
                                       hashing_type=hashing_type,
                                       n_samples_for_hash=n_samples_for_hash)
        result_row['time_to_eval_model_list'] = misstree.model_list_time

        result_row['rset_size_unique'] = misstree.get_unique_tree_count()
        result_row['time_to_eval_model_set'] = misstree.model_set_time
        if tree_type == 'DNF':
            result_row['largest_vbl_complexity'], result_row['largest_unique_splits'] = \
                get_largest_vbl_complexity(misstree.model_list)

        actual_hash_hits, possible_hits = misstree.count_hash_hits()
        result_row['actual_hash_hits'] = actual_hash_hits
        result_row['possible_hash_hits'] = possible_hits

        result_row['rset_size'] = misstree.get_tree_count()
        # try efficient compression
        if tree_type in ['ContinuousMissTree', 'EfficientContinuousMissTree']:
            start = time.time()
            compressed_trees = compress_rset(misstree.model_list, X_train.columns)
            end = time.time()
            result_row['time_new_compression'] = end - start
            assert len(compressed_trees) == misstree.get_unique_tree_count(), "Compressed tree count does not match unique tree count, there may be a bug in the compression function"
        # alt_config = tf_config.copy()
        # alt_config["rashomon_ignore_trivial_extensions"] = False

        # misstree = construct_tree_rset(X_train.reset_index(drop=True),
        #                                y_train.reset_index(drop=True),
        #                                tf_config=alt_config,
        #                                tree_type=tree_type,
        #                                hashing_type=hashing_type,
        #                                n_samples_for_hash=n_samples_for_hash)
        # result_row['time_to_eval_model_list_no_trivial_extensions'] = misstree.model_list_time

        # result_row['rset_size_no_trivial_extensions'] = misstree.get_tree_count()
        # result_row['time_to_eval_model_set_no_trivial_extensions'] = misstree.model_set_time

        # actual_hash_hits, possible_hits = misstree.count_hash_hits()
        # result_row['actual_hash_hits_no_trivial_extensions'] = actual_hash_hits
        # result_row['possible_hash_hits_no_trivial_extensions'] = possible_hits

        # result_row['rset_size_unique_no_trivial_extensions'] = misstree.get_unique_tree_count()

        result_rows.append(result_row.copy())
    return result_rows


def experiment_loop(index_to_run, rerun):
    rows = []
    index = 0
    for reg in [0.01, 0.005]:
        for eps in [0.02, 0.05]:
            for trivial_extension_setting in [True, False]:
                balance = True
            # for balance in [True]:
                # for tree_type in ['MissTree', 'DNF', 'ContinuousMissTree', 'EfficientContinuousMissTree']:
                for tree_type in ['DNF', 'EfficientContinuousMissTree']:
                    # for n_samples_for_hash in [1, 10, 100]:
                    for n_samples_for_hash in [1]:
                        for depth in [4, 5]:
                            for dataset in CONFIGS.keys():
                                # for hashing_type in [None, "training_data", "binary_counting"]:
                                for hashing_type in [None]:
                                    index += 1
                                    if hashing_type is None and n_samples_for_hash > 1:
                                        continue
                                    if index == index_to_run:
                                        if not rerun:
                                            try:
                                                pd.read_csv(f'experiments/hashing_results/{index_to_run}.csv')
                                                return  # if successfully read data without error, don't rerun
                                            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                                                pass
                                        print('running', dataset, tree_type, n_samples_for_hash, balance, hashing_type, reg, eps, depth)
                                        TF_CONFIG['depth_budget'] = depth
                                        TF_CONFIG['regularization'] = reg
                                        TF_CONFIG['rashomon_bound_adder'] = eps
                                        TF_CONFIG['rashomon_ignore_trivial_extensions'] = trivial_extension_setting
                                        rows += experiment(
                                            dataset=dataset, 
                                            tree_type=tree_type, 
                                            balance=balance,
                                            hashing_type=hashing_type,
                                            n_samples_for_hash=n_samples_for_hash,
                                            tf_config=TF_CONFIG
                                        )
                                        df = pd.DataFrame(rows)
                                        df.to_csv(f'experiments/hashing_results/{index_to_run}.csv', index=False)


import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--index', type=int, default=0)
    argparser.add_argument('--rerun', action='store_true', default=False)
    args = argparser.parse_args()
    index_to_run = args.index
    rerun = args.rerun
    experiment_loop(index_to_run, rerun)
