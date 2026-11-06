import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.tree_to_dict import sklearn_tree_to_dict, unbinned_sklearn_tree_to_dict
from models.tree_classifier_wrapper import create_tree_classifier

DEFAULT_RF_CONFIG = {
    'n_estimators': 100,
    'random_state': 42
}

DEFAULT_BAGGING_CONFIG = DEFAULT_RF_CONFIG.copy()
DEFAULT_TREE_CONFIG = {
    'max_depth': 4,
    'random_state': 42
}

DEFAULT_ADABOOST_CONFIG = {
    'estimator': DecisionTreeClassifier(max_depth=3),
    'n_estimators': 50
}

class ForestWrapper:
    def __init__(self, model, tree_type = 'ContinuousMissTreeUnbinned', verbose=False, **kwargs):
        self.trees = []
        self.tree_type = tree_type
        self.verbose = verbose
        self.tree_wrapper_kwargs = kwargs
        if verbose: 
            print('Initialized with tree_type:', tree_type)
        self.model = model

    def fit(self, X, y, return_compress_time=False):
        self.trees = []
        self.model.fit(X, y)
        if y.ndim == 1: #y needs to be a dataframe for tree_classifier's training loss evaluation
            y = y.to_frame()
        if return_compress_time:
            start_time = time.perf_counter()
        for i, tree in enumerate(self.model.estimators_):
            if "Unbinned" in self.tree_type:
                dict_rep = unbinned_sklearn_tree_to_dict(tree.tree_)
            else:
                dict_rep = sklearn_tree_to_dict(tree.tree_)
            self.trees.append(create_tree_classifier(dict_rep, tree_type=self.tree_type, index = i, **self.tree_wrapper_kwargs))
        if return_compress_time:
            end_time = time.perf_counter()
            return end_time - start_time
    def votes(self, X) -> pd.DataFrame:
        predictions = pd.DataFrame(np.zeros((X.shape[0], len(self.trees))), dtype='Int64')
        for i, tree in enumerate(self.trees):
            predictions.iloc[:, i] = tree.predict(X)
        return predictions
    
    def predict(self, X, votes_threshold=0.5) -> pd.Series:
        '''
        Predict positive if greater than votes_threshold proportion
        of the constituent trees predict positive
        '''
        if len(self.trees) == 0:
            raise ValueError("Model not yet fit")
        else: 
            votes = self.votes(X)
            if self.verbose: 
                print('Predicting; num votes of NA values ...')
                print(votes.isna().values.sum())
            return votes.mean(axis=1) > votes_threshold

    def predict_default(self, X):
        return self.model.predict(X)

class RandomForestWrapper(ForestWrapper):
    def __init__(self, rf_config=DEFAULT_RF_CONFIG, tree_type = 'ContinuousMissTreeUnbinned', verbose=False, **kwargs):
        model = RandomForestClassifier(**rf_config)
        super().__init__(model, tree_type=tree_type, verbose=verbose, **kwargs)

class BaggingForestWrapper(ForestWrapper):
    def __init__(self, bagging_config = DEFAULT_BAGGING_CONFIG, tree_config = DEFAULT_TREE_CONFIG, tree_type = 'ContinuousMissTreeUnbinned', verbose = False, **kwargs):
        config = bagging_config.copy()
        config['estimator'] = DecisionTreeClassifier(**tree_config)
        model = BaggingClassifier(**config)
        super().__init__(model, tree_type=tree_type, verbose=verbose, **kwargs)


class AdaBoostWrapper(ForestWrapper):
    def __init__(self, adaboost_config=DEFAULT_ADABOOST_CONFIG, tree_type = 'MissTree', verbose=False, **kwargs):
        model = AdaBoostClassifier(**adaboost_config)
        super().__init__(model, tree_type=tree_type, verbose=verbose, **kwargs)

    def fit(self, X, y, return_compress_time=False):
        self.trees = []
        self.model.fit(X, y)
        if y.ndim == 1: #y needs to be a dataframe for tree_classifier's training loss evaluation
            y = y.to_frame()
        if return_compress_time:
            start_time = time.perf_counter()
        for tree in self.model.estimators_:
            if "Unbinned" in self.tree_type:
                dict_rep = unbinned_sklearn_tree_to_dict(tree.tree_)
            else:
                dict_rep = sklearn_tree_to_dict(tree.tree_)
            self.trees.append(create_tree_classifier(dict_rep, tree_type=self.tree_type, **self.tree_wrapper_kwargs))
        self.tree_weights = self.model.estimator_weights_
        if return_compress_time:
            end_time = time.perf_counter()
            return end_time - start_time

    def predict(self, X, votes_threshold=0.5) -> pd.Series:
        '''
        Predict positive if the weighted average of our tree
        predictions is greater than votes_threshold
        '''
        if len(self.trees) == 0:
            raise ValueError("Model not yet fit")
        else: 
            votes = self.votes(X)
            if self.verbose: 
                print('Predicting; num votes of NA values ...')
                print(votes.isna().values.sum())
            weighted_votes = votes * self.tree_weights.reshape(1, -1)
            return np.sum(weighted_votes, axis=1) / np.sum(self.tree_weights) > votes_threshold