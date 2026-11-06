import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from pystreed import STreeDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from preprocessing.data import load_data
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from gosdt import ThresholdGuessBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import time

class ThresholdTuningWrapper(ThresholdGuessBinarizer):
    '''
    Wrapper for ThresholdGuessBinarizer so that we can use
    grid search over hyperparameters without combining large numbers
    of estimators and large estimator depth simultaneously (which otherwise 
    hampers grid search runtime)
    '''
    def __init__(self, estimator_depth_tuple = (50, 3), random_state=None):
        self.estimator_depth_tuple = estimator_depth_tuple
        n_estimators, max_depth = self.estimator_depth_tuple
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

datasets = ['compas_complete.csv',
'wine_quality.csv',
'wisconsin.csv',
'coupon_full.csv',
'fico_complete.csv',
'netherlands.csv',
'spiral.csv',
'tic-tac-toe.csv',
'iris_virginica.csv',
'iris_versicolor.csv',
'iris_setosa.csv','broward_general_2y.csv']

methods = ['RandomForest', 'AdaBoost', 'GreedyTree',  'AdaBoost_3', 'RandomForest_5', 'OptimalTree', 'Binned_AdaBoost_3', 'Binned_RandomForest_5', 'OptimalTreeUnbinned']

def experiment(dataset, method): 
    # binarize?
    X, y = load_data(dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = kf.split(X)
    result_rows = []
    for fold, (train_index, test_index) in enumerate(folds):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if method == 'RandomForest':
            # tune a random forest on the training set using grid search and evaluate on the test set
            rf = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 1000],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_rf = grid_search.best_estimator_
            y_pred = best_rf.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'RandomForest', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_})
        elif method == 'AdaBoost':
            # tune adaboost boosting classifier
            ada = AdaBoostClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 1000],
                'learning_rate': [0.01, 0.1, 1],
                'estimator': [None, DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=None)],
            }
            grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_ada = grid_search.best_estimator_
            y_pred = best_ada.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'AdaBoost', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_})
        elif method == 'GreedyTree':
            # tune a simple decision tree classifier
            dt = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_dt = grid_search.best_estimator_
            y_pred = best_dt.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'GreedyTree', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_})
        elif method == 'OptimalTree':
            # tune a streed tree; make a pipeline that first does the threshold guessing binarization and then fits a STreeDClassifier, and tune the parameters of both steps using grid search
            estimator = Pipeline([
                ('binarizer', ThresholdTuningWrapper(random_state=42)),
                ('classifier', STreeDClassifier("cost-complex-accuracy", time_limit=900))
            ])
            estimator.score = lambda X, y: accuracy_score(y, estimator.predict(X))
            param_grid = {
                'binarizer__estimator_depth_tuple': [(40, 1),(40, 2),(40, 3),(100, 1),(100, 2),(100, 3),(200, 1),(200, 2)],
                'classifier__max_depth': [3, 4, 5],
                'classifier__cost_complexity': [0.005, 0.01, 0.05],
            }
            grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            # double the time_limit of the best estimator and refit on the entire training set
            best_estimator.named_steps['classifier'].time_limit = best_estimator.named_steps['classifier'].time_limit * 2
            start = time.process_time()
            best_estimator.fit(X_train, y_train)
            end = time.process_time()
            y_pred = best_estimator.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'OptimalTree', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_, 'training_time': end - start})
        elif method == 'OptimalTreeUnbinned':
            estimator = STreeDClassifier("cost-complex-accuracy", time_limit=900)
            estimator.score = lambda X, y: accuracy_score(y, estimator.predict(X))
            param_grid = {
                'max_depth': [3, 4, 5],
                'cost_complexity': [0.005, 0.01, 0.05],
            }
            grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            # double the time_limit of the best estimator and refit on the entire training set
            best_estimator.time_limit = best_estimator.time_limit * 2
            start = time.process_time()
            best_estimator.fit(X_train, y_train)
            end = time.process_time()
            y_pred = best_estimator.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'OptimalTreeUnbinned', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_, 'training_time': end - start})
        elif method == 'Binned_AdaBoost_3':
            # tune adaboost boosting classifier, but cap at depth 3 and n_estimators <=100 so that compression is possible. 
            ada = AdaBoostClassifier(random_state=42)
            estimator = Pipeline([
                ('binarizer', ThresholdTuningWrapper(random_state=42)),
                ('classifier', ada)
            ])
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.01, 0.1, 1],
                'classifier__estimator': [DecisionTreeClassifier(max_depth=3)],
            }
            grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_ada = grid_search.best_estimator_
            y_pred = best_ada.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'Binned_AdaBoost_3', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_})
        elif method == 'Binned_RandomForest_5':
            # required utils.rf_binarize, which is not part of this release
            raise NotImplementedError('Binned_RandomForest_5 is not supported')
        elif method == 'RandomForest_5':
            # tune a random forest on the training set using grid search, but cap at depth 5 and n_estimators <=100 so that compression is possible. 
            rf = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 1000],
                'max_depth': [5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_rf = grid_search.best_estimator_
            y_pred = best_rf.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'RandomForest_5', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_})
        elif method == 'AdaBoost_3':
            # tune adaboost boosting classifier, but cap at depth 3 and n_estimators <=100 so that compression is possible. 
            ada = AdaBoostClassifier(random_state=42)
            # param_grid = {
            #     'n_estimators': [50, 100],
            #     'learning_rate': [0.01, 0.1, 1],
            #     'estimator': [DecisionTreeClassifier(max_depth=3)],
            # }
            param_grid = {
                'n_estimators': [100],
                'learning_rate': [0.01, 0.1, 1],
                'estimator': [DecisionTreeClassifier(max_depth=3)],
            }
            grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring = 'roc_auc')
            grid_search.fit(X_train, y_train)
            best_ada = grid_search.best_estimator_
            y_pred = best_ada.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred > 0.5)
            result_rows.append({'fold': fold, 'dataset': dataset, 'model': 'AdaBoost_3', 'accuracy': acc, 'auc': roc_auc_score(y_test, y_pred), 'best_params': grid_search.best_params_})
        
    return pd.DataFrame(result_rows)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='Index of the configuration to run')
    parser.add_argument('--rerun', action='store_true', help='Whether to rerun even if results file exists')
    args = parser.parse_args()
    index_to_run = args.index
    rerun = args.rerun
    if not rerun:
        import os
        if os.path.exists(f'experiments/tuning/{index_to_run}.csv'):
            print(f'Results for index {index_to_run} already exist. Potentially Skipping...')
            exit(0)
    dataset = datasets[index_to_run % len(datasets)]
    idx = index_to_run // len(datasets)
    method = methods[idx % len(methods)]
    print(f'Running experiment for dataset: {dataset}')
    results_df = experiment(dataset=dataset, method=method)
    results_df.to_csv(f'experiments/tuning/{index_to_run}.csv', index=False)