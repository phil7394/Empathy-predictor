from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def hyperparam_tune(clf, parameters, dev_df, train_df, y_dev, y_train):
    test_fold = [-1] * len(train_df) + [1] * len(dev_df)
    tune_df = train_df.append(dev_df, ignore_index=False)
    y_tune = y_train.append(y_dev, ignore_index=False)
    ps = PredefinedSplit(test_fold)
    gs_clf = GridSearchCV(clf, parameters, cv=ps, n_jobs=-1)
    gs_clf = gs_clf.fit(tune_df, y_tune)
    print('\tBest tuned score:', gs_clf.best_score_)
    print('\tHyperparameter values:')
    for param_name in sorted(parameters.keys()):
        print('\t\t{}: {}'.format(param_name, gs_clf.best_params_[param_name]))


def tune_clf(args, dev_df, train_df, y_dev, y_train):
    if args['model'] != 'dmc':
        if args['model'] == 'dtc':
            clf = DecisionTreeClassifier()
            parameters = {'max_depth': np.arange(1, 3, 1).tolist(), 'random_state': [0]}
        elif args['model'] == 'svc':
            clf = SVC()
            parameters = {'C': np.arange(0.001, 1.0, 0.1).tolist(),
                          'kernel': ['linear', 'poly', 'rbf'],
                          'degree': [0, 1, 2],
                          'gamma': np.arange(0.0, 1.1, 0.4).tolist(),
                          'coef0': np.arange(0.0, 1.1, 0.4).tolist(),
                          'tol': np.arange(0.001, 1.1, 0.4).tolist(),
                          'random_state': [0]
                          }
        else:
            clf = RandomForestClassifier()
            parameters = {'max_depth': np.arange(1, 7, 1).tolist(),
                          'max_features': ['auto', 'sqrt'],
                          'min_samples_leaf': np.arange(10, 50, 10).tolist(),
                          'min_samples_split': np.arange(2, 10, 1).tolist(),
                          'n_estimators': np.arange(1, 10, 1).tolist(),
                          'random_state': [0]
                          }
        print('\n*** Hyperparameter tuning ***')
        print('\n\tModel to tune...\n\t' + str(clf) + '\n')
        hyperparam_tune(clf, parameters, dev_df, train_df, y_dev, y_train)
