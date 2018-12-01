from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit


def hyperparam_tune(clf, parameters, dev_df, train_df, y_dev, y_train):
    test_fold = [-1] * len(train_df) + [1] * len(dev_df)
    tune_df = train_df.append(dev_df, ignore_index=False)
    y_tune = y_train.append(y_dev, ignore_index=False)
    ps = PredefinedSplit(test_fold)
    gs_clf = GridSearchCV(clf, parameters, cv=ps, n_jobs=-1)
    gs_clf = gs_clf.fit(tune_df, y_tune)
    print('Best tuned score:', gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))