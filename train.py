from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import draw_tree


def train_svc(train_df, y_train, **kwargs):
    print('\ttraining SVC...')
    clf = SVC(**kwargs)
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/svc.pkl')
    return clf


def train_random_forest(train_df, y_train, **kwargs):
    print('\ttraining RandomForestClassifier...')
    clf = RandomForestClassifier(**kwargs)
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/rf.pkl')
    return clf


def train_decision_tree(train_df, y_train, **kwargs):
    print('\ttraining DecisionTreeClassifier...')
    clf = DecisionTreeClassifier(**kwargs)
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/dt.pkl')
    return clf


def train_dummy(train_df, y_train, **kwargs):
    print('\ttraining DummyClassifier...')
    clf = DummyClassifier(**kwargs)
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/dummy.pkl')
    return clf


def train_clf(args, cols, train_df, y_train):
    if args['model'] == 'dmc':
        clf = train_dummy(train_df, y_train, strategy='most_frequent', random_state=0)
    elif args['model'] == 'dtc':
        clf = train_decision_tree(train_df, y_train, max_depth=2, random_state=0)
        graph = draw_tree(clf, cols)
        graph.render('misc/decision_graph')
    elif args['model'] == 'svc':
        clf = train_svc(train_df, y_train, C=0.001, kernel='poly', degree=2, gamma=0.4, coef0=0.4, tol=0.401,
                        random_state=0)
    else:
        clf = train_random_forest(train_df, y_train, max_depth=3, max_features='auto', min_samples_leaf=20,
                                  min_samples_split=2,
                                  n_estimators=4, random_state=0)
    return clf
