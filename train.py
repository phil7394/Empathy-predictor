from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_svc(train_df, y_train, **kwargs):
    print('\ttraining SVC...')
    # clf = SVC(C=0.001, kernel='poly', degree=2, gamma=0.4, coef0=0.4, tol=0.401, random_state=0)
    # clf = SVC(C=0.1, kernel='linear', random_state=0)
    clf = SVC(**kwargs)
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/svc.pkl')
    return clf


def train_random_forest(train_df, y_train, **kwargs):
    print('\ttraining RandomForestClassifier...')
    # clf = RandomForestClassifier(max_depth=10, n_estimators=4, random_state=0)
    # clf = RandomForestClassifier(max_depth=3, max_features='auto', min_samples_leaf=20, min_samples_split=2,
    #                              n_estimators=4, random_state=0)
    clf = RandomForestClassifier(**kwargs)
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/rf.pkl')
    return clf


def train_decision_tree(train_df, y_train, **kwargs):
    print('\ttraining DecisionTreeClassifier...')
    # clf = DecisionTreeClassifier(max_depth=1, random_state=0)  # max_depth=1
    clf = DecisionTreeClassifier(**kwargs)  # max_depth=1
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/dt.pkl')
    return clf


def train_dummy(train_df, y_train, **kwargs):
    print('\ttraining DummyClassifier...')
    # clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf = DummyClassifier(**kwargs)
    print('\t{}'.format(clf))
    clf.fit(train_df, y_train)
    joblib.dump(clf, 'pickles/dummy.pkl')
    return clf
