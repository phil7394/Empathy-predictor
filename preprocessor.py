import os

import pandas
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import PredefinedSplit
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.impute import SimpleImputer
import numpy as np


def fill_missing_values(df):
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    filled_data = imp.fit_transform(df)
    filled_df = pandas.DataFrame(filled_data, index=df.index, columns=df.columns)
    return filled_df


def binarize_class(column):
    binarizer = preprocessing.Binarizer(threshold=3)
    return binarizer.transform(column.values.reshape(-1, 1)).reshape(-1)


# TODO refactor
def plot_corr_matrix(df, nr_c, targ):
    corr = df.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, targ)[targ].index
    cm = np.corrcoef(df[cols].values.T)

    plt.figure(figsize=(nr_c / 1.5, nr_c / 1.5))
    sbn.set(font_scale=0.5)
    sbn.heatmap(cm, linewidths=1.5, annot=True, square=True,
                fmt='.2f', annot_kws={'size': 10},
                yticklabels=cols.values, xticklabels=cols.values
                )
    plt.show()


# def one_hot_encode(df):
#     enc = preprocessing.OneHotEncoder()
#     enc.fit(df)
#     onehotlabels = enc.transform(df).toarray()
#     return onehotlabels

def main():
    raw_df = pandas.read_csv('train_data/responses.csv')

    ''' PREPROC '''
    proc_df = fill_missing_values(raw_df)  # fill missing
    proc_df = proc_df.apply(pandas.to_numeric, errors="ignore")  # convert to numerics
    # proc_df.describe(include='all').to_csv("describe.csv")
    # with open('unique_vals', 'w') as f:
    #     for key, val in proc_df.items():
    #         f.write(key + ':' + str(val.unique()) + '\n')
    #         print(key, proc_df[key].dtype)
    # print(proc_df.shape)
    # proc_df = pandas.get_dummies(proc_df)  # discretize categoricals
    # print(disc_df.columns)
    # enc = one_hot_encode(proc_df)

    ''' SPLIT '''
    dev_df, test_df, train_df = trn_dev_tst_split(proc_df)

    ''' FEATURE SELECTION '''

    # plot_corr_matrix(train_df, 10, 'Empathy')

    # sbn.set()
    cols = ['Empathy', 'Life struggles', 'Compassion to animals', 'Judgment calls', 'Children', 'Weight', 'Romantic',
            'Fake', 'Latino', 'Fantasy/Fairy tales']
    # sbn.pairplot(train_df[cols], kind='reg', diag_kind='kde')
    # plt.show()

    ''' DATA WRANGLING'''
    dev_df, test_df, train_df, y_dev, y_test, y_train = data_wrangle(cols, dev_df, test_df, train_df)

    # print(train_df.shape, dev_df.shape, test_df.shape)

    '''TRAINING'''
    # clf = train_dummy(train_df, y_train)

    clf = train_decision_tree(train_df, y_train)

    y_pred_trn = clf.predict(train_df)

    ''' VALIDATION'''
    y_pred_dev = clf.predict(dev_df)

    ''' TUNING '''
    dt_parameters = {'max_depth': np.linspace(1, 9, 9, endpoint=True)}

    hyperparam_tune(clf, dt_parameters, dev_df, train_df, y_dev, y_train)

    '''TESTING'''
    y_pred_tst = clf.predict(test_df)
    '''METRICS'''
    print('Train: ', accuracy_score(y_train, y_pred_trn))
    print('Dev: ', accuracy_score(y_dev, y_pred_dev))
    print('Test: ', accuracy_score(y_test, y_pred_tst))
    # print(classification_report(y_train, y_pred_trn))


def trn_dev_tst_split(proc_df):
    if not os.path.isfile('train_data.csv'):
        print('splitting data...')
        train_df, dev_df, test_df = np.split(proc_df.sample(frac=1), [int(.6 * len(proc_df)), int(.8 * len(proc_df))])
        train_df.to_csv('train_data.csv', index=False)
        dev_df.to_csv('dev_data.csv', index=False)
        test_df.to_csv('test_data.csv', index=False)
    else:
        train_df = pandas.read_csv('train_data.csv')
        dev_df = pandas.read_csv('dev_data.csv')
        test_df = pandas.read_csv('test_data.csv')
    return dev_df, test_df, train_df


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


def train_decision_tree(train_df, y_train):
    clf = DecisionTreeClassifier(max_depth=7, random_state=0)
    print(clf)
    clf.fit(train_df, y_train)
    return clf


def train_dummy(train_df, y_train):
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    print(clf)
    clf.fit(train_df, y_train)
    return clf


def data_wrangle(cols, dev_df, test_df, train_df):
    train_df['Empathy'] = binarize_class(train_df['Empathy'])  # binarize class
    train_df = pandas.get_dummies(train_df)  # discretize categoricals
    y_train = train_df['Empathy']
    train_df.drop('Empathy', inplace=True, axis=1)
    train_df = train_df[cols[1:]]
    dev_df['Empathy'] = binarize_class(dev_df['Empathy'])  # binarize class
    dev_df = pandas.get_dummies(dev_df)  # discretize categoricals
    y_dev = dev_df['Empathy']
    dev_df.drop('Empathy', inplace=True, axis=1)
    dev_df = dev_df[cols[1:]]
    test_df['Empathy'] = binarize_class(test_df['Empathy'])  # binarize class
    test_df = pandas.get_dummies(test_df)  # discretize categoricals
    y_test = test_df['Empathy']
    test_df.drop('Empathy', inplace=True, axis=1)
    test_df = test_df[cols[1:]]
    return dev_df, test_df, train_df, y_dev, y_test, y_train


if __name__ == '__main__':
    main()
