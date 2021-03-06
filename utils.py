import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sbn
import graphviz
from sklearn.tree import export_graphviz


def get_dev_misclfs(y_dev, y_pred_dev, dev_df):
    y_df = pandas.DataFrame({'y_dev': y_dev[y_dev != y_pred_dev]})
    misclf_df = dev_df[y_dev != y_pred_dev]
    misclf_df = misclf_df.reset_index(drop=True)
    y_df = y_df.reset_index(drop=True)
    dev_misclf_df = misclf_df.join(y_df)
    return dev_misclf_df


def plot_corr_matrix(nr_c, cm, cols):
    print('\tplotting top 9 correlated features...')
    plt.figure(figsize=(nr_c, nr_c))
    sbn.set(font_scale=1)
    sbn.heatmap(cm, linewidths=1.5, annot=True, square=True,
                fmt='.2f', annot_kws={'size': 10},
                yticklabels=cols.values, xticklabels=cols.values
                )
    plt.show()


def draw_tree(clf, cols):
    dot_data = export_graphviz(clf, out_file=None, feature_names=cols[1:],
                               class_names=['Empathy:0.0', 'Empathy:1.0'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph


def feature_correlation(df, nr_c, targ):
    print('\tgenerating feature correlation matrix...')
    corr = df.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, targ)[targ].index
    cm = np.corrcoef(df[cols].values.T)
    return cm, cols


def trn_dev_tst_split(proc_df):
    train_df, dev_df, test_df = np.split(proc_df.sample(frac=1, random_state=1234),
                                         [int(.6 * len(proc_df)), int(.8 * len(proc_df))])
    return dev_df, test_df, train_df


def separate_class(df):
    y = df['Empathy']
    df.drop('Empathy', inplace=True, axis=1)
    return df, y


def save_features(feature_list):
    with open('pickles/features.pkl', 'w') as f:
        feature_list = map(lambda x: x + '\n', feature_list)
        f.writelines(feature_list)


def read_features(features_file):
    with open(features_file, 'r') as f:
        feature_list = f.read().splitlines()
    return feature_list


def cmd_arg_parse():
    parser = argparse.ArgumentParser(description='Young People Empathy Predictor')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-s', '--step', help='train/test step', choices=['train', 'test'], required=True)
    required.add_argument('-m', '--model',
                          help='model to use: dmc=DummyClassifer, svc=SVCClassifier, dtc=DecisionTreeClassifier, rfc=FandomForestClassifier',
                          choices=['dmc', 'svc', 'dtc', 'rfc'], required=True)
    optional.add_argument('-p', '--prep',
                          help='do data prep steps: preprocessing, splitting, feature selection (ignored when used with test step option)',
                          choices=['yes', 'no'], default='no')
    parser._action_groups.append(optional)
    args = vars(parser.parse_args())
    return args
