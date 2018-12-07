import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import seaborn as sbn
import matplotlib.pyplot as plt

from utils import feature_correlation, plot_corr_matrix, save_features


def fill_missing_values(df):
    print('\tfilling missing values...')
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    filled_data = imp.fit_transform(df)
    filled_df = pandas.DataFrame(filled_data, index=df.index, columns=df.columns)
    return filled_df


def binarize_class(column):
    print('\tbinarizing class...')
    binarizer = preprocessing.Binarizer(threshold=3)
    return binarizer.transform(column.values.reshape(-1, 1)).reshape(-1)


def normalize(proc_df):
    print('\tnormalizing column...')
    weight_min = proc_df['Weight'].min()
    weight_max = proc_df['Weight'].max()
    proc_df['Weight'] = 5 * (proc_df['Weight'] - weight_min) / (weight_max - weight_min)
    return proc_df


def data_wrangle(cols, df):
    df['Empathy'] = binarize_class(df['Empathy'])  # binarize class
    print('\tdiscretizing categoricals...')
    df = pandas.get_dummies(df)  # discretize categoricals
    df = df[cols]
    print('\t' + str(df.shape))
    return df


def feature_select(dev_df, test_df, train_df):
    print('\n*** Feature selection ***')
    cm, cols = feature_correlation(train_df, 10, 'Empathy')
    plot_corr_matrix(10, cm, cols)
    cols = list(cols)
    save_features(cols)
    # sbn.set(font_scale=0.5)
    # sbn.pairplot(train_df[cols], kind='reg', diag_kind='kde')
    # plt.show()
    print('\n*** Data wrangling ***')
    train_df = data_wrangle(cols, train_df)
    dev_df = data_wrangle(cols, dev_df)
    test_df = data_wrangle(cols, test_df)
    return dev_df, test_df, train_df


def save_preproc_data(train_df, dev_df, test_df):
    train_df.to_csv('preproc_data/train_data.csv', index=False)
    dev_df.to_csv('preproc_data/dev_data.csv', index=False)
    test_df.to_csv('preproc_data/test_data.csv', index=False)
