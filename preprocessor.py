import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


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
    y = df['Empathy']
    df.drop('Empathy', inplace=True, axis=1)
    df = df[cols[1:]]
    return df, y
