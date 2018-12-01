import argparse

from sklearn.metrics import accuracy_score

from preprocessor import *
from train import *
from tuner import *
from utils import *


def main():
    args = cmd_arg_parse()
    if args['step'] == 'train':
        if args['prep'] == 'yes':
            print('Reading raw training data...')
            raw_df = pandas.read_csv('train_data/responses.csv')

            ''' PREPROC '''
            print('*** Prerocessing ***')
            proc_df = fill_missing_values(raw_df)  # fill missing
            proc_df = proc_df.apply(pandas.to_numeric, errors="ignore")  # convert to numerics
            proc_df = normalize(proc_df)
            # proc_df.describe(include='all').to_csv("misc/describe.csv")
            # with open('unique_vals', 'w') as f:
            #     for key, val in proc_df.items():
            #         f.write(key + ':' + str(val.unique()) + '\n')
            #         print(key, proc_df[key].dtype)
            # print(proc_df.shape)

            ''' SPLIT '''
            print('*** Splitting data ***')
            print('\ttrain - 60%, dev - 20%, test - 20%')
            dev_df, test_df, train_df = trn_dev_tst_split(proc_df, True)

            ''' FEATURE SELECTION '''
            print('*** Feature selection ***')
            cm, cols = feature_correlation(train_df, 10, 'Empathy')
            plot_corr_matrix(10, cm, cols)

            # sbn.set()
            cols = list(cols)
            save_features(cols)
            # sbn.pairplot(train_df[cols], kind='reg', diag_kind='kde')

        '''PRE-TUNE TRAINING'''

        cols = read_features('pickles/features.pkl')
        print('Reading preprocessed train data...')
        train_df = pandas.read_csv('preproc_data/train_data.csv')
        dev_df = pandas.read_csv('preproc_data/dev_data.csv')

        print('*** Data wrangling ***')
        train_df, y_train = data_wrangle(cols, train_df)
        dev_df, y_dev = data_wrangle(cols, dev_df)

        print('*** Pre-tune Training ***')
        if args['model'] == 'dmc':
            clf = train_dummy(train_df, y_train, strategy='most_frequent', random_state=0)
        elif args['model'] == 'dtc':
            clf = train_decision_tree(train_df, y_train, max_depth=10, random_state=0)
        elif args['model'] == 'svc':
            clf = train_svc(train_df, y_train, C=0.1, kernel='linear', random_state=0)
        else:
            clf = train_random_forest(train_df, y_train, max_depth=10, n_estimators=4, random_state=0)

        ''' TUNING '''
        if args['model'] != 'dmc':
            print('*** Hyperparameter tuning ***')
            if args['model'] == 'dtc':
                parameters = {'max_depth': np.linspace(1, 4, 1, endpoint=True), 'random_state': [0]}
            elif args['model'] == 'svc':
                parameters = {'C': np.arange(0.001, 1.0, 0.1).tolist(),
                              'kernel': ['linear', 'poly', 'rbf'],
                              'degree': [0, 1, 2],
                              'gamma': np.arange(0.0, 1.1, 0.4).tolist(),
                              'coef0': np.arange(0.0, 1.1, 0.4).tolist(),
                              'tol': np.arange(0.001, 1.1, 0.4).tolist(),
                              'random_state': [0]
                              }
            else:
                parameters = {'max_depth': np.arange(1, 10, 1).tolist(),
                              'max_features': ['auto', 'sqrt'],
                              'min_samples_leaf': [10, 20, 30, 50, 60, 70],
                              'min_samples_split': [2, 5, 10, 15],
                              'n_estimators': np.arange(2, 20, 1).tolist(),
                              'random_state': [0]
                              }

            hyperparam_tune(clf, parameters, dev_df, train_df, y_dev, y_train)

        '''POST-TUNE TRAINING '''
        print('*** Post-tune Training ***')
        if args['model'] == 'dmc':
            clf = train_dummy(train_df, y_train, strategy='most_frequent', random_state=0)
        elif args['model'] == 'dtc':
            clf = train_decision_tree(train_df, y_train, max_depth=2, random_state=0)
            graph = draw_tree(clf, cols)
            graph.render('misc/decision_graph')
        elif args['model'] == 'svc':
            clf = train_svc(train_df, y_train, C=0.001, kernel='poly', degree=2, gamma=0.4, coef0=0.4, tol=0.401,
                            random_state=0)
            plot_svc(clf, train_df)
        else:
            clf = train_random_forest(train_df, y_train, max_depth=3, max_features='auto', min_samples_leaf=20,
                                      min_samples_split=2,
                                      n_estimators=4, random_state=0)
        y_pred_trn = clf.predict(train_df)
        y_pred_dev = clf.predict(dev_df)

        '''EVALUATION'''
        print('*** Train and Dev results ***')
        print('\ttrain data accuracy: {:.2f}%'.format(accuracy_score(y_train, y_pred_trn) * 100))
        print('\tdev data accuracy: {:.2f}%'.format(accuracy_score(y_dev, y_pred_dev) * 100))

        ''' DEV MISCLASSIFICATIONS '''
        dev_misclf_df = get_dev_misclfs(y_dev, y_pred_dev, dev_df)
        dev_misclf_df.head().to_csv('misc/dev_misclfs.csv')


    else:

        '''TESTING'''
        print('*** Testing ***')

        cols = read_features('pickles/features.pkl')
        print('\treading test data...')
        test_df = pandas.read_csv('preproc_data/test_data.csv')
        test_df, y_test = data_wrangle(cols, test_df)
        print('\tloading saved model...')
        if args['model'] == 'dmc':
            clf = joblib.load('pickles/dummy.pkl')
        elif args['model'] == 'dtc':
            clf = joblib.load('pickles/dt.pkl')
        elif args['model'] == 'svc':
            clf = joblib.load('pickles/svc.pkl')
        else:
            clf = joblib.load('pickles/rf.pkl')
        print('\tpredicting on test data...')
        y_pred_tst = clf.predict(test_df)

        '''EVALUATION'''
        print('\ttest data accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_tst) * 100))


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


if __name__ == '__main__':
    main()
