from sklearn.metrics import accuracy_score

from preprocessor import *
from train import *
from tuner import *
from utils import *


def main():
    args = cmd_arg_parse()
    if args['step'] == 'train':
        if args['prep'] == 'yes' or not os.path.isfile('preproc_data/train_data.csv'):
            print('Reading raw training data...')
            raw_df = pandas.read_csv('raw_data/responses.csv')

            ''' PREPROC '''
            print('\n*** Prerocessing ***')
            proc_df = fill_missing_values(raw_df)  # fill missing
            proc_df = proc_df.apply(pandas.to_numeric, errors="ignore")  # convert to numerics
            proc_df = normalize(proc_df)

            ''' SPLIT '''
            print('\n*** Splitting data ***')
            print('\ttrain - 60%, dev - 20%, test - 20%')
            dev_df, test_df, train_df = trn_dev_tst_split(proc_df)

            ''' FEATURE SELECTION '''
            dev_df, test_df, train_df = feature_select(dev_df, test_df, train_df)

            save_preproc_data(train_df, dev_df, test_df)

        print('Reading preprocessed train data...')
        train_df = pandas.read_csv('preproc_data/train_data.csv')
        train_df, y_train = separate_class(train_df)
        dev_df = pandas.read_csv('preproc_data/dev_data.csv')
        dev_df, y_dev = separate_class(dev_df)

        ''' TUNING '''
        tune_clf(args, dev_df, train_df, y_dev, y_train)

        ''' TRAINING '''
        print('\n*** Training ***')
        cols = read_features('pickles/features.pkl')
        clf = train_clf(args, cols, train_df, y_train)

        ''' PREDICTION '''
        y_pred_trn = clf.predict(train_df)
        y_pred_dev = clf.predict(dev_df)

        '''EVALUATION'''
        print('\n*** Train and Dev results ***')
        print('\ttrain data accuracy: {:.2f}%'.format(accuracy_score(y_train, y_pred_trn) * 100))
        print('\tdev data accuracy: {:.2f}%'.format(accuracy_score(y_dev, y_pred_dev) * 100))

        ''' DEV MISCLASSIFICATIONS '''
        dev_misclf_df = get_dev_misclfs(y_dev, y_pred_dev, dev_df)
        dev_misclf_df.head().to_csv('misc/dev_misclfs.csv')


    else:

        '''TESTING'''
        print('\n*** Testing ***')
        print('\treading test data...')
        test_df = pandas.read_csv('preproc_data/test_data.csv')
        test_df, y_test = separate_class(test_df)
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




if __name__ == '__main__':
    main()
