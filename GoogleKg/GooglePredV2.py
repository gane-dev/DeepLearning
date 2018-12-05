from os import path
from fastai.structured import *
from fastai.column_data import *
import pandas as pd  # to manipulate data frames
import numpy as np  # to work with matrix
from datetime import datetime
from sklearn import preprocessing
import logging

import gc
from pandas.core.common import SettingWithCopyWarning
import warnings


# This function is to extract date features
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")  # setting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday  # extracting week day
    df["_day"] = df['date'].dt.day  # extracting day
    df["_month"] = df['date'].dt.month  # extracting day
    df["_year"] = df['date'].dt.year  # extracting day
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)

    return df  # returning the df after the transformations


def NumericalColumns(df):  # fillna numeric feature
    df['totals.pageviews'].fillna(1, inplace=True)  # filling NA's with 1
    df['totals.newVisits'].fillna(0, inplace=True)  # filling NA's with 0

    df['totals.bounces'].fillna(0, inplace=True)  # filling NA's with 0
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0).astype(float)  # filling NA with zero
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"] +1
    df['totals.pageviews'] = df['totals.pageviews'].astype(int)  # setting numerical column as integer
    df['totals.newVisits'] = df['totals.newVisits'].astype(int)  # setting numerical column as integer
    df["totals.hits"] = df["totals.hits"].astype(float)  # setting numerical to float
    df['trafficSource.isTrueDirect'].fillna(False, inplace=True)  # filling boolean with False
    return df  # return the transformed dataframe


def Normalizing(df):
    # Use MinMaxScaler to normalize the column
    # normalizing the transaction Revenue
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].apply(lambda x: np.log1p(x))
    return df


def inv_y(a): return np.exp(a)


def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred)) / targ
    return math.sqrt((pct_var ** 2).mean())


def main():
    PATH = Path('../data/kg-google/')
    train_file_name = f'{PATH}\extracted_fields_train.csv'
    test_file_name = f'{PATH}\extracted_fields_test.csv'
    columns_to_read = ['channelGrouping', 'date', 'fullVisitorId', 'totals.hits','visitNumber',
                       'visitStartTime', 'device.browser', 'device.deviceCategory',
                       'device.isMobile', 'device.operatingSystem', 'geoNetwork.city',
                       'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.region',
                       'geoNetwork.subContinent', 'totals.bounces', 'totals.hits',
                       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue',
                       'trafficSource.isTrueDirect',
                       'trafficSource.keyword',
                       'trafficSource.medium', 'trafficSource.source'
                       ]

    cat_vars = [
        # 'channelGrouping', 'device.deviceCategory', 'device.isMobile',
        #         'device.browser', 'device.operatingSystem', 'geoNetwork.city',
        #         'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.region', 'trafficSource.isTrueDirect',
        #         'trafficSource.keyword', 'trafficSource.medium', 'trafficSource.source',
        #         'geoNetwork.subContinent',                '_weekday', '_day', '_month', '_year',
        '_visitHour']

    # cat_vars = ['device.isMobile',
    #             'geoNetwork.city',
    #             'geoNetwork.continent', 'geoNetwork.country', 'trafficSource.isTrueDirect',
    #             'trafficSource.source',
    #              '_weekday', '_day', '_month','_visitHour']

    contin_vars = ['totals.hits', 'totals.newVisits', 'totals.pageviews','visitNumber','totals.bounces']
    # contin_vars = ['totals.hits', 'totals.pageviews', 'visitNumber']
    dep = 'totals.transactionRevenue'
    train_ratio = 0.75
    csv_fn = f'{PATH}/tmp/subv2.csv'
    csv_submission = f'{PATH}/tmp/submission.csv'

    logger.info('Read training file')
    df_train = pd.read_csv(train_file_name, dtype={'fullVisitorId': 'str'},
                           usecols=columns_to_read, low_memory=False
                           )

    #remove all transaction revenue zero
    # df_train = df_train.dropna(subset=['totals.transactionRevenue'])
    logger.info('Train size' + str(len(df_train)))
    logger.info('Read test file')
    df_test = pd.read_csv(test_file_name, dtype={'fullVisitorId': 'str'}, usecols=columns_to_read, low_memory=False)

    logger.info('Add date related columns')
    df_train = date_process(df_train)
    df_test = date_process(df_test)

    logger.info('Fix numerical columns')
    df_train = NumericalColumns(df_train)
    df_test = NumericalColumns(df_test)

    logger.info('Normalize transaction revenue')
    df_train = Normalizing(df_train)
    df_test = Normalizing(df_test)

    logger.info('Add category and continuous')

    df_train = df_train[cat_vars + contin_vars + [dep, 'date']].copy()
    df_test[dep] = 0.0
    df_test = df_test[cat_vars + contin_vars + [dep, 'date', 'fullVisitorId']].copy()

    logger.info('Convert category')
    for v in cat_vars: df_train[v] = df_train[v].astype('category').cat.as_ordered()
    apply_cats(df_test, df_train)

    for v in contin_vars:
        df_train[v] = df_train[v].fillna(0).astype('float32')
        df_test[v] = df_test[v].fillna(0).astype('float32')

    # Sample size for validation

    samp_size = len(df_train)
    df_samp = df_train.set_index("date")
    df_test = df_test.set_index("date")

    logger.info('Process dataframe')
    df, y, nas, mapper = proc_df(df_samp, 'totals.transactionRevenue', do_scale=True)
    yl = np.log(y)
    df_test1, _, nas, mapper = proc_df(df_test, dep, do_scale=True, skip_flds=['fullVisitorId'], mapper=mapper,
                                       na_dict=nas)

    # train_ratio = 0.9
    train_size = int(samp_size * train_ratio);

    logger.info('Get validation size')
    val_idx = list(range(train_size, len(df)))
    max_log_y = np.max(yl)
    y_range = (0, max_log_y * 1.2)

    logger.info('Get Model')
    md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,
                                           test_df=df_test1)

    logger.info('Categorize size')
    cat_sz = [(c, len(df_samp[c].cat.categories) + 1) for c in cat_vars]

    logger.info('Embedding size')
    emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in cat_sz]

    m = md.get_learner(emb_szs, len(df.columns) - len(cat_vars),
                       0.04, 1, [1000, 500], [0.001, 0.01], y_range=y_range)
    lr = 1e-3

    # m.fit(lr, 5, metrics=[exp_rmspe], cycle_len=1)
    # m.fit(lr, 3, metrics=[exp_rmspe])
    # # prediction
    #
    # x, y = m.predict_with_targs()
    #
    # logger.info('RMSE' + str(exp_rmspe(x, y)))
    #
    # pred_test = m.predict(True)
    # pred_test = np.exp(pred_test)
    #
    # df_test[dep] = pred_test
    # df_test[['fullVisitorId', dep]].to_csv(csv_fn, index=False)
    # df_try_unique = pd.read_csv(csv_fn, low_memory=False)
    # df_try_unique = df_try_unique.groupby('fullVisitorId').mean()
    # df_try_unique = df_try_unique.reset_index()
    # df_try_unique[['fullVisitorId', 'totals.transactionRevenue']].to_csv(csv_submission, index=False)

   #Random forest regressor
    # from sklearn.ensemble import RandomForestRegressor
    # ((val, trn), (y_val, y_trn)) = split_by_idx(val_idx, df.values, yl)
    # m = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2,
    #                           n_jobs=-1, oob_score=True)
    # m.fit(trn, y_trn);
    # predictions = m.predict(df_test1)
    # logger.info(m.score(trn, y_trn), m.score(val, y_val), m.oob_score_, exp_rmspe(preds, y_val))


    # LGBM

    # import lightgbm as lgb
    # ((val, trn), (y_val, y_trn)) = split_by_idx(val_idx, df.values, yl)
    # lgb_train = lgb.Dataset(trn, y_trn)
    # lgb_val = lgb.Dataset(val, y_val)
    # params = {
    #     'objective': 'binary',
    #     'boosting': 'gbdt',
    #     'learning_rate': 0.2,
    #     'verbose': 0,
    #     'num_leaves': 100,
    #     'bagging_fraction': 0.95,
    #     'bagging_freq': 1,
    #     'bagging_seed': 1,
    #     'feature_fraction': 0.9,
    #     'feature_fraction_seed': 1,
    #     'max_bin': 256,
    #     'num_rounds': 100,
    #     'metric': 'auc'
    # }
    #
    # lgbm_model = lgb.train(params, train_set=lgb_train, valid_sets=lgb_val, verbose_eval=5)
    # predictions = lgbm_model.predict(df_test1)


    # df_test[dep] =predictions
    # df_test[['fullVisitorId', dep]].to_csv(csv_fn, index=False)
    # df_try_unique = pd.read_csv(csv_fn, low_memory=False)
    # df_try_unique = df_try_unique.groupby('fullVisitorId').mean()
    # df_try_unique = df_try_unique.reset_index()
    # df_try_unique[['fullVisitorId', 'totals.transactionRevenue']].to_csv(csv_submission, index=False)
    # logger.info('Train done')
    import lightgbm as lgb
    ((val, trn), (y_val, y_trn)) = split_by_idx(val_idx, df.values, yl)
    lgb_train = lgb.Dataset(trn, y_trn)
    lgb_val = lgb.Dataset(val, y_val)
    params = {
        #'objective': 'binary',
        'objective': 'regression',
        'boosting': 'gbdt',
        'learning_rate': 0.2,
        'verbose': 0,
        'num_leaves': 100,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'num_rounds': 100,
        'metric': 'l2_root',
        'num_iterations': 100,
        'num_threads':0,
        'device_type':'cpu' #gpu
    }

    lgbm_model = lgb.train(params, train_set=lgb_train, valid_sets=lgb_val, verbose_eval=5)
    predictions = lgbm_model.predict(df_test1)
    df_test[dep] =predictions-1
    df_test[['fullVisitorId', dep]].to_csv(csv_fn, index=False)
    df_try_unique = pd.read_csv(csv_fn, low_memory=False)
    df_try_unique = df_try_unique.groupby('fullVisitorId').mean()
    df_try_unique = df_try_unique.reset_index()
    df_try_unique[['fullVisitorId', 'totals.transactionRevenue']].to_csv(csv_submission, index=False)
    logger.info('Train done')



def get_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('logging.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)

    return logger_


if __name__ == '__main__':
    logger = get_logger()
    try:
        warnings.simplefilter('error', SettingWithCopyWarning)
        gc.enable()
        logger.info('Process started')

        main()

    except Exception as err:
        logger.exception('Exception occured')
        raise

#https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst