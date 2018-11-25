from os import path
from fastai.text import *
from fastai.structured import *
from fastai.column_data import *
import pandas as pd # to manipulate data frames
import numpy as np # to work with matrix
import json # to convert json in df
from pandas.io.json import json_normalize # to normalize the json fi
from datetime import datetime
from sklearn import preprocessing

# variables
# json columns
json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
PATH=Path('data/kg-google/')
p=0.07
#columns to drop
to_drop = ['customDimensions', 'hits',
       'socialEngagementType', 'visitId', 'visitNumber',
        'device.browserSize', 'device.browserVersion',
       'device.deviceCategory', 'device.flashVersion', 'device.isMobile',
       'device.language', 'device.mobileDeviceBranding',
       'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName',
       'device.mobileDeviceModel', 'device.mobileInputSelector',
       'device.screenColors', 'device.screenResolution',
       'geoNetwork.cityId',
       'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.metro',
       'geoNetwork.networkDomain', 'geoNetwork.networkLocation',
        'totals.bounces',
       'totals.sessionQualityDim',
       'totals.totalTransactionRevenue',
       'trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.criteriaParameters',
       'trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
       'trafficSource.isTrueDirect', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source'
          ]
cat_vars = ['channelGrouping', 'fullVisitorId',
       'device.browser', 'device.operatingSystem',
       'device.operatingSystemVersion', 'geoNetwork.city',
       'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.region',
       'geoNetwork.subContinent','_weekday','_day','_month','_year','_visitHour']
contin_vars = ['totals.hits', 'totals.newVisits',
       'totals.pageviews', 'totals.timeOnSite',
       'totals.transactions', 'totals.visits']

dep = 'totals.transactionRevenue'



# functions
def json_read(filename,columns):
    df = pd.read_csv(PATH / filename,
                     converters={column: json.loads for column in columns},
                     dtype={'fullVisitorId': 'str'},
                     #                      skiprows=lambda i : i> 0 and random.random() > p
                     nrows=2000
                     )

    for column in json_columns:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f'{column}.{subcolumn}' for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    #     print(f'Loaded {os,path.basename(data_frame)}. Shape: {df.shape}')
    return df


# This function is to extract date features
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")  # setting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday  # extracting week day
    df["_day"] = df['date'].dt.day  # extracting day
    df["_month"] = df['date'].dt.month  # extracting day
    df["_year"] = df['date'].dt.year  # extracting day
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)

    return df  # returning the df after the transformations

def NumericalColumns(df):    # fillna numeric feature
    df['totals.transactions'].fillna(1, inplace=True)
    df['totals.pageviews'].fillna(1, inplace=True) #filling NA's with 1
    df['totals.newVisits'].fillna(0, inplace=True) #filling NA's with 0
#     df['totals.bounces'].fillna(0, inplace=True)   #filling NA's with 0
#     df['trafficSource.isTrueDirect'].fillna(False, inplace=True) # filling boolean with False
#     df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True) # filling boolean with True
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0.0).astype(float) #filling NA with zero
    df['totals.pageviews'] = df['totals.pageviews'].astype(int) # setting numerical column as integer
    df['totals.newVisits'] = df['totals.newVisits'].astype(int) # setting numerical column as integer
#     df['totals.bounces'] = df['totals.bounces'].astype(int)  # setting numerical column as integer
    df["totals.hits"] = df["totals.hits"].astype(float) # setting numerical to float
    df['totals.visits'] = df['totals.visits'].astype(int) # seting as int

    return df #return the transformed dataframe

def Normalizing(df):
    # Use MinMaxScaler to normalize the column
    df["totals.hits"] =  (df['totals.hits'] - min(df['totals.hits'])) / (max(df['totals.hits'])  - min(df['totals.hits']))
    # normalizing the transaction Revenue
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].apply(lambda x: np.log1p(x))
    # return the modified df
    return df


# def missing_values(data):
#     total = data.isnull().sum().sort_values(ascending=False)  # getting the sum of null values and ordering
#     percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(
#         ascending=False)  # getting the percent and order of null
#     df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # Concatenating the total and percent
#     print("Total columns at least one Values: ")
#     print(df[~(df['Total'] == 0)])  # Returning values of nulls different of 0
#
#     print("\n Total of Sales % of Total: ", round((df_train[df_train['totals.transactionRevenue'] != np.nan][
#                                                        'totals.transactionRevenue'].count() / len(
#         df_train['totals.transactionRevenue']) * 100), 4))
#
#     return

def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

# Load files
df_train = json_read('train_v2.csv',json_columns)
df_test = json_read('test_v2.csv',json_columns)

#drop columns
df_train.drop(to_drop, axis=1, inplace=True)
df_test.drop(to_drop, axis=1, inplace=True)

#add date fields
df_train = date_process(df_train)
df_test = date_process(df_test)

df_train = NumericalColumns(df_train)
df_test = NumericalColumns(df_test)

df_train = Normalizing(df_train)
df_test = Normalizing(df_test)

df_train = df_train[cat_vars+contin_vars+[dep, 'date']].copy()
df_test[dep] = 0.0
df_test = df_test[cat_vars+contin_vars+[dep, 'date']].copy()
for v in cat_vars: df_train[v] = df_train[v].astype('category').cat.as_ordered()
apply_cats(df_test, df_train)

for v in contin_vars:
    df_train[v] = df_train[v].fillna(0).astype('float32')
    df_test[v] = df_test[v].fillna(0).astype('float32')

samp_size = len(df_train)
df_samp = df_train.set_index("date")

df, y, nas, mapper = proc_df(df_samp, 'totals.transactionRevenue', do_scale=True)
yl = np.log(y)
df_test = df_test.set_index("date")

df_test, _, nas, mapper = proc_df(df_test, dep, do_scale=True,
                                  mapper=mapper, na_dict=nas)

train_ratio = 0.75
# train_ratio = 0.9
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))

md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,
                                       test_df=df_test)
cat_sz = [(c, len(df_samp[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(100, (c+1)//2)) for _,c in cat_sz]

m = md.get_learner(emb_szs,len(df.columns)-len(cat_vars),0.04,1,[1000,500],[0.001,0.01])
# m.summary()
lr=1e-2
m.lr_find()