# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("C:/Users/LENOVO/personal_projects/trade_prediction_new/sales_train.csv")
test_data = pd.read_csv("C:/Users/LENOVO/personal_projects/trade_prediction_new/test.csv")

train_data.head()
test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.info()
train_data['date'] = pd.to_datetime(train_data['date'], format= '%d.%m.%Y')
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
monthly_data = train_data.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':'sum'}).reset_index()
monthly_data.rename(columns={'item_cnt_day' : 'item_cnt_month'}, inplace = True)
monthly_data.head()
from sklearn.preprocessing import StandardScaler
def create_lag_features(df, lags, col):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby(['shop_id', 'item_id'])[col].shift(lag)
    return df
monthly_data = create_lag_features(monthly_data, [1,2,3], 'item_cnt_month')
monthly_data.head()
numerical_cols = ['item_cnt_month'] + [f'item_cnt_month_lag_{lag}' for lag in [1,2,3]]
scaler = StandardScaler()
monthly_data[numerical_cols] = scaler.fit_transform(monthly_data[numerical_cols])
monthly_data.head()
for col in monthly_data.select_dtypes(include=['float']).columns:
    monthly_data[col] = pd.to_numeric(monthly_data[col], downcast='float')

for col in monthly_data.select_dtypes(include=['int']).columns:
    monthly_data[col] = pd.to_numeric(monthly_data[col], downcast='integer')

print(monthly_data.info(memory_usage='deep'))
monthly_data['month'] = monthly_data['date_block_num'] % 12 + 1
# Ensure date_block_num is a larger integer type before calculations
monthly_data['date_block_num'] = monthly_data['date_block_num'].astype('int16')
monthly_data['year'] = (monthly_data['date_block_num'] // 12) + 2013


shop_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
item_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

shop_encoded = shop_encoder.fit_transform(monthly_data[['shop_id']])
item_encoded = item_encoder.fit_transform(monthly_data[['item_id']])

numerical_features = monthly_data.drop(columns=['shop_id', 'item_id']).values
combined_features = sparse.hstack((shop_encoded, item_encoded, numerical_features))

combined_df = pd.DataFrame.sparse.from_spmatrix(combined_features)
combined_df.head()
monthly_data
monthly_data = monthly_data.iloc[:100]
monthly_data['rolling_mean_3'] = monthly_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
monthly_data['rolling_std_3'] = monthly_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(lambda x: x.rolling(window=3, min_periods=1).std())
monthly_data['rolling_mean_6'] = monthly_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
monthly_data['rolling_std_6'] = monthly_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(lambda x: x.rolling(window=6, min_periods=1).std())
monthly_data.head()
selected_features = ['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'rolling_std_6', 'item_cnt_month_lag_1']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
X = monthly_data[selected_features]
y = monthly_data['item_cnt_month']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
categorical_cols = ['shop_id', 'item_id']
numerical_cols = ['date_block_num', 'month', 'year', 'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'rolling_std_6', 'item_cnt_month_lag_1']
my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# Define the custom RMSE log function
def rmse_log(y_true, y_pred):
    # To avoid issues with log(0), replace zero values with a very small positive number
    y_true = np.where(y_true <= 0, np.finfo(float).eps, y_true)
    y_pred = np.where(y_pred <= 0, np.finfo(float).eps, y_pred)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# Create a custom scorer
rmse_log_scorer = make_scorer(rmse_log, greater_is_better=False)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestRegressor(n_estimators=50, random_state=0))])

rf_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring=rmse_log_scorer)
print("Random Forest RMSE Log: ", -rf_scores.mean())
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

final_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestRegressor(n_estimators=50, random_state=0))])

final_model.fit(X_train, y_train)
missing_cols = set(X_train.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0 

test_X = test_data[X_train.columns].copy()

print("Test Data Head:")
test_X.head()
test_preds = final_model.predict(test_X)

submission = pd.DataFrame({
    'ID': test_data['ID'],
    'item_cnt_month': test_preds
})

submission['item_cnt_month'] = submission['item_cnt_month'].clip(0, 20)

submission.to_csv('submission.csv', index=False)
submission.head()
import pickle

# Assuming 'model' is your trained model
with open('tradeprediction.pkl', 'wb') as f:
    pickle.dump(final_model, f)