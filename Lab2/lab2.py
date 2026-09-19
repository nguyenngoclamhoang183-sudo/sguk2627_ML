import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ==========================
# Load Data
# ==========================
train = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/test.csv')

print("Train Shape: ", train.shape)
print("Test Shape: ", test.shape)
# train has 81 columns (79 features + id and target SalePrice) and 1460 entries
# test has 80 columns (79 features + id) and 1459 entries

print(train.info())
print(test.info())

# ==========================
# Check for NULL values
# ==========================
# Train Data
print(train.isnull().sum())
sns.heatmap(train.isnull())

# Test Data
print(test.isnull().sum())
sns.heatmap(test.isnull())

print("Train Shape: ", train.shape)
print("Test Shape: ", test.shape)

# ==========================
# Handling NULL data
# ==========================
# 'Alley','PoolQC','Fence' and 'MiscFeature' columns have more than 70% null
# values in both train and test data. So, we will drop these columns.
# Also, we will drop 'Id' column. For non-categorical columns, we will handle
# null values by filling mean of the column.
# For categorical columns, we will handle null values by filling mode of the column.

# For Train Data
cat_col_train = ['FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType', 'BsmtQual',
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                  'GarageQual', 'GarageCond']

ncat_col_train = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']

for i in cat_col_train:
    train[i] = train[i].fillna(train[i].mode()[0])

for j in ncat_col_train:
    train[j] = train[j].fillna(train[j].mean())

# For Test Data
cat_col_test = ['FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType', 'BsmtQual',
                 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                 'GarageQual', 'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st',
                 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']

ncat_col_test = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                  'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                  'BsmtHalfBath', 'GarageCars', 'GarageArea']

for i in cat_col_test:
    test[i] = test[i].fillna(test[i].mode()[0])

for j in ncat_col_test:
    test[j] = test[j].fillna(test[j].mean())

# ==========================
# Drop Columns
# ==========================
to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']

for k in to_drop:
    train.drop([k], axis=1, inplace=True)
    test.drop([k], axis=1, inplace=True)

sns.heatmap(train.isnull())
sns.heatmap(test.isnull())

print("Train Shape: ", train.shape)
print("Test Shape: ", test.shape)
# Train Shape: (1460, 76)
# Test Shape: (1459, 75)

# It is observed that for some columns in train data few categories are not
# present but available in test data. So, we will concat test data to train
# data, then perform one hot encoding on all categorical columns.

# ==========================
# Concat train and test
# ==========================
final_df = pd.concat([train, test], axis=0)
print(final_df.shape)  # (2919, 76)

all_cat_col = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
               'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
               'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
               'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
               'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
               'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


def cat_onehot_encoding(multicol):
    df_final = final_df
    i = 0
    for fields in multicol:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)

        final_df.drop([fields], axis=1, inplace=True)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1

    df_final = pd.concat([final_df, df_final], axis=1)

    return df_final


final_df = cat_onehot_encoding(all_cat_col)

print(final_df.shape)  # (2919, 237)

# Remove duplicated columns (created during the one-hot-encoding concat above)
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
print(final_df.shape)  # (2919, 177)

# ==========================
# Split back into train/test
# ==========================
df_train = final_df.iloc[:1460, :]
df_test = final_df.iloc[1460:, :]

df_test.drop(['SalePrice'], axis=1, inplace=True)

print("Train Shape: ", df_train.shape)
print("Test Shape: ", df_test.shape)
# Train Shape: (1460, 177)
# Test Shape: (1459, 176)

# ==========================
# Training Data
# ==========================
x_train = df_train.drop(['SalePrice'], axis=1)
y_train = df_train['SalePrice']

# ==========================
# XGBoost
# ==========================
import xgboost

xgb_model = xgboost.XGBRegressor()
xgb_model.fit(x_train, y_train)

# Hyperparameter tuning
param = {
    'n_estimators': [100, 500, 900, 1100, 1500],
    'max_depth': [2, 3, 5, 10, 15],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'min_child_weight': [1, 2, 3, 4],
    'booster': ['gbtree', 'gblinear'],
    'base_score': [0.25, 0.5, 0.75, 1]
}

from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=xgb_model,
                                param_distributions=param,
                                cv=5, n_iter=50,
                                scoring='neg_mean_absolute_error', n_jobs=4,
                                verbose=5,
                                return_train_score=True,
                                random_state=42)
random_cv.fit(x_train, y_train)

print(random_cv.best_estimator_)
# Best found (example from the notebook):
# XGBRegressor(base_score=0.25, booster='gbtree', learning_rate=0.1,
#              max_depth=2, n_estimators=900, ...)

# Retrain using the best params found above.
# Note: newer xgboost versions (2.x/3.x) removed the old `gpu_id` argument
# (use `device='cpu'` or `device='cuda'` instead) and no longer need
# `validate_parameters`/`interaction_constraints`/`monotone_constraints`/
# `importance_type` to be passed explicitly, so those are dropped here.
xgb_model = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
                                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                                  device='cpu',
                                  learning_rate=0.1, max_delta_step=0, max_depth=2,
                                  min_child_weight=1,
                                  n_estimators=900, n_jobs=0, num_parallel_tree=1,
                                  objective='reg:squarederror', random_state=0, reg_alpha=0,
                                  reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                                  verbosity=None)

xgb_model.fit(x_train, y_train)

# ==========================
# Save Model
# ==========================
f = "xgb_model.pkl"
pickle.dump(xgb_model, open(f, 'wb'))

# ==========================
# Predictions
# ==========================
pred_xgb = xgb_model.predict(df_test)
print(pred_xgb.shape)  # (1459,)

# ==========================
# Submission
# ==========================
sub_df = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/sample_submission.csv')
sub_df['SalePrice'] = pred_xgb
sub_df.to_csv('sample_sub_xgb.csv', index=False)


# ==========================
# Decision Tree
# ==========================
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

# Predictions
pred_dt = dt_model.predict(df_test)
print(pred_dt.shape)  # (1459,)

# Submissions
sub_df = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/sample_submission.csv')
sub_df['SalePrice'] = pred_dt
sub_df.to_csv('sample_sub_dt.csv', index=False)


# ==========================
# Artificial Neural Network
# ==========================
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from keras import backend as k


def root_mean_squared_error(y_true, y_pred):
    return k.sqrt(k.mean(k.square(y_pred - y_true)))


# Model
nn_model = Sequential()

nn_model.add(Dense(50, kernel_initializer='he_uniform', activation='relu', input_dim=176))
nn_model.add(Dense(25, kernel_initializer='he_uniform', activation='relu'))
nn_model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
nn_model.add(Dense(1, kernel_initializer='he_uniform'))

nn_model.compile(loss=root_mean_squared_error, optimizer='Adamax')

nn_model.fit(x_train.values, y_train.values, validation_split=0.25, batch_size=10, epochs=1000)

nn_model.save('nn_model.h5')

# Predictions
pred_nn = nn_model.predict(df_test)
print(pred_nn.shape)  # (1459, 1)

# Submission
sub_df = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/sample_submission.csv')
sub_df['SalePrice'] = pred_nn
sub_df.to_csv('sample_sub_nn.csv', index=False)