from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def load_data(path):
    df = pd.read_csv(path)

    return df


def split_feature_label(Xy):
    y = Xy['SalePrice']
    X = Xy.drop('SalePrice', axis=1, inplace=False)

    return X, y


def transform(X, y=None):
    X.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)

    X.fillna(X.mean(), inplace=True)

    if y is not None:
        cond1 = X['GrLivArea'] > 4000
        cond2 = y < 500000
        outlier_indices = X[cond1 & cond2].index
        X.drop(outlier_indices, axis=0, inplace=True)
        y.drop(outlier_indices, axis=0, inplace=True)

    skewed_numeric_features = [
        'MiscVal',
        'PoolArea',
        'LotArea',
        '3SsnPorch',
        'LowQualFinSF',
        'KitchenAbvGr',
        'BsmtFinSF2',
        'ScreenPorch',
        'BsmtHalfBath',
        'EnclosedPorch',
        'MasVnrArea',
        'OpenPorchSF',
        'LotFrontage',
        'WoodDeckSF',
        'MSSubClass',
        'GrLivArea',
    ]
    X[skewed_numeric_features] = X[skewed_numeric_features].apply(np.log1p)

    return X, y


def train_model(X, y):
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    ridge = Ridge(alpha=8)
    lasso = Lasso(alpha=0.001)
    xgb_regressor = XGBRegressor(n_estimators=1000, learning_rate=0.05, colsample_bytree=0.5, subsample=0.8)
    lgbm_regressor = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=4, subsample=0.6, colsample_bytree=0.4, reg_lambda=10)
    estimators = [
        ('ridge', ridge),
        ('lasso', lasso),
        ('xgb_regressor', xgb_regressor),
        ('lgbm_regressor', lgbm_regressor)
    ]
    final_estimator = Lasso(alpha=0.0005)
    stacking_regressor = StackingRegressor(estimators, final_estimator, cv=5)

    pipeline = Pipeline([
        ('one_hot_encoder', one_hot_encoder),
        ('stacking_regressor', stacking_regressor)
    ])

    pipeline.fit(X, y)

    return pipeline


def parse_args():
    parser = ArgumentParser(
        description='Generate the submission file for Kaggle House Prices competition.')
    parser.add_argument(
        '--train', type=Path, default='train.csv',
        help='path of train.csv downloaded from the competition')
    parser.add_argument(
        '--test', type=Path, default='test.csv',
        help='path of test.csv downloaded from the competition')

    return parser.parse_args()


def main(args):
    Xy_train = load_data(args.train)
    X_train, y_train = split_feature_label(Xy_train)
    X_train, y_train = transform(X_train, y_train)
    model = train_model(X_train, y_train)

    X_test = load_data(args.test)
    ids = X_test['Id']
    X_test, _ = transform(X_test)
    y_test = model.predict(X_test)

    submission = {
        'Id': ids,
        'SalePrice': y_test
    }
    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
