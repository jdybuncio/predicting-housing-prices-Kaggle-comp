import pandas as pd
import numpy as np


def cleaning_ames_df(df):
    # create cols
    df['Age'] = df.YrSold - df.YearBuilt
    df['Age_Remodel'] = df.YrSold - df.YearRemodAdd
    df['NonBedroomsAbvGr'] = df.TotRmsAbvGrd - df.BedroomAbvGr
    df['Total_Area'] = df.LotFrontage+ df.LotArea
    df['Total_House_Sqft'] = df.TotalBsmtSF + df.GrLivArea
    df = pd.get_dummies(df,drop_first=True)


    important_cols = ['OverallQual',
    'Total_House_Sqft',
    'GrLivArea',
    'GarageCars',
    'GarageArea',
    'TotalBsmtSF',
    '1stFlrSF',
    'NonBedroomsAbvGr',
    'FullBath',
    'TotRmsAbvGrd',
    'Age',
    'YearBuilt',
    'Age_Remodel',
    'YearRemodAdd',
    'GarageYrBlt',
    'MasVnrArea',
    'Fireplaces',
    'BsmtFinSF1',
    'LotFrontage']

    important_cols_cal = ['ExterQual_TA',
    'KitchenQual_TA',
    'Foundation_PConc',
    'ExterQual_Gd',
    'BsmtQual_TA',
    'BsmtFinType1_GLQ',
    'GarageFinish_Unf',
    'Neighborhood_NridgHt',
    'MasVnrType_None',
    'SaleType_New',
    'GarageType_Detchd',
    'SaleCondition_Partial',
    'Foundation_CBlock',
    'FireplaceQu_Gd',
    'GarageType_Attchd',
    'MasVnrType_Stone',
    'Neighborhood_NoRidge',
    'KitchenQual_Gd',
    'HeatingQC_TA']

    total_cols = important_cols + important_cols_cal
    X = df[total_cols].fillna(0)
    # y = df.SalePrice

    return X