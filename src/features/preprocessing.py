from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.CONSTANTS import TARGET_LABEL, FEATURE_COLUMNS, PROCESSED_DATA_MEAN_PKL, PROCESSED_DATA_STD_PKL

import pandas as pd
import pickle
import os

cwd = os.getcwd()
raw_data_dirpath = '../../data/raw'
models_dirpath = '../../models/'
processed_data_dirpath = '../../data/processed'

# Load Housing Data into dataframe
df = pd.read_csv(os.path.join(cwd, raw_data_dirpath, 'train.csv'), index_col='Id')
df.info()

# Find Categorical and numerical feature columns
categorical_features = [col for col in df.columns if df[col].dtype == 'object']
numerical_features = list(set(df.columns) - set(categorical_features))

# Feature columns MiscFeature, Fence, PoolQC, FireplaceQul, Alley has >80% missing values.
# It'll be a good idea to drop those columns as they can add noise instead of adding value to the model.
feature_col_drop = ['MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley']
df.drop(columns=feature_col_drop, inplace=True)
print("Remaining columns: {}".format(df.columns))

# update numerical feature list
numerical_features = list(set(numerical_features) - set(feature_col_drop))

feature_columns = FEATURE_COLUMNS

# Imputing numerical feature missing values
it_imputer = IterativeImputer(max_iter=10)
processed_data = pd.DataFrame(it_imputer.fit_transform(df[feature_columns]),
                              columns=df[feature_columns].columns).set_index(df.index)
processed_data.info()

# store processed data mean and std for inference request feature scaling
pickle.dump(processed_data.mean(), open(models_dirpath + PROCESSED_DATA_MEAN_PKL, 'wb'))
pickle.dump(processed_data.std(), open(models_dirpath + PROCESSED_DATA_STD_PKL, 'wb'))

# Feature Scaling
processed_data = (processed_data - processed_data.mean()) / processed_data.std()

processed_data = processed_data.join(df[TARGET_LABEL])
processed_data.info()

# Store processed data
os.chdir(processed_data_dirpath)
processed_data.to_csv('processed_data.csv', index=False)

print("Processed data has been stored successfully at {}".format(os.path.join(cwd, processed_data_dirpath)))
