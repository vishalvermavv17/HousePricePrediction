from linearRegression import LinearRegressionUsingGD
from src.CONSTANTS import FEATURE_COLUMNS, TARGET_LABEL, TRAINED_MODEL_NAME

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle

cwd = os.getcwd()
models_dirpath = '../../models/'
processed_data_dirpath = '../../data/processed'

# Load processed data into dataframe
train_data = pd.read_csv(os.path.join(cwd, processed_data_dirpath, 'processed_data.csv'))
train_data.info()

print("feature columns={} and target label={} used for model training.".format(FEATURE_COLUMNS, TARGET_LABEL))
X = train_data[FEATURE_COLUMNS]
y = train_data[TARGET_LABEL]

linear_regressor = LinearRegressionUsingGD(0.01, 200)
linear_regressor = linear_regressor.fit(X, y)

# store linear_regression model
with open(models_dirpath + TRAINED_MODEL_NAME, 'wb') as output_file:
    pickle.dump(linear_regressor, output_file)

# plot error cost for each iteration to verify linear regression is working fine and reducing cost value in each
# iteration.
fig = plt.figure(figsize=(12, 8))
sns.lineplot(data=pd.DataFrame(linear_regressor.cost_, range(len(linear_regressor.cost_))))
plt.show()

# mean squared error
y_pred = linear_regressor.predict(X)
mse = np.sum((y_pred - y)**2)/ X.shape[0]

# root mean squared error
rmse = np.sqrt(mse)
print("MSE: [{}] and RMSE: [{}] for trained model".format(mse, rmse))
