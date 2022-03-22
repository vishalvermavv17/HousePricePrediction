from flask import Flask, render_template, request
from src.CONSTANTS import FEATURE_COLUMNS, PROCESSED_DATA_MEAN_PKL, PROCESSED_DATA_STD_PKL

import numpy as np
import pandas as pd
import pickle
import os

models_dirpath = '../../models/'
template_dir = os.path.join(os.getcwd(), '../templates')

app = Flask(__name__, template_folder=template_dir)
model = pickle.load(open(models_dirpath + 'lr_model.pkl', 'rb'))
processed_data_mean = pickle.load(open(models_dirpath + PROCESSED_DATA_MEAN_PKL, 'rb'))
processed_data_std = pickle.load(open(models_dirpath + PROCESSED_DATA_STD_PKL, 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    req = pd.DataFrame(request.form.to_dict(), index=[0], dtype='float')[FEATURE_COLUMNS]
    # feature scaling
    req = (req - processed_data_mean) / processed_data_std
    print(req.info())
    pred = model.predict(req)
    return render_template('index.html',
                           data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
