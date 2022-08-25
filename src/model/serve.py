from flask import Flask, render_template, request
from src.CONSTANTS import FEATURE_COLUMNS, PROCESSED_DATA_MEAN_PKL, PROCESSED_DATA_STD_PKL

import numpy as np
import pandas as pd
import pickle
import os
import sys
import logging

sys.path.append("models")
sys.path.append("src/model")

curr_file_path = "src/model"
models_dirpath = 'models/'
template_dir = os.path.join(os.getcwd(), 'src/templates')

app = Flask(__name__, template_folder=template_dir)
model = pickle.load(open(os.getcwd() + '/' + models_dirpath + 'lr_model.pkl', 'rb'))
processed_data_mean = pickle.load(open(models_dirpath + PROCESSED_DATA_MEAN_PKL, 'rb'))
processed_data_std = pickle.load(open(models_dirpath + PROCESSED_DATA_STD_PKL, 'rb'))


@app.route('/')
def index():
    logging.info('Rendering index template!')
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    req = pd.DataFrame(request.form.to_dict(), index=[0], dtype='float')[FEATURE_COLUMNS]
    logging.info('Input request=[{}]'.format(req))
    # feature scaling
    req = (req - processed_data_mean) / processed_data_std
    pred = model.predict(req)
    logging.info('Predicted price:[{}]'.format(pred))
    return render_template('index.html',
                           data=int(pred))


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
