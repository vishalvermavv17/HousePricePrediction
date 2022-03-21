import os.path

from flask import Flask, render_template, request
import numpy as np
import pickle
import os

models_dirpath = '../../models/'
template_dir = os.path.join(os.getcwd(), '../templates')

app = Flask(__name__, template_folder=template_dir)
model = pickle.load(open(models_dirpath + 'lr_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['floors']
    val4 = request.form['yr_built']
    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = 2000
    # model.predict([arr])

    return render_template('index.html',
                           data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
