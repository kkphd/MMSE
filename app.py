from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


app = Flask(__name__)


# Load the model.
model = pickle.load(open('model.pkl', 'rb'))


# Recall that features are listed in the following order: MF_01, CDR, eTIV, nWBV, ASF
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    print(f'Features going in: {features}')
    features_array = [np.array(features)]
    prediction = model.predict(features_array)

    output = prediction[0]

    if output == 0:
        prediction_text = "Intact"
    elif output == 1:
        prediction_text = "Impaired"
    else:
        prediction_text = ""

    predict_sentence = f'{prediction_text}'

    return render_template('index.html', prediction_text=predict_sentence)


if __name__ == '__main__':
    app.run()