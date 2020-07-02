from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler


application = Flask(__name__)


# Load the model.
model = pickle.load(open('desired_model.pkl', 'rb'))

# Recall that features are listed in the following order: [MF_01, CDR, eTIV, nWBV]
@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    print(f'Features going in: {features}')
    features_array = [np.array(features)]
    scaled_features_array = RobustScaler().fit_transform(features_array)
    prediction = model.predict(scaled_features_array)
    print(prediction)
    probability = print(model.predict_proba(scaled_features_array))
    print(probability)

    if prediction == 0:
        prediction_text = "Intact"
    elif prediction == 1:
        prediction_text = "Impaired"
    else:
        prediction_text = ""

    predict_sentence = f'{prediction_text}'

    return render_template('index.html', prediction_text=predict_sentence)


if __name__ == '__main__':
    application.run()