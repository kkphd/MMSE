from Step_2_MMSE_EDA import v1_impaired, v1_intact
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request
from joblib import dump, load
import numpy as np
from sklearn.preprocessing import RobustScaler


def production_model():
    impaired_upsampled = resample(v1_impaired, n_samples=124, random_state=42, replace=True)
    upsampled = pd.concat([v1_intact, impaired_upsampled])
    X = upsampled[['MF_01', 'eTIV', 'nWBV']]
    y = upsampled['MMSE_Group_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_scaled = RobustScaler().fit_transform(X_train)
    logreg2 = LogisticRegression(solver='lbfgs', max_iter=10000)
    logreg2_model = logreg2.fit(X_train_scaled, y_train)
    return logreg2_model

application = Flask(__name__)
model = production_model()

# Recall that features are listed in the following order: [MF_01, CDR, eTIV, nWBV]
@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    print(f'Features going in: {features}')
    features_array = [np.array(features)]
    scaled_features_array = RobustScaler().transform(features_array)
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