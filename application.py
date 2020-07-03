import pandas as pd
from flask import Flask, render_template, request
import pickle


application = Flask(__name__)
logreg2_model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('transformer.pkl', 'rb'))


# Recall that features are listed in the following order: [MF_01, CDR, eTIV, nWBV]
@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    features = request.form.values()
    features = list(features)
    print(features)
    df_new = pd.DataFrame(columns=['MF_01', 'CDR', 'eTIV', 'nWBV'])
    new_features = [int(features[0]), float(features[1]), int(features[2]), float(features[3])]
    df_new.loc[0] = new_features

    print(f'Features going in: {features}')
    scaled_features_array = scaler.transform(df_new)
    prediction = logreg2_model.predict(scaled_features_array)

    print(prediction)
    probability = print(logreg2_model.predict_proba(scaled_features_array))
    print(probability)

    if prediction == 0:
        prediction_text = "Intact"
    elif prediction == 1:
        prediction_text = "Impaired"
    else:
        prediction_text = "Error"

    predict_sentence = f'{prediction_text}'

    return render_template('index.html', prediction_text=predict_sentence)


if __name__ == '__main__':
    application.run()