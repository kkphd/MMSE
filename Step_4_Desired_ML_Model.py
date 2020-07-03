# Predicting Cognitive Impairment by Kiran K.
# www.github.com/kkphd/mmse


from Step_2_MMSE_EDA import v1_impaired, v1_intact
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import pickle


# Model 2 appears to be the best model, therefore we will save it for the web application.
scaler = RobustScaler()
impaired_upsampled = resample(v1_impaired, n_samples=124, random_state=42, replace=True)
upsampled = pd.concat([v1_intact, impaired_upsampled])
X = upsampled[['MF_01', 'CDR', 'eTIV', 'nWBV']]
y = upsampled['MMSE_Group_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
logreg2 = LogisticRegression(solver='lbfgs', max_iter=10000)
logreg2_model = logreg2.fit(X_train_scaled, y_train)

pickle.dump(scaler, open('transformer.pkl', 'wb'))
pickle.dump(logreg2_model, open('model.pkl', 'wb'))
