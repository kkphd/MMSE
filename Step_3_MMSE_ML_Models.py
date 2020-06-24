# Predicting Cognitive Impairment by Kiran K.
# www.github.com/kkphd/mmse


from Step_2_MMSE_EDA import v1_impaired, v1_intact
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from collections import defaultdict


# Predict dementia status using supervised learning classification. To enhance the Impaired subjects'
# signal (due to an imbalanced data set), I will up-sample this group by bootstrapping with replacement.
impaired_upsampled = resample(v1_impaired, n_samples=124, random_state=42, replace=True)


# Combine the unsampled Impaired group with the original Intact group.
upsampled = pd.concat([v1_intact, impaired_upsampled])
upsampled['MMSE_Group'].value_counts()


def plot_mmse_upsampled():
    v1_data = upsampled[['eTIV', 'nWBV', 'MMSE_Group', 'Sex']]
    sns.set(style='whitegrid', color_codes=True)
    fig4 = sns.pairplot(v1_data, hue='MMSE_Group')
    fig4.add_legend()
    fig4.fig.set_figheight(8)
    fig4.fig.set_figwidth(10)
    fig4.fig.suptitle('eTIV and nWBV by Cognitive Status', size='16')
    plt.subplots_adjust(top=0.95)

    fig5 = sns.pairplot(v1_data, hue='Sex')
    fig5.add_legend()
    plt.subplots_adjust(top=1.0)
    fig5.fig.set_figheight(8)
    fig5.fig.set_figwidth(10)
    fig5.fig.suptitle('eTIV and nWBV by Sex', size='16')
    plt.subplots_adjust(top=0.95)

plot_mmse_upsampled()


# Binary logistic regression will be used to predict MMSE-derived cognitive status.
def logreg_model1():
    X = upsampled[['MF_01', 'eTIV', 'nWBV', 'ASF']]
    y = upsampled['MMSE_Group_Status']

    # Split the data into training and testing data sets. By convention, 75% will be used for training and
    # the remaining 25% will be used for testing. Data are shuffled by default.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Preprocess the data. I used the Robust Scaling method since outliers skewed the feature variables.
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Fit the model to the data.
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    logreg_model = logreg.fit(X_train_scaled, y_train)
    logreg_model_r2 = logreg_model.score(X_test_scaled, y_test)
    y_pred = logreg_model.predict(X_test_scaled)

    # Create a confusion matrix.
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    precision1 = precision_score(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_logreg = auc(fpr, tpr)
    plt.figure(5)
    plt.title('ROC Curve of Logistic Regression and Decision Tree Models', size='16')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance', color='k')
    plt.plot(fpr, tpr, marker='.', label='Model 1 (AUC = %0.3f)' % auc_logreg)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.show()

    result1 = [1, 'Logistic Regression (without CDR)', logreg_model_r2, tpr, 1-fpr, auc_logreg, precision1]

    return logreg_model_r2, conf_matrix, class_report, auc_logreg, result1, precision1

logreg_model_r2, conf_matrix, class_report, auc_logreg, result1, precision1 = logreg_model1()


# Assess if the model improves after incorporating a test that includes participants' functional status (CDR).
def logreg_model2():
    X = upsampled[['MF_01', 'CDR', 'eTIV', 'nWBV', 'ASF']]
    y = upsampled['MMSE_Group_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    logreg2 = LogisticRegression(solver='lbfgs', max_iter=1000)
    logreg2_model = logreg2.fit(X_train_scaled, y_train)
    logreg2_model_r2 = logreg2_model.score(X_test_scaled, y_test)
    y2_pred = logreg2_model.predict(X_test_scaled)

    conf_matrix2 = confusion_matrix(y_test, y2_pred)
    class_report2 = classification_report(y_test, y2_pred)
    precision2 = precision_score(y_test, y2_pred)

    fpr2, tpr2, thresholds2 = roc_curve(y_test, y2_pred, pos_label=1)
    auc_logreg2 = auc(fpr2, tpr2)
    plt.figure(5)
    plt.plot(fpr2, tpr2, marker='.', label='Model 2 (AUC = %0.3f)' % auc_logreg2)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.show()

    result2 = [2, 'Logistic Regression (with CDR)', logreg2_model_r2, tpr2, 1 - fpr2, auc_logreg2, precision2]

    return logreg2_model_r2, conf_matrix2, class_report2, auc_logreg2, result2, precision2

logreg2_model_r2, conf_matrix2, class_report2, auc_logreg2, result2, precision2 = logreg_model2()


# Using a decision tree may provide additional insights about classification accuracy without and with CDR.
def dectree_model3():
    X = upsampled[['MF_01', 'eTIV', 'nWBV']]
    y = upsampled['MMSE_Group_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    dt = DecisionTreeClassifier(max_depth=2)
    dt1_model = dt.fit(X_train_scaled, y_train)
    dt1_model_r2 = dt1_model.score(X_test_scaled, y_test)
    y3_pred = dt1_model.predict(X_test_scaled)

    conf_matrix3 = confusion_matrix(y_test, y3_pred)
    class_report3 = classification_report(y_test, y3_pred)
    precision3 = precision_score(y_test, y3_pred)

    fpr3, tpr3, thresholds3 = roc_curve(y_test, y3_pred, pos_label=1)
    auc_dt3 = auc(fpr3, tpr3)
    plt.figure(5)
    plt.plot(fpr3, tpr3, marker='.', label='Model 3 (AUC = %0.3f)' % auc_dt3)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.show()

    dt1_feature_names = ['Sex', 'eTIV', 'nWBV']
    dt_class_names = ['Impaired', 'Intact'] # Needs to be in ascending numerical order
    export_graphviz(dt1_model, out_file='tree1.dot',feature_names=dt1_feature_names,
                    class_names=dt_class_names, filled=True)
    dt1_graph = pydotplus.graph_from_dot_file('tree1.dot')

    result3 = [3, 'Decision Tree (without CDR)', dt1_model_r2, tpr3, 1 - fpr3, auc_dt3, precision3]

    return dt1_model_r2, conf_matrix3, class_report3, dt1_graph, result3, precision3

dt1_model_r2, conf_matrix3, class_report3, dt1_graph, result3, precision3 = dectree_model3()


def dectree_model4():
    X = upsampled[['MF_01', 'CDR', 'eTIV', 'nWBV']]
    y = upsampled['MMSE_Group_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    dt = DecisionTreeClassifier(max_depth=2)
    dt2_model = dt.fit(X_train_scaled, y_train)
    dt2_model_r2 = dt2_model.score(X_test_scaled, y_test)
    y4_pred = dt2_model.predict(X_test_scaled)

    conf_matrix4 = confusion_matrix(y_test, y4_pred)
    class_report4 = classification_report(y_test, y4_pred)
    precision4 = precision_score(y_test, y4_pred)

    fpr4, tpr4, thresholds4 = roc_curve(y_test, y4_pred, pos_label=1)
    auc_dt4 = auc(fpr4, tpr4)
    plt.figure(5)
    plt.plot(fpr4, tpr4, marker='.', label='Model 4 (AUC = %0.3f)' % auc_dt4)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.show()

    dt2_feature_names = ['Sex', 'CDR', 'eTIV', 'nWBV']
    dt_class_names = ['Intact', 'Impaired']
    export_graphviz(dt2_model, out_file='tree2.dot', feature_names=dt2_feature_names,
                    class_names=dt_class_names, filled=True)
    dt2_graph = pydotplus.graph_from_dot_file('tree2.dot')

    result4 = [4, 'Decision Tree (with CDR)', dt2_model_r2, tpr4, 1 - fpr4, auc_dt4, precision4]

    return dt2_model_r2, conf_matrix4, class_report4, dt2_graph, result4, precision4

dt2_model_r2, conf_matrix4, class_report4, dt2_graph, result4, precision4 = dectree_model4()


def result_to_dict(result_arr, result_dict):
    result_dict['Model #'].append(result_arr[0])
    result_dict['Model Description'].append(result_arr[1])
    result_dict['Accuracy'].append(result_arr[2]) # Accuracy = AUC
    result_dict['Sensitivity'].append(result_arr[3][1])
    result_dict['Specificity'].append(result_arr[4][1])
    result_dict['Positive Predictive Value'].append(result_arr[6])


result_dict = defaultdict(list)
result_to_dict(result1, result_dict)
result_to_dict(result2, result_dict)
result_to_dict(result3, result_dict)
result_to_dict(result4, result_dict)

summary_result = pd.DataFrame.from_dict(result_dict)
summary_result = summary_result.set_index('Model #')