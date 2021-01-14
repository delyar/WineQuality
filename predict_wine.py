import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from subprocess import Popen
from sklearn.neural_network import MLPClassifier
import numpy as np
import os

wine_data = pd.read_csv('winequality-red.csv')
X = wine_data[[
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]].values

y = wine_data[['quality']]

# Map quality to 'good' and 'poor' to get a better quality
y_mean = np.mean(y)
y_third_quartile = np.percentile(y, 75)
i = 0
for quality_in_int in y.values.ravel():
    if quality_in_int >= int(y_third_quartile):
        y.values[i][0] = 1  # 1 means Good
    else:
        y.values[i][0] = 0  # 0 means Bad
    i += 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Support Vector Machine:
svm_model = svm.SVC()
svm_model.fit(X_train, y_train.values.ravel())

svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
# svm_precision_score = precision_score(y_test, svm_predictions, average='micro')
# print(svm_precision_score)

# Logistic Regression:
log_reg_model = LogisticRegression(solver='liblinear')
log_reg_model.fit(X_train, y_train.values.ravel())

log_reg_predictions = log_reg_model.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)

# Neural Network:
nn = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(10, 15), random_state=1)
nn.fit(X_train, y_train.values.ravel())

nn_predictions = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)

# making correctness labels:
svm_correctness = []
log_reg_correctness = []
nn_correctness = []
j = 0
for prediction in svm_predictions:
    if prediction != y_test.values[j][0]:
        svm_correctness.append('SVM Wrong')
    else:
        svm_correctness.append('SVM Correct')
    j += 1

z = 0
for prediction in log_reg_predictions:
    if prediction != y_test.values[z][0]:
        log_reg_correctness.append('LogReg Wrong')
    else:
        log_reg_correctness.append('LogReg Correct')
    z += 1

t = 0
for prediction in nn_predictions:
    if prediction != y_test.values[t][0]:
        nn_correctness.append('NN Wrong')
    else:
        nn_correctness.append('NN Correct')
    t += 1

print('---------------------------------------')
print("SVM Accuracy: ", svm_accuracy)
print("Logistic Regression Accuracy: ", log_reg_accuracy)
print("Neural Network Accuracy: ", nn_accuracy)
print('---------------------------------------')

# Open In Excel
if os.path.exists('wine_prediction.csv'):
    os.remove('wine_prediction.csv')
else:
    print("The file does not exist")

with open('wine_prediction.csv', 'w', newline='') as csvfile:
    fieldnames = ['svm_prediction',
                  'svm correct?',
                  'log_reg_prediction',
                  'log_reg_correct?',
                  'nn_prediction',
                  'nn_correct?',
                  'actual_quality 1: Good, 0: Poor',
                  'fixed acidity',
                  'volatile acidity',
                  'citric acid',
                  'residual sugar',
                  'chlorides',
                  'free sulfur dioxide',
                  'total sulfur dioxide',
                  'density',
                  'pH',
                  'sulphates',
                  'alcohol'
                  ]
    the_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    the_writer.writeheader()
    counter = 0
    for prediction in svm_predictions:
        the_writer.writerow({'svm_prediction': prediction,
                             'svm correct?': svm_correctness[counter],
                             'log_reg_prediction': log_reg_predictions[counter],
                             'log_reg_correct?': log_reg_correctness[counter],
                             'nn_prediction': nn_predictions[counter],
                            'nn_correct?': nn_correctness[counter],
                             'actual_quality 1: Good, 0: Poor': y_test.values[counter][0],
                             'fixed acidity': X_test[counter][0],
                             'volatile acidity': X_test[counter][1],
                             'citric acid': X_test[counter][2],
                             'residual sugar': X_test[counter][3],
                             'chlorides': X_test[counter][4],
                             'free sulfur dioxide': X_test[counter][5],
                             'total sulfur dioxide': X_test[counter][6],
                             'density': X_test[counter][7],
                             'pH': X_test[counter][8],
                             'sulphates': X_test[counter][9],
                             'alcohol': X_test[counter][10]
                             })
        counter += 1

p = Popen('wine_prediction.csv', shell=True)
