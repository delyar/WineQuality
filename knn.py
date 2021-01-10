import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
# Features
x = data[[
    'buying',
    'maint',
    'safety'
]].values
# Label
y = data[['class']]

# Converting string Data to int data for ML Understandable
# x
Le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = Le.fit_transform(x[:, i])
# Y
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)

# Create KNN Model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
# Train the model:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn.fit(x_train, y_train)
prediction = x_test
accuracy = metrics.accuracy_score(y_test, prediction)
print('Predictions: ', prediction)
print('Accuracy: ', accuracy)