import pandas as pd
from pydataset import data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Get the Data
titanic = data('titanic')
titanic.sample(5)  # 5 random datapoints in dataset

# Feature Engineering
titanic = pd.get_dummies(titanic, drop_first=True)

# Test Train Split
X = titanic.drop('survived_yes', axis=1)
y = titanic['survived_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train the model using the training data
LogReg = LogisticRegression(solver='lbfgs')
LogReg.fit(X_train, y_train)

print(LogReg.predict(np.array([[0, 0, 1, 1, ]])))

print('Logistic Regression Model Score', LogReg.score(X_test, y_test))
