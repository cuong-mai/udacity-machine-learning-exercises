# Import statements
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model. - DONE
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
# model = SVC(C=0.01, kernel='poly', degree=4)
model = SVC(kernel='rbf', gamma=26.3)

# TODO: Fit the model. - DONE
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred. - DONE
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc. - DONE
acc = accuracy_score(y, y_pred)
print(acc)