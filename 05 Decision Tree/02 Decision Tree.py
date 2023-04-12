#Importing basic libaries
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

#Reading in data
df = pd.read_csv('villagers.csv')

print('Data:')
print(df)
df.head()



print(df.shape)
print(df.groupby('Name').size())
print(df['Gender'].value_counts())

#Encoding
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder() 
df['Gender'] = lb.fit_transform(df['Gender'])
df['Name'] = lb.fit_transform(df['Name'])
df['Species'] = lb.fit_transform(df['Species'])
df['Personality'] = lb.fit_transform(df['Personality'])
df['Birthday'] = lb.fit_transform(df['Birthday'])

# Select Features
feature = df.drop('Gender', axis=1)

# Select Target
target = df['Gender']

# Set Training and Testing Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature , target, 
                                                    shuffle = True, 
                                                    random_state=42,
                                                    test_size=0.2)

# Show the Training and Validation Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l2', C=1.0)

# Train the model on the training data
logreg.fit(X_train, y_train)

# Predict the target labels on the test set
y_test_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Calculate accuracy and other evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1_score = f1_score(y_test, y_test_pred)

print('Accuracy:' , (accuracy))
print('Demical: {:.2f}'.format(accuracy))

print('Precision:' , (precision))
print('Decimal: {:.2f}'.format(precision))

print('Recall:' , (recall))
print('Decimal: {:.2f}'.format(recall))

print('F1 Score:' , (f1_score))
print('Decimal: {:.2f}'.format(f1_score))

