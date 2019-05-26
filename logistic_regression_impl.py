# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('weatherAUS.csv')
# df['qualify'] = df['qualify'].map({'yes': True, 'no': False})
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainToday'] = df['RainToday'].fillna(0)
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].fillna(0)
# X columns - 13, 14 (humidity), 19 - 21
# Y columns - 23

X = df.iloc[:, [13,14,19,20,21]].values
X = X.astype(int)
y = df.iloc[:,23].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# we've preprocessed and scaled the data so far, create logistic
# regression and then train it

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0, n_jobs = 1)
lr.fit(X_train, y_train)

# check y_pred

y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# visualizing cm
import seaborn as sns
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

sns.countplot(x='RainTomorrow', data=df)