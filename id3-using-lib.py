import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes.csv', index_col=0)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=2)
tree.fit(X_train, y_train)

# Dự đoán với tập kiểm tra
y_pred = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))