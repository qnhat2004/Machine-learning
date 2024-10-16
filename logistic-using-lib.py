import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
df = pd.read_csv('diabetes.csv', index_col=0)

# Chia dữ liệu thành X và y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Mã hóa các cột chứa dữ liệu dạng object
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Logistic Regression
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

# Dự đoán với tập kiểm tra
y_pred = clf.predict(X_test)

# Tính toán và in độ chính xác
print(accuracy_score(y_test, y_pred))
