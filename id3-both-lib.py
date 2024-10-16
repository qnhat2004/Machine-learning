import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Đọc dữ liệu
df = pd.read_csv('diabetes.csv', index_col=0)

# Xử lý các cột dạng phân loại bằng LabelEncoder trước khi chia dữ liệu
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Tách dữ liệu thành X và y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Định nghĩa mô hình và grid search
dt = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [5, 10, 20, 25],
    'min_samples_leaf': [10, 20, 50, 100, 120],
    'criterion': ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring="accuracy")

# Tìm kiếm siêu tham số tốt nhất
best_model = grid_search.fit(X_train, y_train)

# In ra các siêu tham số tốt nhất
print("Best parameters found: ", grid_search.best_params_)

# Dự đoán với tập kiểm tra
y_pred = best_model.predict(X_test)

# In ra độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")

# In thêm các chỉ số đánh giá khác
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# In báo cáo chi tiết và ma trận nhầm lẫn
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
