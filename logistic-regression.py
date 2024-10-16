from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(S):
    """
    S: a numpy array
    return sigmoid function of each element of S, avoiding overflow issues
    """
    S = np.clip(S, -500, 500)  # giới hạn giá trị để tránh tràn số
    return 1/(1 + np.exp(-S))

def bias_trick(X):
    N = X.shape[0]
    return np.concatenate((X, np.ones((N, 1))), axis=1)

def prob(w, X):
    """
    X: a 2d numpy array of shape (N, d). N datatpoint, each with size d
    w: a 1d numpy array of shape (d)
    """
    return sigmoid(X.dot(w))

def loss(w, X, y, lam):
    """
    X, w as in prob 
    y: a 1d numpy array of shape (N). Each elem = 0 or 1 
    """
    z = prob(w, X)
    z = np.clip(z, 1e-15, 1 - 1e-15)  # Giới hạn giá trị của z
    return -np.mean(y*np.log(z) + (1-y)*np.log(1-z)) + 0.5*lam/X.shape[0]*np.sum(w*w)

def predict(w, X, threshold=0.5):
    """
    predict output of each row of X
    X: a numpy array of shape
    threshold: a threshold between 0 and 1 
    """
    res = np.zeros(X.shape[0])
    res[np.where(prob(w, X) > threshold)[0]] = 1
    return res 

def logistic_regression(w_init, X, y, lam=0.001, lr=0.1, nepoches=2000):
    # lam - regularization parameter, lr - learning rate, nepoches - number of epochs
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init 
    # store history of loss in loss_hist
    loss_hist = [loss(w_init, X, y, lam)]
    ep = 0 
    while ep < nepoches: 
        ep += 1
        mix_ids = np.random.permutation(N)
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            zi = sigmoid(xi.dot(w))
            w = w - lr*((zi - yi)*xi + lam*w)
        loss_hist.append(loss(w, X, y, lam))
        if np.linalg.norm(w - w_old)/d < 1e-6:
            break 
        w_old = w
    return w, loss_hist 

np.random.seed(2)
# Đọc dữ liệu
df = pd.read_csv('diabetes.csv', index_col=0)

# Chia dữ liệu thành X và y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reset chỉ số của y_train để đảm bảo các chỉ số khớp với X_train
y_train = y_train.reset_index(drop=True)

# bias trick cho tập huấn luyện và kiểm tra
X_train_bar = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
X_test_bar = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)

# Khởi tạo trọng số ngẫu nhiên
w_init = np.random.randn(X_train_bar.shape[1])

# Tham số regularization
lam = 0.0001

# Huấn luyện mô hình Logistic Regression trên tập huấn luyện
w, loss_hist = logistic_regression(w_init, X_train_bar, y_train, lam, lr=0.05, nepoches=500)

# Dự đoán trên tập kiểm tra
y_pred = predict(w, X_test_bar)

# Tính toán và in ra độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print('Solution of Logistic Regression:', w)
print('Final loss:', loss(w, X_train_bar, y_train, lam))
print('Accuracy on test set:', accuracy)
