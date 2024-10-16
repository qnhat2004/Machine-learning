import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # to plot charts
import seaborn as sns # used for data visualization
import warnings # avoid warning flash
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Dropping duplicate values
df = df.drop_duplicates()

# Check for missing or zero values
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())  # Normal distribution
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())  # Normal distribution
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())  # Skewed distribution
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())  # Skewed distribution
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())  # Skewed distribution

# Select features and target variable
df_selected = df.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis='columns')

# Normalize data using QuantileTransformer
from sklearn.preprocessing import QuantileTransformer
quantile = QuantileTransformer()
X = quantile.fit_transform(df_selected.drop('Outcome', axis=1))
y = df_selected['Outcome']

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the model with GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [5, 10, 20, 25],
    'min_samples_leaf': [10, 20, 50, 100, 120],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring="accuracy")
best_model = grid_search.fit(X_train, y_train)

# Predict with the test set
dt_pred = best_model.predict(X_test)

# Calculate accuracy score
def accuracy_score(y_test, y_pred):
    return np.sum(np.equal(y_test, y_pred)) / len(y_test)

# Print the accuracy
print("Accuracy Score: ", accuracy_score(y_test, dt_pred))

# Optionally, print other metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
print("\nClassification Report:\n", classification_report(y_test, dt_pred))
print("\nF1 Score: ", f1_score(y_test, dt_pred))
print("\nPrecision: ", precision_score(y_test, dt_pred))
print("\nRecall: ", recall_score(y_test, dt_pred))

# Confusion Matrix heatmap
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt="d")
plt.show()
