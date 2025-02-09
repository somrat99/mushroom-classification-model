import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("./mushrooms.csv")


# info of dataset
print(df.info())

# describe the dataset
print(df.describe())

# check for missing values
print(df.isnull().sum())



# dataset columns
columns = df.columns

# Encoding the categorical columns
for col in columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# Dropping Veil-Type because there is only one type of value so it does not help in modeling
df.drop(['veil-type'],axis = 1,inplace = True)



# correlation matrix heatmap
corr_matrix = df.corr(method='pearson')
plt.figure(figsize=(20, 16)) 
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", square=True,linewidths=1,annot_kws={"size": 5})
plt.title('Correlation Matrix (Pearson method)', fontsize=16)  # Fixed title
plt.show()



# modeling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

X = df.drop(['class'],axis =1)
y = df['class']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000,random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

