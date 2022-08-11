import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# reading dataset
dataset = pd.read_csv("heart_disase.csv")

# checking null values
# print(dataset.isnull().sum())

# replacing null values with the mean value
dataset["education"].fillna(dataset["education"].mean(), inplace=True)
dataset["cigsPerDay"].fillna(dataset["cigsPerDay"].mean(), inplace=True)
dataset["BPMeds"].fillna(dataset["BPMeds"].mean(), inplace=True)
dataset["totChol"].fillna(dataset["totChol"].mean(), inplace=True)
dataset["BMI"].fillna(dataset["BMI"].mean(), inplace=True)
dataset["heartRate"].fillna(dataset["heartRate"].mean(), inplace=True)
dataset["glucose"].fillna(dataset["glucose"].mean(), inplace=True)

# rechecking null values
# print(dataset.isnull().sum())

# preparing X and y for split
X = dataset.drop("TenYearCHD", axis=1)
y = dataset.loc[:, "TenYearCHD"]

# creating train sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
