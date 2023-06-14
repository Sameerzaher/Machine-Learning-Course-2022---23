# This program predicts if a passenger will survive on the titanic

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


# create model base on desicion tree deep 2
def get_desicion_tree(s):
    # Split the data into independent 'X' and dependent 'y' variables
    X = s.iloc[:, 4:7].values
    y = s.iloc[:, 7].values

    # Initialize a DecisionTreeClassifier with criterion "entropy" and max depth 2
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)

    # Fit the classifier to the training data (X, y) and return it
    return clf.fit(X, y)


# Crate sample model
def create_models(data, m, models):
    # The proven success rate of 63.2%
    p = 0.632

    for i in range(m):
        # Get a random sample of 63.2% of the data with a random state
        t = resample(data, n_samples=int(len(data) * p), random_state=i)
        # Get a random sample of 36.8% of the data with replacement
        subsample = resample(data, n_samples=len(data) - len(t), replace=True)
        # Combine the two data sets
        s = pd.concat([t, subsample])
        # Add the current model to the list of models
        models.append(get_desicion_tree(s))


# Change atribute to numeric number
def change_table_val_from_category_to_numeric(data, num):
    labelEncoder = LabelEncoder()
    # Convert string column to int and append new coulmn
    for i in range(num):
        data["is_" + data.columns[i]] = labelEncoder.fit_transform(data.iloc[:, i].values)


def majority_key(k):
    u, c = np.unique(k, return_counts=True)
    return u[c.argmax()]


def prediction(models, dt):
    # list of all prediction of our model
    p = list()
    for m in models:
        # Predict class or regression value for X. For a classification model, the predicted class for each sample in X is returned. For a regression model, the predicted value based on X is returned.
        p.append(m.predict(dt.iloc[:, 4:7].values))
    # Combine the m resulting models and give majority key inside
    majority = np.apply_along_axis(majority_key, 0, np.array(p))
    # Change data from numeric to string
    result_test = np.sum(majority == dt.is_survived) * 100 / len(majority)
    print("Success {}%".format(result_test))
    return majority


def begin(num_m, num_f):
    # Load the data from file
    data = pd.read_csv("DATAFILE.csv")
    TestData = pd.read_csv('TESTDATAFILE.csv', names=["pclass", "age", "gender", "survived"])
    change_table_val_from_category_to_numeric(data, num_f)
    change_table_val_from_category_to_numeric(TestData, num_f)
    models = list()
    # becouse of the bias in the training is better to get rid of duplicate to get better model
    create_models(data.drop_duplicates(), num_m, models)
    p = prediction(models, TestData)
    # Drop all numeric coulmn
    TestData = TestData.iloc[:, :4]
    TestData['prediction'] = p
    # Change data from numeric to string
    TestData['prediction'] = np.where(TestData['prediction'] == 1, 'yes', 'no')
    TestData.to_csv('ResultBagging.csv', index=False)
    print(TestData)


if __name__ == "__main__":
    # adaBoost model =100 feture=4
    begin(100, 4)
