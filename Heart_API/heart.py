#Heart

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


def prediction(cp, trestbps, chol, fbs, restecg, thalach, exang):
    df = pd.read_csv('heart.csv')

    categorical_val = []
    continous_val = []
    for column in df.columns:
        if len(df[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continous_val.append(column)

    categorical_val.remove('target')
    dataset = pd.get_dummies(df, columns = categorical_val)

    cols = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']       
    X = df[cols]
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
    print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

    model = ensemble.RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

    clf_report = classification_report(y_test, y_pred)
    print('Classification report')
    print("---------------------")
    print(clf_report)
    print("_____________________")
    X1_test=np.array([cp, trestbps, chol, fbs, restecg, thalach, exang])
    X1_test=X1_test.reshape((1,-1))
    return model.predict(X1_test)[0]

pickle.dump(model, open('model.pkl', 'wb'))