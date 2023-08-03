#Cancer

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prediction(concave_points_mean, area_mean, radius_mean, perimeter_mean, concavity_mean):
    df = pd.read_csv('cancer.csv')
    df.drop(df.columns[[0,-1]], axis=1, inplace=True)


    # Split the features data and the target
    X_data = df.drop(['diagnosis'], axis=1)
    y_data = df['diagnosis']

    # Encoding the target value
    y_enc = np.asarray([1 if c == 'M' else 0 for c in y_data])
    cols = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean',]
    X_data = df[cols]
    print(X_data.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_enc,test_size=0.3, random_state=43)

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
    X1_test=np.array([concave_points_mean, area_mean, radius_mean, perimeter_mean, concavity_mean])
    X1_test=X1_test.reshape((1,-1))
    return model.predict(X1_test)[0]
