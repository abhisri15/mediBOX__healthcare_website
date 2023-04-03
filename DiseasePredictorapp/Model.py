import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

train = pd.read_csv('Training.csv')
test = pd.read_csv('Testing.csv')

# train.head()
# test.head()


# droping a useless column named 'Unnamed column'
train.drop('Unnamed: 133', axis=1, inplace=True)

# train.shape

# test.shape

# train.columns

train['prognosis'].value_counts()

train['prognosis'].value_counts().count()

train.isna().sum()

# no null values / missing data -- train set
test.isna().sum()

# no null values/missing data -- test set
symptom_count = train.apply(lambda x: True
if x['itching'] == 1 else False, axis=1)

# Count number of True in the series
num_rows = len(symptom_count[symptom_count == True].index)

symtom_dict = {}
for index, column in enumerate(train.columns):
    symtom_dict[column] = index

train['prognosis'].replace({}, inplace=True)

# train['prognosis']


# for col in train.columns:
#     if col =='prognosis':
#         continue
#     sns.countplot(data = train , x = col)
#     plt.show()


Y = train['prognosis']
X = train.drop('prognosis', axis=1)
y_test = test['prognosis']
X_test = test.drop('prognosis', axis=1)

# plt.rcParams["figure.figsize"] = (20,10)
# corr = X.corr()
# sns.heatmap(corr, square=True, annot=False, cmap="RdBu_r")
# plt.title("Feature Correlation")
# plt.tight_layout()
# plt.show()


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, Y_train.shape)
print(f"Training set size {X_train.shape[0]}")
print(X_val.shape, Y_val.shape)
print(f"Validation set size {X_val.shape[0]}")
print(X_test.shape, y_test.shape)
print(f"Training set size {X_test.shape[0]}")

rf = RandomForestClassifier(n_estimators=10, bootstrap=True, max_depth=10)

rf = rf.fit(X_train, Y_train)
confidence = rf.score(X_val, Y_val)
print(f"Training Accuracy {confidence}")
Y_pred = rf.predict(X_val)
print(f"Validation Prediction {Y_pred}")
accuracy = accuracy_score(Y_val, Y_pred)
print(f"Validation accuracy {accuracy}")
conf_mat = confusion_matrix(Y_val, Y_pred)
print(f"confusion matrix {conf_mat}")
clf_report = classification_report(Y_val, Y_pred)
print(f"classification report {clf_report}")

score = cross_val_score(rf, X_val, Y_val, cv=3)
print(score)

# random forest
result = rf.predict(X_test)
pickle.dump(rf , open('model.pkl','wb'))
accuracy = accuracy_score(y_test, result)
clf_report = classification_report(y_test, result)
print(f"accuracy {accuracy}")
print(f"clf_report {clf_report}")
pickle.dump(rf,open(r'model.pkl','wb'))
test.join(pd.DataFrame(rf.predict(X_test), columns=["predicted"]))[["prognosis", "predicted"]]