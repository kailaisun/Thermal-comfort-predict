import pandas as pd
from evalc import evaluate_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Transformer_class import TransformerClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error


file_path='data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')


# Select features and label
label_column = 'TSV-7 scale'
excluded_columns = ['TSV-7 scale']
feature_columns = [col for col in data.columns if col not in excluded_columns]


X = data[feature_columns]
y = data[label_column]
y=y-min(y)


# One-hot encode categorical columns (if applicable)
X = pd.get_dummies(X, drop_first=True)



class OrdinalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,learner):
        self.learner = learner
        self.ordered_learners = dict()
        self.classes = []

    def fit(self,X,y):
        self.classes = np.sort(np.unique(y))
        assert self.classes.shape[0] >= 3, f'OrdinalClassifier needs at least 3 classes, only {self.classes.shape[0]} found'

        for i in range(self.classes.shape[0]-1):
            N_i = np.vectorize(int)(y > self.classes[i])
            learner = clone(self.learner).fit(X,N_i)
            self.ordered_learners[i] = learner

    def predict(self,X):
        return np.vectorize(lambda i: self.classes[i])(np.argmax(self.predict_proba(X), axis=1))

    def predict_proba(self,X):
        predicted = [self.ordered_learners[k].predict_proba(X)[:,1].reshape(-1,1) for k in self.ordered_learners]

        N_1 = 1-predicted[0]
        N_K  = predicted[-1]
        N_i= [predicted[i] - predicted[i+1] for i in range(len(predicted) - 1)]

        probs = np.hstack([N_1, *N_i, N_K])

        # probs=probs*[0.5,0.63, 1, 1.63, 1, 0.63, 0.5]
        # probs = probs * [1.63, 1, 0.63, 0.5, 0.63, 1, 1.63]


        return probs


# Re-split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



clf =  KNeighborsClassifier()# XGBClassifier(n_estimators=20,max_depth=10,objective='multi:softmax',num_class=7)  
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_class = clf.predict(X_test)

# Calculate accuracy and generate a classification report
accuracy = accuracy_score(y_test, y_pred_class)
class_report = classification_report(y_test, y_pred_class)
print('baseline')
print(accuracy), print(mean_absolute_error(y_test,y_pred_class)),print(class_report)


model = OrdinalClassifier( KNeighborsClassifier())  #XGBClassifier(n_estimators=20,max_depth=10,objective='multi:softprob',num_class=7)
model.fit(X_train, y_train)
y_pred_class = model.predict(X_test)

# Calculate accuracy and generate a classification report
accuracy = accuracy_score(y_test, y_pred_class)
class_report = classification_report(y_test, y_pred_class)
print('order')
print(accuracy),  print(mean_absolute_error(y_test,y_pred_class)),print(class_report)

