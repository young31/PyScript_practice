import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocessing(df: pd.DataFrame):
    return

def get_summary(tr_X, tr_y, test_X, test_y, models=None):
    if models is None:
        models = [
            LogisticRegression(),
            SVC(),
            LinearSVC(),
            KNeighborsClassifier(),
            GaussianNB(),
            Perceptron(),
            RandomForestClassifier(),
            DecisionTreeClassifier(),
        ]

    metrics = [
        accuracy_score, precision_score, recall_score, f1_score
    ]

    scores = {}
    for model in models:
        model.fit(tr_X, tr_y)
        
        pred = model.predict(test_X)
        
        tmp = [
            m(test_y, pred) for m in metrics
        ]
        
        scores[model.__str__()] = tmp

    res = pd.DataFrame(scores).T

    res.columns = ['acc', 'precision', 'recall', 'f1']

    return res