import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

from xgboost import XGBClassifier
import pickle
from graphviz import Digraph
from xgboost import plot_tree, plot_importance

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score



seed = 2

test_size = 0.2

iris_data = load_iris()
X, y = iris_data.data, iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

def fit(X, y):
  model = XGBClassifier()
  model.fit(X, y)
  return model


def eval(model, X, y):
  y_pred = model.predict(X)
  preds= [round(value) for value in y_pred]
  accuracy = accuracy_score(y, preds)
  return preds, accuracy

model = fit(X_train, y_train)
_, accuracy = eval(model, X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

def save(model):
  with open('iris.model', 'wb') as fout:
    pickle.dump(model, fout)

def restore():
  with open('iris.model', 'rb') as fin:
    iris_model = pickle.load(fin)
    _, accuracy = eval(iris_model, X_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

# plot_tree(model, num_trees=4)
# plt.show()
# plt.savefig('tree-from-top-to-bottom.png')

# plot_tree(model, num_trees=0, rankdir='LR')
# plt.show()
# plt.savefig('tree-from-left-to-right.png')

kfold = KFold(n_splits=5, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(model.feature_importances_)

# df = pd.DataFrame(data=model.feature_importances_, index=iris_data.feature_names).T 
# df.plot.bar(figsize=(12, 6))
# plt.show()
# plot_importance(model)
# plt.show()


from sklearn.feature_selection import SelectFromModel

thresholds = np.sort(model.feature_importances_)
for c in thresholds:
  selection = SelectFromModel(model, threshold=c, prefit=True)
  select_X_train = selection.transform(X_train)
  select_X_test = selection.transform(X_test)
  selection_model = fit(select_X_train, y_train)
  _, accuracy = eval(selection_model, select_X_test, y_test)
  print("Threshold = %.3f, n = %d, Accuracy: %.2f%%" % (c, select_X_train.shape[1], accuracy*100.0))
  

# from time import time

# results = []
# num_threads = [1, 2, 3, 4]
# for n in num_threads:
#   start = time()
#   model = XGBClassifier(nthread=n)
#   model.fit(X_train, y_train)
#   elapsed = time() - start
#   print(n, elapsed)
#   results.append(elapsed)


# # Single Thread XGBoost, Parallel Thread CV
# start = time()
# model = XGBClassifier(nthread=1)
# results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
# print("Single XGB Parallel CV: %s" % (time() - start))

# # Parallel Thread XGBoost, Single Thread CV
# start = time()
# model = XGBClassifier(nthread=-1)
# results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_log_loss', n_jobs=1)
# print("Paralle XGB Single CV: %s" % (time() - start))

# # Parallel Thread XGBoost, Parallel Thread CV
# start = time()
# model = XGBClassifier(nthread=-1)
# results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
# print("Parallel XGB Parallel CV: %s" % (time() - start))

from sklearn.model_selection import GridSearchCV

n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)
print(grid_result)
