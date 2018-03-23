# -*- coding:UTF-8 -*-
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys

def split_train_test(data, test_size=0.2):
  train_data, test_data = train_test_split(data, test_size=test_size)
  train_libsvm_file  = "train_libsvm_feature"
  test_libsvm_file = "test_libsvm_feature"
  with open(train_libsvm_file, "w") as fout:
    for line in train_data:
      fout.write(line)
  with open(test_libsvm_file, "w") as fout:
    for line in test_data:
      fout.write(line)
  dtrain = xgb.DMatrix(train_libsvm_file)
  dtest = xgb.DMatrix(test_libsvm_file)
  return dtrain, dtest

def load_train_test(all_text_libsvm_feature_file, test_size=0.2):
  data = []
  with open("all_text_libsvm_feature") as fin:
    for line in fin:
      data.append(line)
  return split_train_test(data, test_size)

def fit(dtrain):
  param = {'max_depth':1, 'eta':0.5, 'silent':1, 'objective':'binary:logistic'}
  watchlist = [(dtrain, 'train')]
  num_round = 100
  bst = xgb.train(param, dtrain, num_round, watchlist)
  return bst

def predict(bst, dtest):
  preds = bst.predict(dtest)
  # 因为2分类所以阈值是 0.5
  preds = [int(pred > 0.5) for pred in preds]
  return preds

def fmt_float(num):
  return "%.4f" % num

def main():
  argv = sys.argv
  if len(argv) > 1:
    test_size = float(argv[1])
  else:
    test_size = 0.2

  all_text_libsvm_feature_file = "all_text_libsvm_feature"
  dtrain, dtest = load_train_test(all_text_libsvm_feature_file, test_size)

  bst = fit(dtrain)
  y_preds = predict(bst, dtest)

  y_test = dtest.get_label()
  precision = precision_score(y_test, y_preds)
  recall = recall_score(y_test, y_preds)
  f1 = f1_score(y_test, y_preds)
  # 展示到控制台
  table = []
  th = ["f1", "precision", "recall"]
  table.append(th)
  tr = [fmt_float(f1), fmt_float(precision), fmt_float(recall)]
  table.append(tr)
  for row in table:
    msg = "\t".join(row)
    print(msg)
  bst.save_model("ai-judger.model")

if __name__ == '__main__':
  main()