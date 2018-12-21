import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import csv
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold

with open("security_test.csv.pkl", "rb") as f:
    file_names = pickle.load(f)
    outfiles = pickle.load(f)

with open("cnn_lstm_result.pkl", "rb") as f:
    cnn_train_result = pickle.load(f)
    cnn_out_result = pickle.load(f)

with open("tfidf_result.pkl", "rb") as f:
    tfidf_train_result = pickle.load(f)
    tfidf_out_result = pickle.load(f)

with open("textcnn_result.pkl", "rb") as f:
    textcnn_train_result = pickle.load(f)
    textcnn_out_result = pickle.load(f)

with open("mulitl_version_lstm_result.pkl", "rb") as f:
    mulitl_version_lstm_train_result = pickle.load(f)
    mulitl_version_lstm_test_result = pickle.load(f)

with open("textcnn_lstm_result.pkl", "rb") as f:
    textcnn_lstm_train_result = pickle.load(f)
    textcnn_lstm_test_result = pickle.load(f)	
	
with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)

train = np.hstack([tfidf_train_result, textcnn_train_result, mulitl_version_lstm_train_result, cnn_train_result,
                   textcnn_lstm_train_result])
test = np.hstack(
    [tfidf_out_result, textcnn_out_result, mulitl_version_lstm_test_result, cnn_out_result, textcnn_lstm_test_result])
meta_test = np.zeros(shape=(len(outfiles), 8))
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
dout = xgb.DMatrix(test)
for i, (tr_ind, te_ind) in enumerate(skf.split(train, labels)):
    print('FOLD: {}'.format(str(i)))
    X_train, X_train_label = train[tr_ind], labels[tr_ind]
    X_val, X_val_label = train[te_ind], labels[te_ind]
    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    dtest = xgb.DMatrix(X_val, X_val_label)  # label可以不要，此处需要是为了测试效果

    param = {'max_depth': 6, 'eta': 0.01, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
             'num_class': 8, 'subsample': 0.9,
             'colsample_bytree': 0.85}  # 参数
    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 10000  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)
    preds = bst.predict(dout)
    meta_test += preds

meta_test /= 5.0
result = meta_test
# print(result)
out = []
for i in range(len(file_names)):
    tmp = []
    a = result[i].tolist()
    # for j in range(len(a)):
    #     a[j] = ("%.5f" % a[j])

    tmp.append(file_names[i])
    tmp.extend(a)
    out.append(tmp)
with open("./submit/mulltimodel_xgd_boost_tf+cnn_mlstm{}.csv".format(
        str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))),
        "w",
        newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
                     ])
    # 写入多行用writerows
    writer.writerows(out)
