import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import csv
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy  as np

with open("security_test.csv.pkl", "rb") as f:
    file_names = pickle.load(f)
    outfiles = pickle.load(f)

with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)

print("start tfidf...")
vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9, )  # tf-idf特征抽取ngram_range=(1,5)

train_features = vectorizer.fit_transform(files)

out_features = vectorizer.transform(outfiles)

# with open("tfidf_feature_no_limit.pkl", 'wb') as f:
#     pickle.dump(train_features, f)
#     pickle.dump(out_features, f)
#
# with open("tfidf_feature_no_limit.pkl", 'rb') as f:
#     train_features = pickle.load(f)
#     out_features = pickle.load(f)
print(train_features.shape)
print(out_features.shape)
meta_train = np.zeros(shape=(len(files), 8))
meta_test = np.zeros(shape=(len(outfiles), 8))
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
for i, (tr_ind, te_ind) in enumerate(skf.split(train_features, labels)):
    X_train, X_train_label = train_features[tr_ind], labels[tr_ind]
    X_val, X_val_label = train_features[te_ind], labels[te_ind]

    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind))
    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    dtest = xgb.DMatrix(X_val, X_val_label)
    dout = xgb.DMatrix(out_features)
    param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
             'num_class': 8, 'subsample': 0.8,
             'colsample_bytree': 0.85}  # 参数

    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 300  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    # dtr = xgb.DMatrix(train_features)
    pred_val = bst.predict(dtest)
    pred_test = bst.predict(dout)
    meta_train[te_ind] = pred_val
    meta_test += pred_test
meta_test /= 5.0
with open("tfidf_result.pkl", 'wb') as f:
    pickle.dump(meta_train, f)
    pickle.dump(meta_test, f)

    # preds = bst.predict(dout)
    #
    #
    # result = preds
    # # print(result)
    # out = []
    # for i in range(len(file_names)):
    #     tmp = []
    #     a = result[i].tolist()
    #     # for j in range(len(a)):
    #     #     a[j] = ("%.5f" % a[j])
    #
    #     tmp.append(file_names[i])
    #     tmp.extend(a)
    #     out.append(tmp)
    # with open("result_xgd_boost_{}.csv".format(str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))), "w",
    #           newline='') as csvfile:
    #     writer = csv.writer(csvfile)

    # 先写入columns_name
    # writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
    #                  ])
    # # 写入多行用writerows
    # writer.writerows(out)
