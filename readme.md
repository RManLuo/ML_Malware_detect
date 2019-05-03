# Machine Learning for Malware Detection

edited by Raymond Luo

## Environment

``` bash
python3
RTX2080 I7 9700K 16G RAM
ubuntu 18.04
```

## Requirement

```bash
sudo pip3 install -r requirements.txt
```

Please put the `security_train.xlsx` and `security_test.xlsx` in related floders

## Loadfile

```bash
python3 loadfile.py
```

## Train multi vision lstm

```
python3 train_lstm.py
```

## Train LSTM

```
python3 train_lstm2.py
```

## Train textcnn LSTM

```
python3 train_lstm3.py
```

## Train CNN

```
python3 train_textcnn.py
```

## Train tf-idf model

```
python3 xgboost.py
```

## Stack model

```
python3 stack_result.py
```

## Result

Result will be in `submit` floder

# FILE PREPROCESS

### Data format

| label   | class  | explain                                                      |
| ------- | ------ | ------------------------------------------------------------ |
| file_id | bigint | Number of files                                              |
| label   | bigint | 文件标签，0-正常/1-勒索病毒/2-挖矿程序/3-DDoS木马/4-蠕虫病毒/5-感染型病毒/6-后门程序/7-木马程序. |
| api     | string | The API call list of the file.                               |
| tid     | bigint | The thread number of the file.                               |
| index   | string | The order of the API call.                                   |
----
[阿里云安全恶意程序检测](https://tianchi.aliyun.com/competition/entrance/231694/introduction?spm=5176.12281925.0.0.60c17137jVy9vv)
### Our method:

Firstly, we grouped the file by ‘file-id’. For each files, we then grouped the API calls by ‘tid’ and sorted the API calls by its ‘index’. Finally, we concatenate each thread to one line to get our final represent of 1 sample.

`Sample
LdrLoadDll LdrGetProcedureAddress LdrGetProcedureAddress LdrGetProcedureAddress LdrGetProcedureAddress.......
label 5`

# MODEL

For best score we use 5 models and stack them to get our final score.



## TF-idf model:

In order to get the whole information of the sample. We firstly extract the TF-idf features of the sample. We extract ‘TF-IDF’ feature after we preprocess the sample with 1-5 gram.

```python
vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9, )  # tf-idf特征抽取ngram_range=(1,5)
```

Then we classify the sample base on XGboost method.

 ``` python
param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
          'num_class': 8, 'subsample': 0.8,
          'colsample_bytree': 0.85}  # 参数

 evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
 num_round = 300  # 循环次数
 bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)
 ```

## Text cnn model

In this model, we used ‘text cnn’ to extract the features and classify the sample.

Each sample has been cut to maximum 6000 length and encode as one-hot label.

We input the sample to 1 embedding layer then go to the CNN layers.

We used 2,4,6,8,10 5 different kernel sizes, each convolutional layer has 32 filters. 

After the cnn layers, we concatenate the feature extracted by CNN layers and classify them.

```python
main_input = Input(shape=(maxlen,), dtype='float64')
_embed = Embedding(304, 256, input_length=maxlen)(main_input)
_embed = SpatialDropout1D(0.25)(_embed)
warppers = []
num_filters = 64
kernel_size = [2, 3, 4, 5]
conv_action = 'relu'
for _kernel_size in kernel_size:
    for dilated_rate in [1, 2, 3, 4]:
        conv1d = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation=conv_action,
                        dilation_rate=dilated_rate)(_embed)
        warppers.append(GlobalMaxPooling1D()(conv1d))

fc = concatenate(warppers)
fc = Dropout(0.5)(fc)
# fc = BatchNormalization()(fc)
fc = Dense(256, activation='relu')(fc)
fc = Dropout(0.25)(fc)
# fc = BatchNormalization()(fc)
preds = Dense(8, activation='softmax')(fc)

model = Model(inputs=main_input, outputs=preds)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
return model

```

## CNN_LSTM model

In order to get the sequence information of the sample. We used 3 LSTM based model.

The first one is simple CNN+LSTM. All cnn layer has the same kernel size.

```python
 main_input = Input(shape=(maxlen,), dtype='float64')
    embedder = Embedding(304, 256, input_length=maxlen)
    embed = embedder(main_input)
    # avg = GlobalAveragePooling1D()(embed)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(64, 3, padding='same', activation='relu')(embed)

    conv1_2 = Conv1D(64, 3, padding='same', activation='relu')(conv1_1)

    cnn1 = MaxPool1D(pool_size=2)(conv1_2)
    conv1_1 = Conv1D(64, 3, padding='same', activation='relu')(cnn1)

    conv1_2 = Conv1D(64, 3, padding='same', activation='relu')(conv1_1)

    cnn1 = MaxPool1D(pool_size=2)(conv1_2)
    conv1_1 = Conv1D(64, 3, padding='same', activation='relu')(cnn1)

    conv1_2 = Conv1D(64, 3, padding='same', activation='relu')(conv1_1)

    cnn1 = MaxPool1D(pool_size=2)(conv1_2)
    rl = CuDNNLSTM(256)(cnn1)
    # flat = Flatten()(cnn3)
    # drop = Dropout(0.5)(flat)
    fc = Dense(256)(rl)

    main_output = Dense(8, activation='softmax')(rl)
    model = Model(inputs=main_input, outputs=main_output)
    return model

```

## Mulit vision LSTM

Inspired by out text cnn method. We thought that if different kernel size can have different versions why can we have a mulit version LSTM model. So we first use 3,5,7 3 different kerner sizes. In spite of arrange them as linner order, we use them to extract the features from the embedding layer independly. Between each lyayer we used averge pooling instead of max pooling to get more information of the sequence rather than only one API call. After the CNN layers we have three feature vectors which have same size. We then use  
$$
max_{elements}(v_1,v_2,v_3)
$$
in order to get the most significant feature in different version. After that we used LSTM layer to analyze the sequence and classify it. 

```python
embed_size = 256
    num_filters = 64
    kernel_size = [3, 5, 7]
    main_input = Input(shape=(maxlen,))
    emb = Embedding(304, 256, input_length=maxlen)(main_input)
    # _embed = SpatialDropout1D(0.15)(emb)
    warppers = []
    warppers2 = []  # 0.42
    warppers3 = []
    for _kernel_size in kernel_size:
        conv1d = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(emb)
        warppers.append(AveragePooling1D(2)(conv1d))
    for (_kernel_size, cnn) in zip(kernel_size, warppers):
        conv1d_2 = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(cnn)
        warppers2.append(AveragePooling1D(2)(conv1d_2))
    for (_kernel_size, cnn) in zip(kernel_size, warppers2):
        conv1d_2 = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(cnn)
        warppers3.append(AveragePooling1D(2)(conv1d_2))
    fc = Maximum()(warppers3)
    rl = CuDNNLSTM(512)(fc)
    main_output = Dense(8, activation='softmax')(rl)
    model = Model(inputs=main_input, outputs=main_output)
    return model
```

## Text cnn lstm

This model is mostly  as same as multi version LSTM but we concatenate the 3 feature in order. Because we think this method did't affect the order of the sequence. In each part the order of API calls did't change.

``` python
main_input = Input(shape=(maxlen,), dtype='float64')

    embedder = Embedding(304, 256, input_length=maxlen)
    embed = embedder(main_input)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(16, 3, padding='same')(embed)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv1D(32, 3, padding='same')(relu1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPool1D(pool_size=4)(relu1_2)
    # cnn2模块，kernel_size = 4
    conv2_1 = Conv1D(16, 4, padding='same')(embed)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv1D(32, 4, padding='same')(relu2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPool1D(pool_size=4)(relu2_2)
    # cnn3模块，kernel_size = 5
    conv3_1 = Conv1D(16, 5, padding='same')(embed)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv1D(32, 5, padding='same')(relu3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPool1D(pool_size=4)(relu3_2)
    # 拼接三个模块
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    lstm = CuDNNLSTM(256)(cnn)
    f = Flatten()(cnn1)
    fc = Dense(256, activation='relu')(f)
    D = Dropout(0.5)(fc)
    main_output = Dense(8, activation='softmax')(lstm)
    model = Model(inputs=main_input, outputs=main_output)
    return model
```

### Stack result

We stack the results of the model mentioned above.

``` python
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
```
