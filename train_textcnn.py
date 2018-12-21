import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, CuDNNLSTM, \
    SimpleRNNCell, SpatialDropout1D
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, concatenate, MaxPool2D,GlobalMaxPooling1D
from keras import optimizers
from keras import regularizers
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import time
import numpy as np
from scipy import interp
from sklearn import metrics
from keras import backend as K
from keras.models import load_model
import csv
from sklearn.model_selection import StratifiedKFold

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

Fname = 'malware_'
Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
tensorboard = TensorBoard(log_dir='./Logs/' + Time, histogram_freq=0, write_graph=False, write_images=False,
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

with open("security_test.csv.pkl", "rb") as f:
    file_names = pickle.load(f)
    outfiles = pickle.load(f)
with open("security_train.csv.pkl", "rb") as f:
    labels_d = pickle.load(f)
with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)
maxlen = 6000


# with open("wordsdic.pkl", 'rb') as f:
#     tokenizer = pickle.load(f)
#

labels = np.asarray(labels)

labels = to_categorical(labels, num_classes=8)

tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                      split=' ',
                      char_level=False,
                      oov_token=None)
tokenizer.fit_on_texts(files)
tokenizer.fit_on_texts(outfiles)

# with open("wordsdic.pkl", 'wb') as f:
#     pickle.dump(tokenizer, f)

vocab = tokenizer.word_index
print(tokenizer.word_index)
print(len(vocab))
x_train_word_ids = tokenizer.texts_to_sequences(files)
x_out_word_ids = tokenizer.texts_to_sequences(outfiles)

x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=maxlen)

x_out_padded_seqs = pad_sequences(x_out_word_ids, maxlen=maxlen)


# with open('datasets.pkl', 'wb') as f:
#     pickle.dump(x_train_padded_seqs, f)
#     pickle.dump(x_out_padded_seqs, f)
#     pickle.dump(labels, f)


# with open('datasets.pkl', 'rb') as f:
#     x_train_padded_seqs = pickle.load(f)
#     # x_test_padded_seqs = pickle.load(f)
#     x_out_padded_seqs = pickle.load(f)
#     # y_train = pickle.load(f)
#     # y_test = pickle.load(f)
#     labels = pickle.load(f)


def TextCNN():
    num_filters = 64
    kernel_size = [2, 4, 6, 8, 10]
    conv_action = 'relu'
    _input = Input(shape=(maxlen,), dtype='int32')
    _embed = Embedding(304, 256, input_length=maxlen)(_input)
    _embed = SpatialDropout1D(0.15)(_embed)
    warppers = []
    for _kernel_size in kernel_size:
        conv1d = Conv1D(filters=32, kernel_size=_kernel_size, activation=conv_action, padding="same")(_embed)
        warppers.append(MaxPool1D(2)(conv1d))

    fc = concatenate(warppers)
    fc = Flatten()(fc)
    fc = Dropout(0.5)(fc)
    # fc = BatchNormalization()(fc)
    fc = Dense(256, activation='relu')(fc)
    fc = Dropout(0.5)(fc)
    # fc = BatchNormalization()(fc)
    preds = Dense(8, activation='softmax')(fc)

    model = Model(inputs=_input, outputs=preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def dila():
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
def fasttext():
    main_input = Input(shape=(maxlen,), dtype='float64')
    embedder = Embedding(304, 256, input_length=maxlen)
    embed = embedder(main_input)
    # cnn1模块，kernel_size = 3
    gb = GlobalAveragePooling1D()(embed)
    main_output = Dense(8, activation='softmax')(gb)
    model = Model(inputs=main_input, outputs=main_output)
    return model


meta_train = np.zeros(shape=(len(x_train_padded_seqs), 8))
meta_test = np.zeros(shape=(len(x_out_padded_seqs), 8))
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
for i, (tr_ind, te_ind) in enumerate(skf.split(x_train_padded_seqs, labels_d)):
    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind))
    X_train, X_train_label = x_train_padded_seqs[tr_ind], labels[tr_ind]
    X_val, X_val_label = x_train_padded_seqs[te_ind], labels[te_ind]

    model = dila()
    # model = load_model('model_weight.h5')
    # print(model.summary())
    # exit()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model_save_path = './model/model_weight_testcnn_{}.h5'.format(str(i))
    if i in [-1]:
        model = model.load_weights(model_save_path)
        print(model.evaluate(X_val, X_val_label))
    else:
        ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min', baseline=None,
                            restore_best_weights=False)

        checkpoint = model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
        history = model.fit(X_train, X_train_label,
                            batch_size=32,
                            epochs=100,
                            shuffle=True,
                            validation_data=(X_val, X_val_label), callbacks=[tensorboard, ear,checkpoint])
        model.load_weights(model_save_path)
        # model.save('./model/model_weight_TextCNN_1st_{}.h5'.format(str(i)))

        # model = load_model('model_weight.h5')
    pred_val = model.predict(X_val)
    pred_test = model.predict(x_out_padded_seqs)

    meta_train[te_ind] = pred_val
    meta_test += pred_test
    K.clear_session()
meta_test /= 5.0
with open("textcnn_result.pkl", 'wb') as f:
    pickle.dump(meta_train, f)
    pickle.dump(meta_test, f)

#
# with open("TextCNN_1st_result.pkl", 'wb') as f:
#     pickle.dump(train_result, f)
#     pickle.dump(out_result, f)
#
# result = model.predict(x_out_padded_seqs)
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
# with open("result_textcnn_TextCNN_1st.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#
#     # 先写入columns_name
#     writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
#                      ])
#     # 写入多行用writerows
#     writer.writerows(out)
