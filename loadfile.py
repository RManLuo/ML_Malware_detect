import pandas as pd
import pickle
import numpy as np

train_path = r'security_train/security_train.csv'
test_path = r'security_test/security_test.csv'


def read_train_file(path):
    labels = []
    files = []
    data = pd.read_csv(path)
    # for data in data1:
    goup_fileid = data.groupby('file_id')
    for file_name, file_group in goup_fileid:
        print(file_name)
        file_labels = file_group['label'].values[0]
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        api_sequence = ' '.join(result['api'])
        labels.append(file_labels)
        files.append(api_sequence)
    print(len(labels))
    print(len(files))
    with open(path.split('/')[-1] + ".txt", 'w') as f:
        for i in range(len(labels)):
            f.write(str(labels[i]) + ' ' + files[i] + '\n')



def read_test_file(path):
    names = []
    files = []
    data = pd.read_csv(path)
    # for data in data1:
    goup_fileid = data.groupby('file_id')
    for file_name, file_group in goup_fileid:
        print(file_name)
        # file_labels = file_group['label'].values[0]
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        api_sequence = ' '.join(result['api'])
        # labels.append(file_labels)
        names.append(file_name)
        files.append(api_sequence)
    print(len(names))
    print(len(files))
    with open("security_test.csv.pkl", 'wb') as f:
        pickle.dump(names, f)
        pickle.dump(files, f)
    # with open(path.split('/')[-1] + ".txt", 'w') as f:
    #     for i in range(len(names)):
    #         f.write(str(names[i]) + ' ' + files[i] + '\n')


def load_train2h5py(path="security_train.csv.txt"):
    labels = []
    files = []
    with open(path) as f:
        for i in f.readlines():
            i = i.strip('\n')
            labels.append(i[0])
            files.append(i[2:])
    labels = np.asarray(labels)
    print(labels.shape)
    with open("security_train.csv.pkl", 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(files, f)


# def load_test2h5py(path="D:\ML_Malware\security_test.csv.txt"):
#     labels = []
#     files = []
#     with open(path) as f:
#         for i in f.readlines():
#             i = i.strip('\n')
#             labels.append(i[0])
#             files.append(' '.join(i.split(" ")[1:]))
#     labels = np.asarray(labels)
#     print(labels.shape)
#     with open("security_test.csv.pkl", 'wb') as f:
#         pickle.dump(labels, f)
#         pickle.dump(files, f)


if __name__ == '__main__':
    print("read train file.....")
    read_train_file(train_path)
    load_train2h5py()
    print("read test file......")
    read_test_file(test_path)

    # load_test2h5py()
