import os
from subprocess import call
import numpy as np

print("")

if not os.path.exists("cifar-10-python.tar.gz"):
    print("Downloading...")
    call(
        "wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")

cifar_python_directory = os.path.abspath("cifar-10-batches-py")
if not os.path.exists(cifar_python_directory):
    print("Extracting...")
    call(
        "tar -zxvf cifar-10-python.tar.gz",
        shell=True
    )
    print("Extracting successfullycd .. done to {}.".format(cifar_python_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")

print("Converting...")
cifar_caffe_directory = os.path.abspath('cifar_10/')
if not os.path.exists(cifar_caffe_directory):

    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dictionary = cPickle.load(fo)
        fo.close()
        return dictionary


    def load_data(train_file):

        d = unpickle(
            os.path.join(cifar_python_directory, train_file)
        )
        data = d['data']
        labels = d['labels']

        return (
            np.array(data), np.array(labels)
        )

    def loadCIFAR10():
        batches = np.arange(1, 5)
        train_x = np.array([])
        train_y = np.array([])
        for i in batches:
            train_file = "data_batch_" + str(i)
            x, y = load_data(train_file)
            train_x = np.append(train_x, x)
            train_y = np.append(train_y, y)

        test_x, test_y = load_data("test_batch")

        return train_x, train_y, test_x, test_y

else:
    print("Conversion was already done. Did not convert twice.\n")