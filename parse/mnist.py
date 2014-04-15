
import os
import gzip
import cPickle
import numpy as np
import skimage.transform

def load_data(dataset):
    """Loads the dataset. Snippets from:
    https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/logistic_sgd.py

    Parameters
    ----------
    dataset: string
        The path to the dataset (here MNIST)
    """

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    return train_set, valid_set, test_set

def resize_data(data, output_image_shape):
    """Resize the input MNIST data.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input MNIST data with flattened image dimensions

    output_image_shape : tuple or ndarray
        Target image shape x_dim, y_dim
    """
    data = data.reshape(data.shape[0], np.sqrt(data.shape[1]), np.sqrt(data.shape[1]))

    rdata = np.empty((data.shape[0], output_image_shape[0]*output_image_shape[1]), dtype=data.dtype)
    for i in xrange(data.shape[0]):
        rdata[i] = skimage.transform.resize(data[i], output_image_shape).reshape(rdata.shape[1])

    return rdata