"""
Common utilities for testing
"""
from ..parse import mnist
import numpy as np

def generate_MNIST_data():
    trn,val,test = mnist.load_data('../mnist.pkl.gz')
    bin_trn = np.round(trn[0])
    return bin_trn