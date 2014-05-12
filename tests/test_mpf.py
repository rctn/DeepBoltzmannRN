from .common import generate_MNIST_data
from .. import simpledbm
import copy
import numpy as np

def test_checkGradientWeights():
    """Test weight gradients numerically
    """
    trn = generate_MNIST_data()
    n_units = (trn.shape[1],20,20)
    dbm = simpledbm.sdbm(n_units, rng=np.random)
    meanSteps = 3
    eps = 1e-4
    weights_init = copy.deepcopy(dbm.weights)
    numerical_dKdw = simpledbm.list_zeros_like(weights_init)
    # for layer_i in range(len(weights_init)):
    #     for unit_i in range(weights_init[layer_i].shape[0]):
    #         for unit_j in range(weights_init[layer_i].shape[1]):
    #             dbm.weights[layer_i][unit_i,unit_j] += eps
    #             K_plus_eps = dbm.flowSamples(trn[:1], 1., meanSteps)
    #             dbm.weights[layer_i][unit_i,unit_j] -= 2*eps
    #             K_minus_eps = dbm.flowSamples(trn[:1], 1., meanSteps)
    #             numerical_dKdw[layer_i][unit_i,unit_j] = (K_plus_eps-K_minus_eps)/(2*eps)
    #             dbm.weights[layer_i][unit_i,unit_j] += eps

    dbm.weights = copy.deepcopy(weights_init)
    dbm.ExTrain(trn[:1], 1, 1., meanSteps)
    dKdW = []
    for i in range(len(weights_init)):
        dKdW.append(weights_init[i] - dbm.weights[i])
    
    for i in range(len(weights_init)):
        x = np.abs(dKdW[i]-numerical_dKdw[i])
        import ipdb; ipdb.set_trace()

    pass


def test_checkGradientBias():
    trn = generate_MNIST_data()
    n_units = (trn.shape[1],200)
    dbm = simpledbm.sdbm(n_units, rng=np.random)
    meanSteps = 3
    eps = 1e-4
    bias_init = copy.deepcopy(dbm.bias)

    bias_plus_eps = copy.deepcopy(bias_init)
    bias_plus_eps[0][9] += eps

    bias_minus_eps = copy.deepcopy(bias_init)
    bias_minus_eps[0][9] -= eps

    dbm.bias = bias_plus_eps
    K_plus_eps = dbm.flowSamples(trn[:10], .005, meanSteps)
    dbm.bias = bias_minus_eps
    K_minus_eps = dbm.flowSamples(trn[:10], .005, meanSteps)
    print((K_plus_eps-K_minus_eps)/(2*eps))

    dbm.bias = copy.deepcopy(bias_init)
    dbm.ExTrain(trn[:10], 1, .005, meanSteps)
    dKdW = []
    for i in range(len(bias_init)):
        dKdW.append(bias_init[i] - dbm.bias[i])
    
    print(dKdW[0][9])
    pass