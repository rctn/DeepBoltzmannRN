from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.optimize import fmin_bfgs

def flowRBM(params,*args):
    """MPF objective function for RBM. Used to pretrain DBM layers.

    Parameters
    ----------
    params : array-like, shape (n_weights+n_biasv+nbiash)
        Weights (flattened), visible biases, and hidden biases.
        
    *args : tuple or list or args
        Expects eps which is coefficient for MPF and state which is a
        vectors of visible states to learn on.
    """
    eps = args[0]
    data = args[1]
    n_visible = args[2]
    n_hidden = args[3]
    layer = args[4]
    n_data = data.shape[0]
    num = n_visible*n_hidden
    weights = params[:num].reshape((n_visible,n_hidden))
    biasv = params[num:num+n_visible]
    num += n_visible
    biash = params[num:num+n_hidden]
    sflow = np.sum([np.exp(.5*(energydiff(weights,biasv,biash,state,ii))) for ii in xrange(n_units)])
    k = eps*sflow/n_data
    return k

def gradFlowRBM(params,*args):
    """Gradient of MPF objective function for RBM. Used to pretrain DBM layers.

    Parameters
    ----------
    params : array-like, shape (n_units^2+2*n_units)
        Weights (flattened), visible biases, and hidden biases.

    *args : tuple or list or args
        Expects eps which is coefficient for MPF and state which is a
	vectors of visible states to learn on.
    """
    eps = args[0]
    state = args[1]
    n_data = state.shape[0]
    n_total = params.shape[0]
    n_units = int(np.sqrt(1+n_total)-1)
    weights = params[:n_units**2].reshape((n_units,n_units))
    biasv = params[n_units**2:n_units**2+n_units]
    biash = params[n_units**2+n_units:n_units**2+2*n_units]
    dkdw = np.zeros_like(weights)
    dkdbv = np.zeros_like(biasv)
    dkdbh = np.zeros_like(biash)
    for ii in xrange(n_visible):
        diffew = dedwRBM(weights,biasv,biash,state)-dedwRBMBF(weights,biasv,biash,state,ii)
	diffebv = dedbvRBM(weights,biasv,biash,state)-dedbvRBMBF(weights,biasv,biash,state,ii)
	diffebh = dedbhRBM(weights,biasv,biash,state)-dedbhRBMBF(weights,biasv,biash,state,ii)
	diffe = np.exp(.5*(energyRBM(weights,biasv,biash,state)-energyRBMBF(weights,biasv,biash,state,ii)))
	dkdw += np.dot(np.transpose(diffew,axes=(1,2,0)),diffe)
	dkdbv += np.dot(diffebv.T,diffe)
	dkdbh += np.dot(diffebh.T,diffe)
    return eps*np.concatenate((dkdw.flatten(),dkdbv,dkdbh))/n_data


def energydiff(weights,biasv,biash,data,n):
    """Energy function for RBM
    
    Parameters
    ----------
    weights : array-like, shape (n_units,n_units)
        Visble to hidden weights
        
    biasv : array-like, shape n_units
        Biases for visible units

    biash : array-like, shape n_units
        Biases for hidden units
    """
    flip = data.copy()
    flip[:,n] = 1-flip[:,n]
    logTerm = np.sum(np.log(1.+np.exp(biash+np.dot(data,weights))),axis=1)
    logTermBF = np.sum(np.log(1.+np.exp(biash+flip.dot(weights))),axis=1)
    return (-data.dot(biasv)-logTerm)-(flip.dot(biasv)-logTermBF)

def dedwdiff(weights,biash,data,n):
    n_data = data.shape[0]
    flip = data.copy()
    flip[:,n] = 1-flip[:,n]
    dedw = -np.array([np.outer(data[ii],sigm(biash+state[ii].dot(weights))) for ii in xrange(n_data)])
    dedwBF = -np.array([np.outer(flip[ii],sigm(biash+flip[ii].dot(weights))) for ii in xrange(n_data)])
    return dedw-dedwBF

def dedbvdiff(data,n):
    flip = state.copy()
    flip[:,n] = 1-flip[:,n]
    return (-state)-(-flip)

def dedbhdiff(weights,biash,state,n):
    n_data = state.shape[0]
    flip = state.copy()
    flip[:,n] = 1-flip[:,n]
    dedbh = -sigm(np.tile(biash,(n_data,1))+data.dot(weights))
    dedbhBF = -sigm(np.tile(biash,(n_data,1))+flip.dot(weights))
    return dedbh-dedbhBF

def sigm(x):
    """Sigmoid function

    Parameters
    ----------
    x : array-like
        Array of elements to calculate sigmoid for.
    """
    return 1./(1+np.exp(-x))

class pretrainRBM(object):
    """RBM object used to pretrain a DBM layer"""

    def __init__(self,n_units,layer,rng):
        if layer == 'bottom':
            self.n_visible = 2*n_units
            self.n_hidden = n_units
        elif layer == 'middle':
            self.n_visible = n_units
            self.n_hidden = n_units
        elif layer == 'top':
            self.n_visible = n_units
            self.n_hidden = 2*n_units
        else:
            raise ValueError
        self.layer = layer
        self.rng = rng
        self.weights = rng.randn(self.n_visible,self,n_hidden)
        self.biasv = rng.randn(self.n_visible)
        self.biash = rng.randn(self.n_hidden)
        self.contrain_weights()
    
    def constrainWeights(self):
        if self.layer == 'bottom':
            num = int(self.n_visible/2)
            weights = .5*(self.weights[:num]+self.weights[num:])
            self.weights[:num] = weights
            self.weights[num:] = weights
            biasv = .5*(self.biasv[:num]+self.biasv[num:])
            self.biasv[:num] = biasv 
            self.biasv[num:] = biasv
        elif self.layer == 'top':
            num = int(self.n_hidden/2)
            weights = .5*(self.weights[:,:num]+self.weights[:,num:])
            self.weights[:,:num] = weights
            self.weights[:,num:] = weights
            biash = .5*(self.biash[:num]+self.biash[num:])
            self.biash[:num] = biash 
            self.biash[num:] = biash

    def trainRBM(self,eps,data):
        params = np.concatenate(self.weights.flatten(),self.biasv,self.biash)
        params = fmin_bfgs(self.flowRBM,params,self.gradFlowRBM,args=(eps,data))[0]
        num = self.n_visible*self.n_hidden
        self.weights = params[:num].reshape(self.n_visible,self.n_hidden)
        self.biasv = params[num:num+self.n_visible]
        num += self.n_visible
        self.biash = params[num:]
        self.constrainWeights()
        return (self.weights,self.biasv,self.biash)

