from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.optimize import minimize

def flowRBM(params,*args):
    """MPF objective function for RBM. Used to pretrain DBM layers.

    Parameters
    ----------
    params : array-like, shape (n_weights+n_biasv+nbiash)
        Weights (flattened), visible biases, and hidden biases.
        
    *args : tuple or list or args
        Expects eps which is coefficient for MPF and data which is a
        vectors of visible states to learn on.
    """
    eps = args[0]
    data = args[1]
    n_visible = args[2]
    n_hidden = args[3]
    n_data = data.shape[0]
    num = n_visible*n_hidden
    weights = params[:num].reshape((n_visible,n_hidden))
    biasv = params[num:num+n_visible]
    num += n_visible
    biash = params[num:num+n_hidden]
    sflow = np.sum([np.exp(.5*(energyDiff(weights,biasv,biash,data,ii))) for ii in xrange(n_visible)])
    k = eps*sflow/n_data
    return k

def gradFlowRBM(params,*args):
    """Gradient of MPF objective function for RBM. Used to pretrain DBM layers.

    Parameters
    ----------
    params : array-like, shape (n_units^2+2*n_units)
        Weights (flattened), visible biases, and hidden biases.

    *args : tuple or list or args
        Expects eps which is coefficient for MPF and data which is a
	vectors of visible states to learn on.
    """
    eps = args[0]
    data = args[1]
    n_visible = args[2]
    n_hidden = args[3]
    n_data = data.shape[0]
    num = n_visible*n_hidden
    weights = params[:num].reshape((n_visible,n_hidden))
    biasv = params[num:num+n_visible]
    num += n_visible
    biash = params[num:num+n_hidden]
    dkdw = np.zeros_like(weights)
    dkdbv = np.zeros_like(biasv)
    dkdbh = np.zeros_like(biash)
    for ii in xrange(n_visible):
	diffe = np.exp(.5*energyDiff(weights,biasv,biash,data,ii))
	dkdw += np.dot(np.transpose(dedwDiff(weights,biash,data,ii),axes=(1,2,0)),diffe)
	dkdbv += np.dot(dedbvDiff(data,ii).T,diffe)
	dkdbh += np.dot(dedbhDiff(weights,biash,data,ii).T,diffe)
    return eps*np.concatenate((dkdw.flatten(),dkdbv,dkdbh))/n_data


def energyDiff(weights,biasv,biash,data,n):
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

def dedwDiff(weights,biash,data,n):
    n_data = data.shape[0]
    flip = data.copy()
    flip[:,n] = 1-flip[:,n]
    dedw = -np.array([np.outer(data[ii],sigm(biash+data[ii].dot(weights))) for ii in xrange(n_data)])
    dedwBF = -np.array([np.outer(flip[ii],sigm(biash+flip[ii].dot(weights))) for ii in xrange(n_data)])
    return dedw-dedwBF

def dedbvDiff(data,n):
    flip = data.copy()
    flip[:,n] = 1-flip[:,n]
    return (-data)-(-flip)

def dedbhDiff(weights,biash,data,n):
    n_data = data.shape[0]
    flip = data.copy()
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

    def __init__(self,n_visible,n_hidden,layer,rng):
        if layer == 'bottom':
            self.n_visible = 2*n_visible
            self.n_hidden = n_hidden
        elif layer == 'middle':
            self.n_visible = n_visible
            self.n_hidden = n_hidden
        elif layer == 'top':
            self.n_visible = n_visible
            self.n_hidden = 2*n_hidden
        else:
            raise ValueError
        self.layer = layer
        self.rng = rng
        self.weights = rng.randn(self.n_visible,self.n_hidden)
        self.biasv = rng.randn(self.n_visible)
        self.biash = rng.randn(self.n_hidden)
        self.constrainWeights()
    
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
        if self.layer == 'bottom':
            states = np.tile(data,(1,2))
        else:
            states = data
        params = np.concatenate((self.weights.flatten(),self.biasv,self.biash))
        params = minimize(flowRBM,params,jac=gradFlowRBM,args=(eps,states,self.n_visible,self.n_hidden)).x
        num = self.n_visible*self.n_hidden
        self.weights = params[:num].reshape(self.n_visible,self.n_hidden)
        self.biasv = params[num:num+self.n_visible]
        num += self.n_visible
        self.biash = params[num:]
        self.constrainWeights()
        return (self.weights,self.biasv,self.biash)

