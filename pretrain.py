from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
from numba import autojit

def flow(params,*args):
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
    sflow = iterEnergy(weights,biasv,biash,data,n_visible)
    #np.sum([np.exp(.5*(energyDiff(weights,biasv,biash,data,ii))) for ii in xrange(n_visible)])
    k = eps*sflow/n_data
    return k

  @autojit
  def iterEnergy(weights,biasv,biash,data,n_visible):
    result = 0.
    for ii in xrange(n_visible):
        result+= np.exp(.5*(energyDiff(weights,biasv,biash,data,ii)))
    return result

def gradFlow(params,*args):
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
    dkdw,dkdbv,dkdbh = iterde(weights,biasv,biash,data,n_visible)
    return eps*np.concatenate((dkdw.flatten(),dkdbv,dkdbh))/n_data

  @autojit
  def iterde(weights,biasv,biash,data,n_visible):
    dkdw = np.zeros_like(weights)
    dkdbv = np.zeros_like(biasv)
    dkdbh = np.zeros_like(biash)
    for ii in xrange(n_visible):
	diffe = np.exp(.5*energyDiff(weights,biasv,biash,data,ii))
        dkdw += np.einsum('ijk,i->jk',dedwDiff(weights,biash,data,ii),diffe)
        dkdbv[ii] += np.dot(1.-2.*data[:,ii],diffe)
        dkdbh += np.einsum('ij,i->j',dedbhDiff(weights,biash,data,ii),diffe)
    return (dkdw,dkdbv,dkdbh)


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
    flip[:,n] = 1.-flip[:,n]
    logTerm = np.sum(np.log(1.+np.exp(biash+data.dot(weights))),axis=1)
    logTermBF = np.sum(np.log(1.+np.exp(biash+flip.dot(weights))),axis=1)
    return (-data[:,n]*biasv[n]-logTerm)-(-flip[:,n]*biasv[n]-logTermBF)

def dedwDiff(weights,biash,data,n):
    n_data = data.shape[0]
    flip = data.copy()
    flip[:,n] = 1.-flip[:,n]
    terms = biash+data.dot(weights)
    termsBF = terms+(1.-2*np.outer(data[:,n],weights[n,:]))
    dedw = -np.einsum('ij,ik->ijk',data,sigm(terms))
    dedwBF = -np.einsum('ij,ik->ijk',flip,sigm(termsBF))
    return dedw-dedwBF

def dedbvDiff(data,n):
    flip = data.copy()
    flip[:,n] = 1-flip[:,n]
    return (-data)-(-flip)

def dedbhDiff(weights,biash,data,n):
    n_data = data.shape[0]
    terms = biash+data.dot(weights)
    termsBF = terms+(1.-2*np.outer(data[:,n],weights[n,:]))
    dedbh = -sigm(terms)
    dedbhBF = -sigm(termsBF)
    return dedbh-dedbhBF

def sigm(x):
    """Sigmoid function

    Parameters
    ----------
    x : array-like
        Array of elements to calculate sigmoid for.
    """
    return 1./(1.+np.exp(-x))

class preRBM(object):
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
        self.weights = rng.uniform(low=-4.*np.sqrt(6./(self.n_visible+self.n_hidden)),
                                   high=4.*np.sqrt(6./(self.n_visible+self.n_hidden)),
                                   size=(self.n_visible,self.n_hidden))
        self.biasv = rng.uniform(low=-4.*np.sqrt(6./(self.n_visible+self.n_hidden)),
                                   high=4.*np.sqrt(6./(self.n_visible+self.n_hidden)),
                                   size=self.n_visible)
        self.biash = rng.uniform(low=-4.*np.sqrt(6./(self.n_visible+self.n_hidden)),
                                   high=4.*np.sqrt(6./(self.n_visible+self.n_hidden)),
                                   size=self.n_hidden)
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

    def train(self,eps,data):
        if self.layer == 'bottom':
            states = np.tile(data,(1,2))
        else:
            states = data
        params = np.concatenate((self.weights.flatten(),self.biasv,self.biash))
        params = minimize(flow,params,method='L-BFGS-B',jac=gradFlow,args=(eps,states,self.n_visible,self.n_hidden)).x
        num = self.n_visible*self.n_hidden
        self.weights = params[:num].reshape(self.n_visible,self.n_hidden)
        self.biasv = params[num:num+self.n_visible]
        num += self.n_visible
        self.biash = params[num:]
        self.constrainWeights()
        if self.layer == 'bottom':
            weights = self.weights[:int(self.n_visible/2)]
            biasv = self.biasv[:int(self.n_visible/2)]
            biash = self.biash
        elif self.layer == 'middle':
            weights = self.weights/2.
            biasv = self.biasv
            biash = self.biash
        elif self.layer == 'top':
            weights = self.weights[:,:int(self.n_hidden/2)]
            biasv = self.biasv
            biash = self.biash[:int(self.n_hidden/2)]
        else:
            raise ValueError
        return (weights,biasv,biash)

    def trainStep(self,eps,data,steps):
        dkdw = np.zeros_like(self.weights)
        dkdbv = np.zeros_like(self.biasv)
        dkdbh = np.zeros_like(self.biash)
        n_data = data.shape[0]
        for jj in xrange(steps):
            for ii in xrange(self.n_visible):
	        diffe = np.exp(.5*energyDiff(self.weights,self.biasv,self.biash,data,ii))
                dkdw += np.einsum('ijk,i->jk',dedwDiff(self.weights,self.biash,data,ii),diffe)
                #dkdbv += np.einsum('ij,i->j',dedbvDiff(data,ii),diffe)
                dkdbv[ii] += np.dot(1.+2*data[:,ii],diffe)
                dkdbh += np.einsum('ij,i->j',dedbhDiff(self.weights,self.biash,data,ii),diffe)
            self.weights -= eps*dkdw/float(n_data)
            self.biasv -= eps*dkdbv/float(n_data)
            self.biash -= eps*dkdbh/float(n_data)

        self.constrainWeights()
        if self.layer == 'bottom':
            weights = self.weights[:int(self.n_visible/2)]
            biasv = self.biasv[:int(self.n_visible/2)]
            biash = self.biash
        elif self.layer == 'middle':
            weights = self.weights/2.
            biasv = self.biasv
            biash = self.biash
        elif self.layer == 'top':
            weights = self.weights[:,:int(self.n_hidden/2)]
            biasv = self.biasv
            biash = self.biash[:int(self.n_hidden/2)]
        else:
            raise ValueError
        return (weights,biasv,biash)

    def nextActivation(self,data,steps):
        n_data = data.shape[0]
        if self.layer == 'bottom':
            n_units = self.n_hidden
            weights = self.weights[:int(self.n_visible/2)]
            biash = self.biash
        elif self.layer == 'middle':
            n_units = self.n_visible
            weights = self.weights/2.
        elif self.layer == 'top':
            n_units = self.n_visible
            weights = self.weights[:,:int(self.n_hidden/2)]
            biash = self.biash[:int(self.n_hidden/2)]
        else:
            raise ValueError
        hidden_state = rng.randn(n_data,n_units)
        for ii in xrange(steps):
            terms = np.tile(bias,(n_data,1))+data.dot(weights)
        return sigm(terms)


    
