from __future__ import division
from __future__ import print_function
import numpy as np

class sdbm(object):
    """Simple Deep Boltzmann Machine (DBM)

    A Deep Boltzmann Machine trained with Minimum Probability 
    Flow.

    Parameters
    ----------
    n_layers : int
        Number of layers

    n_units : int
        Number of units in each layer

    weights : array-like, shape (n_layers-1, n_units, n_units), optional
        Initialize the weights

    bias : array-like, shape (n_layers, n_units), optional
        Initialize the biases

    state : array-like, shape (n_layers, n_units), optional
        Initialize the state of all units

    rng : RandomState, optional
        Random number generator to use to make results reproducible
    """
    def __init__(self,n_layers,n_units,weights=None,bias=None,state=None,rng=np.random.RandomState(23455)):
        self.rng = rng

        if weights is None:
            self.weights = rng.randn(n_layers-1,n_units,n_units)
        else:
            self.weights = weights
        if bias is None:
            self.bias = rng.randn(n_layers,n_units)
        else:
            self.bias = bias
        if state is None:
           self.state = rng.randint(2,size=(n_layers,n_units))
        else:
            self.state = state
           
        self.n_layers = n_layers
        self.n_units = n_units

    def pretrain(self,data):
        pass
    def train(self,data):
        pass

    def dedwDiff(self,state,layer,layerFlip,position):
        """Compute the negative difference between dedw for a given state and 
        dedw for a neighboring flipped state.

        Parameters
        ----------
        state : array-like, shape (n_layers, n_units)
            State of all the units

        layer : int
            Layer index for flipped state

        position : int
            Position in layer for flipped state
        """
        stateFlip = np.copy(state)
        stateFlip[layerFlip,position] = int(not state[layerFlip,position])
        return np.outer(state[layer],state[layer+1])-np.outer(stateFlip[layer],stateFlip[layer+1])
    
    def dedbDiff(self,state,layer,position):
        """Compute the nevative difference between dedb for a given state and 
        dedw for a neighboring flipped state.

        Parameters
        ----------
        state : array-like, shape (n_layers, n_units)
            State of all the units

        layer : int
            Layer index for flipped state

        position : int
            Position in layer for flipped state
        """
        stateFlip = np.copy(state)
        stateFlip[layer,position] = int(not state[layer,position])
        return state[layer]-stateFlip[layer]

    def eDiff(self,state,layerFlip,position):
        """Compute the difference between energy for a given state and 
        dedw for a neighboring flipped state.

        Parameters
        ----------
        state : array-like, shape (n_layers, n_units)
            State of all the units

        layer : int
            Layer index for flipped state

        position : int
            Position in layer for flipped state
        """
        stateFlip = np.copy(state)
        stateFlip[layerFlip,position] = int(not state[layerFlip,position])
        return self.energy(self.weights,self.bias,state)-self.energy(self.weights,self.bias,stateFlip)

    def mpfTrain(self,vis,steps,eps,stepsSample):
        """Adjust weights/biases of the network to minimize probability flow, K via
        gradient descent.

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Dataset to train on

        steps : int
            Number of iterations to run MPF (parameter updates)

        eps : float
            Learning rate

        stepsSample : int
            Number of iterations to sample from P(h|v) for the DBM
        """
        nData = vis.shape[0]
        # Propagate visible data up the network (hopefully hidden states can be considered
        # observed data)
        fullState = np.array([self.sampleHidden(vis1,stepsSample) for vis1 in vis])
        for ii in xrange(steps):
            dw = np.zeros_like(self.weights)
            db = np.zeros_like(self.bias)
            # Update weights and biases
            for state in fullState:
                # For visible
                # Gradient of flow w.r.t biases
                for kk in xrange(self.n_units):
                    db[0] += self.dedbDiff(state,0,kk)*np.exp(.5*(self.eDiff(state,0,kk)))

                # Gradient of flow w.r.t. weights
                for jj in xrange(self.n_layers-1):
                    for kk in xrange(self.n_units):
                        dw[jj] += self.dedwDiff(state,jj,jj,kk)*np.exp(.5*(self.eDiff(state,jj,kk)))
                        ep = self.eDiff(state,jj+1,kk)
                        dw[jj] += self.dedwDiff(state,jj,jj+1,kk)*np.exp(.5*(ep))
                        db[jj+1] += self.dedbDiff(state,jj+1,kk)*np.exp(.5*(ep))

            self.weights -= eps*dw/nData
            self.bias -= eps*db/nData
            
    def sampleHidden(self,vis,steps):
        """Sample from P(h|v) for the DBM via gibbs sampling for each
        layer: P(h_layer_i|h_layer_i+1, h_layer_i-1)

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Visible data to condition on

        steps : int
            Number of steps to gibbs sample
        """
        stateUp = np.copy(self.state)
        stateUp[0] = vis
        for ii in xrange(steps):
            rands = self.rng.rand(self.n_layers-1,self.n_units)
            for jj in xrange(1,self.n_layers-1):
                for kk in xrange(self.n_units):
                    # Calculate terms for weights above and weights below
                    terms = self.bias[jj,kk]+np.dot(stateUp[jj-1],self.weights[jj-1,:,kk])+np.dot(stateUp[jj+1],self.weights[jj,kk])
                    prob = 1/(1+np.exp(terms))
                    if rands[jj-1,kk] <= prob:
                        stateUp[jj,kk] = 1
                    else:
                        stateUp[jj,kk] = 0
            # Sampling for the top layer
            for kk in xrange(self.n_units):
                top = self.n_layers-1
                terms = self.bias[top,kk]+np.dot(stateUp[top-1],self.weights[top-1,:,kk])
                prob = 1/(1+np.exp(terms))
                if rands[top-1,kk] <= prob:
                    stateUp[top,kk] = 1
                else:
                    stateUp[top,kk] = 0
        return stateUp

    def sampleVisible(self,state):
        """Sample from P(v|h) of the DBM.

        Parameters
        ----------
        state : array-like, shape (n_layers, n_units)
            State of all the units
        """
        vis = np.copy(self.state[0])
        rands = self.rng.rand(self.n_units)
        for kk in xrange(self.n_units):
            terms = self.bias[0,kk]+np.dot(state[1],self.weights[0,:,kk])
            prob = 1/(1+np.exp(terms))
            if rands[kk] <= prob:
                vis[kk] = 1
            else:
                vis[kk] = 0

        return vis

    def sampleFull(self,steps):
        """Sample from P(h,v) of the DBM using Gibbs sampling.

        Parameters
        ----------
        steps : int
            Number of steps to gibbs sample
        """
        state = self.state.copy()
        for i in xrange(steps):
            vis = self.sampleVisible(state)
            state = self.sampleHidden(vis, steps)

        return state

    def energy(self,weights,bias,state):
        """Calcluate energy of a DBM

        Parameters
        ----------
        weights : array-like, shape (n_layers-1, n_units, n_units)

        bias : array-like, shape (n_layers, n_units)

        state : array-like, shape (n_layers, n_units)
        """
        ebias = np.sum([np.dot(bias[ii],state[ii]) for ii in xrange(self.n_layers)])
        eweights = np.sum([np.dot(state[ii],np.dot(weights[ii],state[ii+1])) for ii in xrange(self.n_layers-1)])
        return ebias+eweights
    
    def curEnergy(self):
        ebias = np.sum([np.dot(self.bias[ii],self.state[ii]) for ii in xrange(self.n_layers)])
        eweights = np.sum([np.dot(self.state[ii],np.dot(self.weights[ii],self.state[ii+1])) for ii in xrange(self.n_layers-1)])
        return ebias+eweights

    def getWeights(self):
        return self.weights
    def getBias(self):
        return self.bias
    def getState(self):
        return self.state
    def setWeights(self):
        pass
    def setBias(self):
        pass
    def setState(self):
        pass

