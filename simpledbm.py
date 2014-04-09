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

    def pretrain(self,vis,steps,eps):
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
        stateFlip[layerFlip,position] = int(not round(state[layerFlip,position]))
        return -np.outer(state[layer],state[layer+1])+np.outer(stateFlip[layer],stateFlip[layer+1])
    
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
        stateFlip[layer,position] = int(not round(state[layer,position]))
        return -state[layer]+stateFlip[layer]

    def eDiff(self,state,layerFlip,position):
        """Compute the difference between energy for a given state and 
        energy for a neighboring flipped state.

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
        stateFlip[layerFlip,position] = int(not round(state[layerFlip,position]))
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

        # Gibbs Sample hidden units
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

            
    def ExTrain(self,vis,steps,eps,meanSteps,updateSteps):
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

        meanSteps : int
            Number of mean-field cycles per layer

        updateSteps : int
            Number of mean-field cycles for whole network
        """
        nData = vis.shape[0]
        # Propagate visible data up the network (hopefully hidden states can be considered
        # observed data)

        #Find meanfield estimates
        fullState = np.around(self.ExHidden(vis,meanSteps,updateSteps))
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

    def ExHidden(self,vis,meanSteps,updateSteps):
        """Finds Expectation for hidden units using mean-field variational approach

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Visible data to condition on

        meanSteps : int
            Number of mean-field steps to cycle through

        updateSteps : int
            Number of times to run through layers. Must be >=2 for any top down feedback
        """
        curState = np.zeros((vis.shape[0],self.n_layers,self.n_units))
        # Initialize state to visible and zeros
        for ii in xrange(vis.shape[0]):
            curState[ii,0] = vis[ii]
        for ii in xrange(updateSteps):
            # Find activations for internal layers
            for jj in xrange(1,self.n_layers-1):
                # Apply mean field equations
                for kk in xrange(meanSteps):
                    terms = np.tile(self.bias[jj],(vis.shape[0],1))+np.dot(curState[:,jj-1],self.weights[jj-1])+np.dot(curState[:,jj+1],self.weights[jj])
                    curState[:,jj] = 1./(1+np.exp(-terms))
            # Find activation for top layer
            # Apply mean field equations
            for kk in xrange(meanSteps):
                terms = np.tile(self.bias[self.n_layers-1],(vis.shape[0],1))+np.dot(curState[:,self.n_layers-2],self.weights[self.n_layers-2])
                curState[:,self.n_layers-1] = 1./(1+np.exp(-terms))
        return curState

        
            
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
                terms = self.bias[jj] + self.weights[jj-1].dot(stateUp[jj-1]) + self.weights[jj].T.dot(stateUp[jj+1])
                probs = 1/(1+np.exp(-terms))
                stateUp[jj] = rands[jj-1] <= probs

            # Sampling for the top layer
            top = self.n_layers-1
            terms = self.bias[top] + self.weights[top-1].dot(stateUp[top-1])
            probs = 1/(1+np.exp(-terms))
            stateUp[top] = rands[top-1] <= probs

        return stateUp

    def sampleVisible(self,state):
        """Sample from P(v|h) of the DBM.

        Parameters
        ----------
        state : array-like, shape (n_layers, n_units)
            State of all the units
        """
        vis = np.empty_like(self.state[0])
        rands = self.rng.rand(self.n_units)
        terms = self.bias[0] + self.weights[0].T.dot(state[1])
        probs = 1/(1+np.exp(-terms))
        vis = rands <= probs

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
        negative_energy = (bias*state).sum()
        for ii in xrange(self.n_layers-1):
            negative_energy += state[ii].dot(weights[ii].dot(state[ii+1]))

        return -negative_energy
    
    def curEnergy(self):
        """Calculate current energy of DBM
        """
        return self.energy(self.weights,self.bias,self.state)

    def scoreSamples(self, vis):
        """Evaluate the fitness of the model for a given dataset. Calculate
        the expectation of P(v|h) with respect to P(h) as a proxy for an
        unormalized P(v).

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Dataset to evaluate fitness on.
        """
        p_v = 0.
        n_hsamples = 10
        for i in range(vis.shape[0]):
            for j in range(n_hsamples):
                # Sample P(h)
                state = self.sampleFull(20)
                # Set to P(v|h)
                state[0] = vis[i]
                p_v += np.exp(-self.energy(self.weights, self.bias, state))

        return p_v/(vis.shape[0]*n_hsamples)

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

