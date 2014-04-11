from __future__ import division
from __future__ import print_function
import numpy as np

#Auxillary Functions
def flow(params,*args):
    """MPF objective function for RBM. Used to pretrain DBM layers.

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
    n_total = params.shape[0]
    n_units = int(np.sqrt(1+n_total)-1)
    n_data = state.shape[0]
    weights = params[:n_units**2].reshape((n_units,n_units))
    biasv = params[n_units**2:n_units**2+n_units]
    biash = params[n_units**2+n_units:n_units**2+2*n_units]
    sflow = np.sum([np.exp(.5*(energy(weights,biasv,biash,state)-energyBF(weights,biasv,biash,state,ii))) for ii in xrange(n_units)])
    k = eps*sflow/n_data
    return k

def gradFlow(params,*args):
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
    for ii in xrange(n_units):
        diffew = dedw(weights,biasv,biash,state)-dedwBF(weights,biasv,biash,state,ii)
        diffebv = dedbv(weights,biasv,biash,state)-dedbvBF(weights,biasv,biash,state,ii)
        diffebh = dedbh(weights,biasv,biash,state)-dedbhBF(weights,biasv,biash,state,ii)
        diffe = np.exp(.5*(energyRBM(weights,biasv,biash,state)-energyRBMBF(weights,biasv,biash,state,ii)))
        dkdw += np.dot(np.transpose(diffew,axes=(1,2,0)),diffe)
        dkdbv += np.dot(diffebv.T,diffe)
        dkdbh += np.dot(diffebh.T,diffe)
    return eps*np.concatenate((dkdw.flatten(),dkdbv,dkdbh))/n_data

def sigm(x):
    """Sigmoid function

    Parameters
    ----------
    x : array-like
        Array of elements to calculate sigmoid for.
    """
        return 1./(1+np.exp(-x))

def energyRBM(weights,biasv,biash,state):
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
        logTerm = np.sum(np.log(1.+np.exp(biash+np.dot(state,weights))),axis=1)
            return -np.dot(state,biasv)-logTerm

def energyRBMBF(weights,biasv,biash,state,n):
        flip = state.copy()
            flip[:,n] = 1-flip[:,n]
                return energy(weights,biasv,biash,flip)

def dedw(weights,biasv,biash,state):
        n_data = state.shape[0]
            return -np.array([np.outer(state[ii],sigm(biash+np.dot(state[ii],weights))) for ii in xrange(n_data)])
def dedbv(weights,biasv,biash,state):
        return -state
def dedbh(weights,biasv,biash,state):
        n_data = state.shape[0]
            return -sigm(np.tile(biash,(n_data,1))+np.dot(state,weights))

def dedwBF(weights,biasv,biash,state,n):
        n_data = state.shape[0]
            flip = state.copy()
                flip[:,n] = 1-flip[:,n]
                    return dedw(weights,biasv,biash,flip)
def dedbvBF(weights,biasv,biash,state,n):
        flip = state.copy()
            flip[:,n] = 1-flip[:,n]
                return dedbv(weights,biasv,biash,flip)
def dedbhBF(weights,biasv,biash,state,n):
        flip = state.copy()
            flip[:,n] = 1-flip[:,n]
                return dedbh(weights,biasv,biash,flip)


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

            
    def ExTrain(self,vis,steps,eps,meanSteps,intOnly = None):
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
        if intOnly is None:
            intOnly = False
        nData = vis.shape[0]
        # Propagate visible data up the network (hopefully hidden states can be considered
        # observed data)

        #Find meanfield estimates
        fullState = np.around(self.ExHidden(vis,meanSteps))
        if intOnly:
            fullState = np.around(fullState)
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

    def ExHidden(self,vis,meanSteps):
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
        for ii in xrange(meanSteps):
            # Find activations for internal layers
            for jj in xrange(1,self.n_layers-1):
                # Apply mean field equations
                terms = np.tile(self.bias[jj],(vis.shape[0],1))+np.dot(curState[:,jj-1],self.weights[jj-1])+np.dot(curState[:,jj+1],self.weights[jj])
                curState[:,jj] = 1./(1+np.exp(-terms))
            # Find activation for top layer
            # Apply mean field equations
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

        TODO: Recycle P(v,h) samples to estimate partition and reuse the set 
        of P(h) samples for all visible datapoints. Recycling and reuse is
        important for the environment and necessary because partition function 
        Z(W,b) changes when parameter are updated.

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Dataset to evaluate fitness on.
        """
        average_p_v = 0.
        n_hsamples = 100
        
        #Normalization Constant
        z = 0.

        for i in xrange(n_hsamples):
            # Sample P(h)
            state = self.sampleFull(20)
            for j in xrange(vis.shape[0]):
                K = (self.weights.dot(state[1]) + self.bias[0]).T
                # Set to P(v|h)
                state[0] = vis[j]

                # Unormalized P(v|h)
                p_v = np.exp(-self.energy(self.weights, self.bias, state))

                # Divide P(v|h) by normalization
                for k in xrange(K.shape[0]):
                    p_v /= 1+np.exp(K[k])
                p_v /= np.exp(self.bias[1].dot(state[1]))

                average_p_v += p_v

        average_p_v /= n_hsamples*vis.shape[0]

        return average_p_v

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

