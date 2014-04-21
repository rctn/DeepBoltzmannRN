from __future__ import division
from __future__ import print_function
import numpy as np

#Auxillary Functions
#########################################3
#RBM functions for pretraining
def flowRBM(params,*args):
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
    sflow = np.sum([np.exp(.5*(energyRBM(weights,biasv,biash,state)-energyRBMBF(weights,biasv,biash,state,ii))) for ii in xrange(n_units)])
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
    for ii in xrange(n_units):
        diffew = dedwRBM(weights,biasv,biash,state)-dedwRBMBF(weights,biasv,biash,state,ii)
        diffebv = dedbvRBM(weights,biasv,biash,state)-dedbvRBMBF(weights,biasv,biash,state,ii)
        diffebh = dedbhRBM(weights,biasv,biash,state)-dedbhRBMBF(weights,biasv,biash,state,ii)
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

def dedwRBM(weights,biasv,biash,state):
    n_data = state.shape[0]
    return -np.array([np.outer(state[ii],sigm(biash+np.dot(state[ii],weights))) for ii in xrange(n_data)])

def dedbvRBM(weights,biasv,biash,state):
    return -state

def dedbhRBM(weights,biasv,biash,state):
    n_data = state.shape[0]
    return -sigm(np.tile(biash,(n_data,1))+np.dot(state,weights))

def dedwRBMBF(weights,biasv,biash,state,n):
    n_data = state.shape[0]
    flip = state.copy()
    flip[:,n] = 1-flip[:,n]
    return dedw(weights,biasv,biash,flip)

def dedbvRBMBF(weights,biasv,biash,state,n):
    flip = state.copy()
    flip[:,n] = 1-flip[:,n]
    return dedbv(weights,biasv,biash,flip)

def dedbhRBMBF(weights,biasv,biash,state,n):
    flip = state.copy()
    flip[:,n] = 1-flip[:,n]
    return dedbh(weights,biasv,biash,flip)

#########################################################
#DBM functions

def energyDiffV(weights,biases,states,layer,n):
    """Energy difference function for DBM, vectorized over different states

    Parameters
    ----------
    weights : array-like, shape (n_layers,n_units,n_units)
        Layer to layer weights

    biases : array-like, shape (n_layers,n_units)
        Biases for units

    states : array-like, shape (n_data,n_layers,n_units)
        Array of state to calculate energy for

    layer : int
        Layer for bit-flip

    n : int
        Unit for bit-flip
    """
    flip = states.copy()
    flip[:,layer,n] = 1-flip[:,layer,n]
    negative_energy = np.einsum('ijk,jk',states,biases)-np.einsum('ijk,jk',flip,biases)
    for ii in xrange(weights.shape[0]):
        #Data term
        negative_energy += np.einsum('ij,jk,ik->i',states[:,ii],weights[ii],states[:,ii+1])
        #Bit-flip term
        negative_energy -= np.einsum('ij,jk,ik->i',flip[:,ii],weights[ii],flip[:,ii+1])
    return -negative_energy

def dedwDiffV(weights,states,layer,layerF,n):
    """Calcuates the difference in the derivative of the energy
       w.r.t the weights of a given layer for a vector of states

    Parameters
    ----------
    weights : array-like, shape (n_layers,n_units,n_units)
        Layer to layer weights

    states : array-like, shape (n_data,n_layers,n_units)
        Array of states to calculate energy for

    layer : int
        Which layer the weights are from

    layerF : int
        Which layer the bit-flip is in

    n : int
        Which unit to bit-flip
    """
    flip = states.copy()
    flip[:,layerF,n] = 1-flip[:,layerF,n]
    return -(np.einsum('ij,ik->ijk',states[:,layer],states[:,layer+1])-np.einsum('ij,ik->ijk',flip[:,layer],flip[:,layer+1]))

def dedbDiffV(states,layer,n):
    """Calcuates the derivative of the energy w.r.t the biases of a given layer

    Parameters
    ----------
    states : array-like, shape (n_data,n_layers,n_units)
        Array of states to calculate energy for

    layer : int
        Which layer the weights are from

    n : int
        Unit to bit-flip
    """
    flip = states.copy()
    flip[:,layer,n] = 1-flip[:,layer,n]
    return -(states[:,layer]-flip[:,layer])

def energy(weights,bias,state):
    """Calcluate energy of a DBM

    Parameters
    ----------
    weights : array-like, shape (n_layers-1, n_units, n_units)

    bias : array-like, shape (n_layers, n_units)

    state : array-like, shape (n_layers, n_units)
    """
    negative_energy = (bias*state).sum()
    for ii in xrange(weights.shape[0]):
        negative_energy += state[ii].dot(weights[ii].dot(state[ii+1]))

    return -negative_energy

def dedbDiff(state,layer,position):
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

def dedwDiff(state,layer,layerFlip,position):
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
            self.weights = rng.uniform(low=-4 * np.sqrt(6. / (n_layers*n_units)),
                                        high=4 * np.sqrt(6. / (n_layers*n_units)),
                                        size=(n_layers-1,n_units,n_units))
        else:
            self.weights = weights
        if bias is None:
            self.bias = rng.uniform(low=-4 * np.sqrt(6. / (n_layers*n_units)),
                                    high=4 * np.sqrt(6. / (n_layers*n_units)),
                                    size=(n_layers,n_units))
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
        return energy(self.weights,self.bias,state)-energy(self.weights,self.bias,stateFlip)

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
                    db[0] += dedbDiff(state,0,kk)*np.exp(.5*(self.eDiff(state,0,kk)))

                # Gradient of flow w.r.t. weights
                for jj in xrange(self.n_layers-1):
                    for kk in xrange(self.n_units):
                        dw[jj] += dedwDiff(state,jj,jj,kk)*np.exp(.5*(self.eDiff(state,jj,kk)))
                        ep = self.eDiff(state,jj+1,kk)
                        dw[jj] += dedwDiff(state,jj,jj+1,kk)*np.exp(.5*(ep))
                        db[jj+1] += dedbDiff(state,jj+1,kk)*np.exp(.5*(ep))

            self.weights -= eps*dw/nData
            self.bias -= eps*db/nData

    def ExTrain(self,vis,steps,eps,meanSteps,intOnly=False):
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

        intOnly : boolean, optional
            Round mean-field values to binary
        """
        nData = vis.shape[0]
        # Propagate visible data up the network (hopefully hidden states can be considered
        # observed data)

        #Find meanfield estimates
        fullStates = self.ExHidden(vis,meanSteps)
        if intOnly:
            fullStates = np.around(fullStates)

        for ii in xrange(steps):
            dw = np.zeros_like(self.weights)
            db = np.zeros_like(self.bias)
            for layer_i in xrange(self.n_layers):
                for unit_i in xrange(self.n_units):
                    originalState = fullStates[:,layer_i,unit_i]
                    flippedState = 1.-fullStates[:,layer_i,unit_i]

                    diffe = -originalState*self.bias[layer_i,unit_i]+flippedState*self.bias[layer_i,unit_i]

                    # All layers except top
                    if layer_i < (self.n_layers-1):
                        W_h = self.weights[layer_i,unit_i].dot(fullStates[:,layer_i+1].T)
                        vT_W_h = originalState*W_h
                        vfT_W_h = flippedState*W_h
                        diffe += -vT_W_h+vfT_W_h

                    # All layers except bottom (visible)
                    if layer_i > 0:
                        vT_W = fullStates[:,layer_i-1].dot(self.weights[layer_i-1,:,unit_i])
                        vT_W_h = vT_W*originalState
                        vT_W_hf = vT_W*flippedState
                        diffe += -vT_W_h+vT_W_hf
                    # import ipdb; ipdb.set_trace()
                    diffe = np.exp(.5*(diffe))

                    # Bias update
                    diffeb = -originalState+flippedState
                    db[layer_i,unit_i] += diffeb.dot(diffe)

                    # Weights update
                    if layer_i < (self.n_layers-1):
                        diffew = -np.einsum('i,ij->ij',originalState,fullStates[:,layer_i+1])+ \
                                    np.einsum('i,ij->ij',flippedState,fullStates[:,layer_i+1])
                        dw[layer_i,unit_i] += np.einsum('i,ij->j',diffe,diffew)

                    if layer_i > 0:
                        diffew = -np.einsum('ij,i->ij',fullStates[:,layer_i-1],originalState)+ \
                                    np.einsum('ij,i->ij',fullStates[:,layer_i-1],flippedState)
                        dw[layer_i-1,:,unit_i] += np.einsum('i,ij->j',diffe,diffew)
            
            self.weights -= eps*dw/float(nData)
            self.bias -= eps*db/float(nData)

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
            # Find activations for internal layers going backwards
            for jj in xrange(self.n_layers-2,0,-1):
                # Apply mean field equations
                terms = np.tile(self.bias[jj],(vis.shape[0],1))+np.dot(curState[:,jj-1],self.weights[jj-1])+np.dot(curState[:,jj+1],self.weights[jj])
                curState[:,jj] = 1./(1+np.exp(-terms))

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
            state = self.sampleHidden(vis,1)

        return state
    
    def curEnergy(self):
        """Calculate current energy of DBM
        """
        return energy(self.weights,self.bias,self.state)

    def scoreSamples(self, vis, n_samples, steps):
        """Evaluate the fitness of the model for a given dataset. Calculate
        the expectation of P(v|h) with respect to P(h) as a proxy for an
        unormalized P(v).

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Dataset to evaluate fitness on

        n_samples : int
            Number of samples to draw to estimate expectation

        steps : int
            Number of steps to gibbs sample
        """
        average_p_v = 0.
        
        for i in xrange(n_samples):
            # Sample P(h)
            state = self.sampleFull(steps)

            # Calculate Normalization
            log_z = 0.
            K = (self.weights.dot(state[1]) + self.bias[0]).T
            for k in xrange(K.shape[0]):
                log_z += np.log(1+np.exp(K[k]))
            log_z += self.bias[1].dot(state[1])
            for j in xrange(vis.shape[0]):  
                # Set to P(v|h)
                state[0] = vis[j]
                # Normalized P(v|h)
                p_v = np.exp(-energy(self.weights, self.bias, state)-log_z)
                average_p_v += p_v

        average_p_v /= n_samples*vis.shape[0]
        return average_p_v

    def getWeights(self):
        return self.weights
    def getBiases(self):
        return self.bias
    def getState(self):
        return self.state
    def setWeights(self,weights):
        if self.weights.shape == weights.shape:
            self.weights.shape = weights
        pass
    def setBiases(self):
        pass
    def setState(self):
        pass

