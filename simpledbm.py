from __future__ import division
from __future__ import print_function
import pretrain
import numpy as np


def sigm(x):
    """Sigmoid function

    Parameters
    ----------
    x : array-like
        Array of elements to calculate sigmoid for.
    """
    return 1./(1.+np.exp(-x))

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

        self.meanState = None

           
        self.n_layers = n_layers
        self.n_units = n_units

    def pretrain(self,vis,eps):
        """Trains each layer with a modified RBM

        Parameters
        ----------
        vis : array-like, shape (n_data,n_units)
            Array of training data

        eps : float
            coefficient for MPF
        """
        # Train bottom layer
        rbm = pretrain.rbm(self.n_units,self.n_units,'bottom',self.rng)
        weights,biasv,biash = rbm.train(eps,vis)
        self.weights[0] = weights
        self.bias[0] = biasv
        self.bias[1] = biash
        newStates = rbm.nextActivation(vis)
        # Train middle layers
        if self.n_layers > 2:
            for ii in xrange(1,self.n_layers-2):
                rbm = pretrain.rbm(self.n_units,self.n_units,'middle',self.rng)
                weights,biasv,biash = rbm.train(eps,newStates)
                self.weights[ii] = weights
                self.bias[ii+1] = biasv
                newStates = rbm.nextActivation(newStates)
        # Train top layer
        rbm = pretrain.rbm(self.n_units,self.n_units,'top',self.rng)
        weights,biasv,biash = rbm.train(eps,newStates)
        self.weights[self.n_layers-1] = weights
        self.bias[self.n_layers] = biash

    def ExTrain(self,vis,steps,eps,meanSteps):
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
        """
        nData = vis.shape[0]
        # Propagate visible data up the network (hopefully hidden states can be considered
        # observed data)
        #Find meanfield estimates
        muStates = self.ExHidden(vis,meanSteps)

        ####
        # meanHStates = muStates.mean(0)[1:]
        # tmask = (meanHStates > .9) + (meanHStates < .1)
        # tmask = np.tile(tmask,(nData,1))
        # tmask = (self.rng.uniform(size=tmask.shape) < .1) & tmask

        # hstemp = muStates[:,1:,:]
        # hstemp[tmask[:,np.newaxis,:]] = 0.5
        # muStates[:,:1,:] = hstemp

        # tmask = tmask.ravel()
        # reset = ((self.rng.uniform(size=(muStates.shape[0], tmask.shape[0])) < 0.1) & tmask.reshape((1,-1))).ravel()
        # fullStatesShape = muStates.shape
        # muStates = muStates.reshape((-1,))

        # muStates[reset] = 0.7
        # muStates = muStates.reshape((fullStatesShape))

        ###

        for ii in xrange(steps):
            dw = np.zeros_like(self.weights)
            db = np.zeros_like(self.bias)
            for layer_i in xrange(self.n_layers):
                diffeL = np.tile(-self.bias[layer_i], (nData,1))
                diffeU = np.tile(-self.bias[layer_i], (nData,1))
                # All layers except top
                if layer_i < (self.n_layers-1):
                    W_h = self.weights[layer_i].dot(muStates[:,layer_i+1].T).T
                    diffeL += -W_h
                # All layers except bottom (visible)
                if layer_i > 0:
                    vT_W = muStates[:,layer_i-1].dot(self.weights[layer_i-1])
                    diffeU += -vT_W
                
                # Bias update
                diffebL = -muStates[:,layer_i]*np.exp(.5*diffeL) + (1.-muStates[:,layer_i])*np.exp(-.5*diffeL)
                db[layer_i] += diffebL.sum(0)

                # Weights update
                # All layers except top
                if layer_i < (self.n_layers-1):
                    dkdw = np.einsum('ij,ik->jk',diffebL,muStates[:,layer_i+1])
                    dw[layer_i] += dkdw
                # All layers except bottom (visible)
                if layer_i > 0:
                    diffebU = -muStates[:,layer_i]*np.exp(.5*diffeU) + (1.-muStates[:,layer_i])*np.exp(-.5*diffeU)
                    dkdw = np.einsum('ij,ik->jk',muStates[:,layer_i-1],diffebU)
                    dw[layer_i-1] += dkdw

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
            Number of times to run through layers.
        """
        # Initialize state to visible
        if (self.meanState is None) or (self.meanState.shape[0] != vis.shape[0]):
            self.meanState = np.zeros(shape=(vis.shape[0],self.n_layers,self.n_units))+.5
        self.meanState[:,0] = vis
        for ii in xrange(meanSteps):
            # Find activations for internal layers
            for jj in xrange(1,self.n_layers-1):
                # Apply mean field equations
                terms = np.tile(self.bias[jj],(vis.shape[0],1))+self.meanState[:,jj-1].dot(self.weights[jj-1])+self.weights[jj].dot(self.meanState[:,jj+1].T).T
                self.meanState[:,jj] = sigm(terms)
            # Find activation for top layer
            # Apply mean field equations
            terms = np.tile(self.bias[self.n_layers-1],(vis.shape[0],1))+self.meanState[:,self.n_layers-2].dot(self.weights[self.n_layers-2])
            self.meanState[:,self.n_layers-1] = sigm(terms)
            # Find activations for internal layers going backwards
            for jj in xrange(self.n_layers-2,0,-1):
                # Apply mean field equations
                terms = np.tile(self.bias[jj],(vis.shape[0],1))+self.meanState[:,jj-1].dot(self.weights[jj-1])+self.weights[jj].dot(self.meanState[:,jj+1].T).T
                self.meanState[:,jj] = sigm(terms)

        return self.meanState
            
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
            rands = self.rng.rand(2,self.n_layers-1,self.n_units)
            # Sample bottom layers going up
            for jj in xrange(1,self.n_layers-1):
                terms = self.bias[jj] + stateUp[jj-1].dot(self.weights[jj-1]) + self.weights[jj].dot(stateUp[jj+1])
                probs = sigm(terms)
                stateUp[jj] = rands[0,jj-1] <= probs

            # Sampling for the top layer, before going back down
            top = self.n_layers-1
            terms = self.bias[top] + stateUp[top-1].dot(self.weights[top-1])
            probs = sigm(terms)
            stateUp[top] = rands[1,top-1] <= probs

            # Sample bottom hidden layers going down
            for jj in xrange(self.n_layers-2,0,-1):
                terms = self.bias[jj] + stateUp[jj-1].dot(self.weights[jj-1]) + self.weights[jj].dot(stateUp[jj+1])
                probs = sigm(terms)
                stateUp[jj] = rands[1,jj-1] <= probs

        return stateUp

    def sampleVisible(self,state):
        """Sample from P(v|h) of the DBM.

        Parameters
        ----------
        state : array-like, shape (n_layers, n_units)
            State of all the units
        """
        rands = self.rng.rand(self.n_units)
        terms = self.bias[0] + self.weights[0].dot(state[1])
        probs = sigm(terms)
        vis = rands <= probs

        return vis

    def sampleFull(self,vis,steps):
        """Sample from P(h,v) of the DBM using Gibbs sampling.

        Parameters
        ----------
        vis : array-like, shape (n_units)
            Visible data to initially condition on during Gibbs
            sampling

        steps : int
            Number of steps to gibbs sample
        """
        for i in xrange(steps):
            state = self.sampleHidden(vis,3)
            vis = self.sampleVisible(state)
        state[0] = vis
        return state

    def generateConfabulations(self,vis,n_burn,n_keep,steps):
        """Generate n_keep confabulations after n_burn burn-in for the visible layer.
           Output is P(v|h), from sampling, not a sample of v.

        Parameters
        ----------
        vis : array-like, shape (n_units)
            Visible data to initially condition on during Gibbs
            sampling

        n_burn : int
            Number of samples to throw out

        n_keep : int
            number of samples to keep

        steps : int
            Number of steps to gibbs sample
        """
        confabs = np.zeros(shape=(n_keep,self.n_units))
        for ii in xrange(n_burn):
            vis = self.sampleFull(vis,steps)[0]
        for ii in xrange(n_keep):
            state = self.sampleFull(vis,steps)
            vis = state[0]
            terms = self.bias[0]+np.dot(self.weights[0],state[1])
            confabs[ii] = sigm(terms)
        return confabs
    
    def curEnergy(self):
        """Calculate current energy of DBM
        """
        return energy(self.weights,self.bias,self.state)


    def flowSamples(self, vis, meanSteps, intOnly=False):
        """Calculate the probability flow K for a given dataset up to
        a factor epsilon (KL divergence between data distribution
        and distribution after an infinitesimal time).

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Dataset to compute flow on

        meanSteps : int
            Number of mean-field cycles per layer

        intOnly : boolean, optional
            Round mean-field values to binary
        """
        nData = vis.shape[0]

        #Find meanfield estimates
        fullStates = self.ExHidden(vis,meanSteps)
        if intOnly:
            fullStates = np.around(fullStates)

        flows = 0.
        for layer_i in xrange(self.n_layers):
            originalState = fullStates[:,layer_i,:]
            flippedState = 1.-fullStates[:,layer_i,:]

            diffe = (-originalState+flippedState)*self.bias[layer_i]
            # All layers except top
            if layer_i < (self.n_layers-1):
                W_h = self.weights[layer_i].dot(fullStates[:,layer_i+1].T).T
                diffe += (-originalState+flippedState)*W_h
            # All layers except bottom (visible)
            if layer_i > 0:
                vT_W = fullStates[:,layer_i-1].dot(self.weights[layer_i-1])
                diffe += vT_W*(-originalState+flippedState)
                
            diffe = np.exp(.5*(diffe))
            flows += diffe.sum()

        return flows/nData

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
            self.weights = weights
        else:
            raise ValueError
    def setBiases(self,bias):
        if self.bias.shape == bias.shape:
            self.bias = bias
        else:
            raise ValueError
    def setState(self,state):
        if self.state.shape == state.shape:
            self.state = state
        else:
            raise ValueError

