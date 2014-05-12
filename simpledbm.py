from __future__ import division
from __future__ import print_function
import pretrain
import numpy as np
import copy

def sigm(x):
    """Sigmoid function

    Parameters
    ----------
    x : array-like
        Array of elements to calculate sigmoid for.
    """
    return 1./(1.+np.exp(-x))

def list_zeros_like(l):
    """Initialize a list of arrays with zeros like the
    arrays in a given list.

    Parameters
    ----------
    l : list
        List of arrays
    """
    new_l = []
    for a in l:
        new_l.append(np.zeros_like(a))

    return new_l

def flow(init_W,init_b,nData):
    import theano
    import theano.tensor as T

    n_layers = len(init_b)

    bias = []
    weights = []
    muStates = []
    for layer_i in xrange(n_layers):
        bias.append(theano.shared(value=init_b[layer_i],
                                    name='b'+str(layer_i),
                                    borrow=True))
        weights.append(theano.shared(value=init_W[layer_i],
                                    name='W'+str(layer_i),
                                    borrow=True))
        muStates.append(T.matrix('mu'+str(layer_i)))

    for layer_i in xrange(n_layers):
        diffe = T.tile(bias[layer_i].copy(), (nData,1))
        # All layers except top
        if layer_i < (n_layers-1):
            W_h = weights[layer_i].dot(muStates[layer_i+1].T).T
            diffe += W_h

        if layer_i > 0:
            vT_W = muStates[layer_i-1].dot(weights[layer_i-1])
            diffe += vT_W

        exK = muStates[layer_i]*T.exp(.5*-diffe) + (1.-muStates[layer_i])*T.exp(.5*diffe)
        flows += exK.sum()
    return flows

class sdbm(object):
    """Simple Deep Boltzmann Machine (DBM)

    A Deep Boltzmann Machine trained with Minimum Probability 
    Flow.

    Parameters
    ----------
    n_units : array-like, int, shape (n_layers)
        Number of units in each layer

    weights : array-like, list, optional
        Initialize the weights with a list of length (n_layers-1). 
        Each element in the list corresponds to a numpy array of shape 
        (n_units_prev_layer, n_units_current_layer).

    bias : array-like, list, optional
        Initialize the biases with a list of length (n_layers). Each
        element in the list corresponds to a numpy array of shape 
        (n_units_current_layer).

    state : array-like, list, optional
        Initialize the state of all units with a list of length (n_layers).
        Each element in the list corresponds to a numpy array of shape
        (n_units_current_layer).

    rng : RandomState, optional
        Random number generator to use to make results reproducible
    """
    def __init__(self,n_units,weights=None,bias=None,state=None,rng=np.random.RandomState(235)):
        if weights is None:
            self.weights = []
            for layer_i in range(1,len(n_units)):
                self.weights.append(rng.uniform(low=-4 * np.sqrt(6. / (n_units[layer_i-1] + n_units[layer_i])),
                                            high=4 * np.sqrt(6. / (n_units[layer_i-1] + n_units[layer_i])),
                                            size=(n_units[layer_i-1],n_units[layer_i])))
        else:
            self.weights = weights
        if bias is None:
            self.bias = []
            for layer_i in range(len(n_units)):
                # self.bias.append(rng.uniform(low=-4 * np.sqrt(6. / (n_units[layer_i])),
                #                         high=4 * np.sqrt(6. / (n_units[layer_i])),
                #                         size=n_units[layer_i]))
                self.bias.append(np.zeros(n_units[layer_i]))
        else:
            self.bias = bias
        if state is None:
            self.state = []
            for layer_i in range(len(n_units)):
                self.state.append(rng.randint(2,size=(n_units[layer_i])))
        else:
            self.state = state

        self.rng = rng
        self.meanState = None
        self.n_layers = len(n_units)
        self.n_units = n_units
        self.temperature = 1.

    def pretrain(self,vis,eps):
        """Trains each layer with a modified RBM

        Parameters
        ----------
        vis : array-like, shape (n_data,n_units)
            Array of training data

        eps : float
            coefficient for MPF
        """
        # TODO: adjust this method for variable n_units
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
        # Find meanfield estimates
        muStates = self.ExHidden(vis,meanSteps,sample=False)
        ####
        # meanHStates = muStates.mean(0)[1:]
        # tmask = (meanHStates > .9) + (meanHStates < .1)
        # tmask = np.tile(tmask,(nData,1))
        # tmask = (self.rng.uniform(size=tmask.shape) < .1) * tmask

        # hstemp = muStates[:,1,:]
        # hstemp[tmask] = 0.5
        # muStates[:,1,:] = hstemp

        # tmask = tmask.ravel()
        # reset = ((self.rng.uniform(size=(muStates.shape[0], tmask.shape[0])) < 0.1) & tmask.reshape((1,-1))).ravel()
        # fullStatesShape = muStates.shape
        # muStates = muStates.reshape((-1,))

        # muStates[reset] = 0.7
        # muStates = muStates.reshape((fullStatesShape))

        ###
        for ii in xrange(steps):
            dkdw = list_zeros_like(self.weights)
            for layer_i in xrange(self.n_layers):
                diffe = np.tile(self.bias[layer_i].copy(), (nData,1))
                # All layers except top
                if layer_i < (self.n_layers-1):
                    W_h = self.weights[layer_i].dot(muStates[layer_i+1].T).T
                    diffe += W_h
                # All layers except bottom (visible)
                if layer_i > 0:
                    vT_W = muStates[layer_i-1].dot(self.weights[layer_i-1])
                    diffe += vT_W
                
                # Bias update
                diffeb = -muStates[layer_i]*np.exp(.5*-diffe) + (1.-muStates[layer_i])*np.exp(.5*diffe)
                dkdbl = 0.5*diffeb.sum(0)
                self.bias[layer_i] -= eps*dkdbl/float(nData)

                # Weights update
                # All layers except top
                if layer_i < (self.n_layers-1):
                    dkdwl = 0.5*np.einsum('ij,ik->jk',diffeb,muStates[layer_i+1])
                    dkdw[layer_i] += dkdwl
                    # self.weights[layer_i] -= eps*dkdwl/float(nData)
                    import ipdb; ipdb.set_trace()
                # All layers except bottom (visible)
                if layer_i > 0:
                    dkdwlprev = 0.5*np.einsum('ij,ik->jk',muStates[layer_i-1],diffeb)
                    dkdw[layer_i-1] += dkdwlprev
                    # self.weights[layer_i-1] -= eps*dkdwlprev/float(nData)
                    import ipdb; ipdb.set_trace()

            for layer_i in xrange(self.n_layers-1):
                self.weights[layer_i] -= eps*dkdw[layer_i]/float(nData)

    def ExTrainFull(self,vis,steps,eps,meanSteps):
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
        # Find meanfield estimates
        dataStates = self.ExHidden(vis,meanSteps,sample=False)
        nondataStates = self.ExFull(vis,meanSteps,sample=False)
        ####
        # meanHStates = muStates.mean(0)[1:]
        # tmask = (meanHStates > .9) + (meanHStates < .1)
        # tmask = np.tile(tmask,(nData,1))
        # tmask = (self.rng.uniform(size=tmask.shape) < .1) * tmask

        # hstemp = muStates[:,1,:]
        # hstemp[tmask] = 0.5
        # muStates[:,1,:] = hstemp

        # tmask = tmask.ravel()
        # reset = ((self.rng.uniform(size=(muStates.shape[0], tmask.shape[0])) < 0.1) & tmask.reshape((1,-1))).ravel()
        # fullStatesShape = muStates.shape
        # muStates = muStates.reshape((-1,))

        # muStates[reset] = 0.7
        # muStates = muStates.reshape((fullStatesShape))

        ###
        for ii in xrange(steps):
            dataE = 0.
            nondataE = 0.
            for layer_i in xrange(self.n_layers):
                dataE += dataStates[layer_i].dot(self.bias[layer_i])
                nondataE += nondataStates[layer_i].dot(self.bias[layer_i])
                if layer_i < (self.n_layers-1):
                    dataE += np.einsum('ij,jk,ik->i',dataStates[layer_i],self.weights[layer_i],dataStates[layer_i+1])
                    nondataE += np.einsum('ij,jk,ik->i',nondataStates[layer_i],self.weights[layer_i],nondataStates[layer_i+1])

            for layer_i in xrange(self.n_layers):
                expdiffe = np.exp(0.5*(dataE-nondataE))
                dkdbl = 0.5*expdiffe.dot(-dataStates[layer_i]+nondataStates[layer_i])
                self.bias[layer_i] -= eps*dkdbl/float(nData)
                if layer_i < (self.n_layers-1):
                    dkdwl = .5*(-np.einsum('ij,ik,i->jk',dataStates[layer_i],dataStates[layer_i+1],expdiffe) + np.einsum('ij,ik,i->jk',nondataStates[layer_i],nondataStates[layer_i+1],expdiffe))
                    self.weights[layer_i] -= eps*dkdwl/float(nData)

    def ExHidden(self,vis,meanSteps,sample=False):
        """Finds the expectation for hidden units using mean-field variational approach

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Visible data to condition on

        meanSteps : int
            Number of mean-field steps to cycle through

        sample : boolean, optional
            Return a sampling based on the mean-field values
        """
        nData = vis.shape[0]
        # Initialize state to visible
        if (self.meanState is None) or (self.meanState[0].shape[0] != nData):
            self.meanState = []
            for n_unit in self.n_units:
                self.meanState.append(np.zeros((nData,n_unit))+.5)

        self.meanState[0] = vis
        for ii in xrange(meanSteps):
            # Find activations for internal layers
            for jj in xrange(1,self.n_layers-1):
                terms = np.tile(self.bias[jj].copy(),(nData,1))+self.meanState[jj-1].dot(self.weights[jj-1])+self.weights[jj].dot(self.meanState[jj+1].T).T
                self.meanState[jj] = sigm(terms)
            # Find activation for top layer
            terms = np.tile(self.bias[self.n_layers-1].copy(),(nData,1))+self.meanState[self.n_layers-2].dot(self.weights[self.n_layers-2])
            self.meanState[self.n_layers-1] = sigm(terms)
            # Find activations for internal layers going backwards
            for jj in xrange(self.n_layers-2,0,-1):
                terms = np.tile(self.bias[jj].copy(),(nData,1))+self.meanState[jj-1].dot(self.weights[jj-1])+self.weights[jj].dot(self.meanState[jj+1].T).T
                self.meanState[jj] = sigm(terms)

        if sample:
            sampleState = []
            for layerState in self.meanState:
                sampleState.append((self.rng.rand(*layerState.shape) <= layerState).astype('float'))
            return sampleState
        return self.meanState

    def ExFull(self,vis,meanSteps,sample=False):
        """Finds the expectation for all units using mean-field variational approach

        Parameters
        ----------
        vis : array-like, shape (n_data, n_units)
            Visible data to condition on

        meanSteps : int
            Number of mean-field steps to cycle through

        sample : boolean, optional
            Return a sampling based on the mean-field values
        """
        nData = vis.shape[0]
        # Initialize state to visible
        if (self.meanState is None) or (self.meanState[0].shape[0] != nData):
            self.meanState = []
            for n_unit in self.n_units:
                self.meanState.append(np.zeros((nData,n_unit))+.5)

        self.meanState[0] = vis
        for ii in xrange(meanSteps):
            # Find activations for internal layers
            for jj in xrange(1,self.n_layers-1):
                terms = np.tile(self.bias[jj].copy(),(nData,1))+self.meanState[jj-1].dot(self.weights[jj-1])+self.weights[jj].dot(self.meanState[jj+1].T).T
                self.meanState[jj] = sigm(terms)
            # Find activation for top layer
            terms = np.tile(self.bias[self.n_layers-1].copy(),(nData,1))+self.meanState[self.n_layers-2].dot(self.weights[self.n_layers-2])
            self.meanState[self.n_layers-1] = sigm(terms)
            # Find activations for internal layers going backwards
            for jj in xrange(self.n_layers-2,0,-1):
                terms = np.tile(self.bias[jj].copy(),(nData,1))+self.meanState[jj-1].dot(self.weights[jj-1])+self.weights[jj].dot(self.meanState[jj+1].T).T
                self.meanState[jj] = sigm(terms)
            # Find activation for bottom layer
            terms = np.tile(self.bias[0].copy(),(nData,1))+self.weights[0].dot(self.meanState[1].T).T
            self.meanState[0] = sigm(terms)
        if sample:
            sampleState = []
            for layerState in self.meanState:
                sampleState.append((self.rng.rand(*layerState.shape) <= layerState).astype('float'))
            return sampleState
        return self.meanState

    def sampleHidden(self,vis,steps):
        """Sample from P(h|v) for the DBM via gibbs sampling for each
        layer: P(h_layer_i|h_layer_i+1, h_layer_i-1)

        Parameters
        ----------
        vis : array-like, shape (n_units)
            Visible data to condition on

        steps : int
            Number of steps to gibbs sample
        """
        stateUp = copy.deepcopy(self.state)
        stateUp[0] = vis
        for ii in xrange(steps):
            # Sample bottom layers going up
            for jj in xrange(1,self.n_layers-1):
                terms = self.bias[jj] + stateUp[jj-1].dot(self.weights[jj-1]) + self.weights[jj].dot(stateUp[jj+1])
                probs = sigm(terms)
                stateUp[jj] = self.rng.rand(self.n_units[jj]) <= probs

            # Sampling for the top layer, before going back down
            top = self.n_layers-1
            terms = self.bias[top] + stateUp[top-1].dot(self.weights[top-1])
            probs = sigm(terms)
            stateUp[top] = self.rng.rand(self.n_units[top]) <= probs

            # Sample bottom hidden layers going down
            for jj in xrange(self.n_layers-2,0,-1):
                terms = self.bias[jj] + stateUp[jj-1].dot(self.weights[jj-1]) + self.weights[jj].dot(stateUp[jj+1])
                probs = sigm(terms)
                stateUp[jj] = self.rng.rand(self.n_units[jj]) <= probs

        return stateUp

    def sampleVisible(self,state):
        """Sample from P(v|h) of the DBM.

        Parameters
        ----------
        state : array-like, shape (n_layers, n_units)
            State of all the units
        """
        terms = self.bias[0] + self.weights[0].dot(state[1])
        probs = sigm(terms)
        vis = self.rng.rand(self.n_units[0]) <= probs

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
            state = self.sampleHidden(vis,4)
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
        confabs = np.zeros((n_keep,vis.shape[0]))
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

    def flowSamples(self, vis, epsilon, meanSteps, intOnly=False):
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
        muStates = self.ExHidden(vis,meanSteps)
        if intOnly:
            fullStates = np.around(fullStates)

        flows = 0.
        for layer_i in xrange(self.n_layers):
            diffe = np.tile(self.bias[layer_i].copy(), (nData,1))
            # All layers except top
            if layer_i < (self.n_layers-1):
                W_h = self.weights[layer_i].dot(muStates[layer_i+1].T).T
                diffe += W_h
            # All layers except bottom (visible)
            if layer_i > 0:
                vT_W = muStates[layer_i-1].dot(self.weights[layer_i-1])
                diffe += vT_W
            exK = muStates[layer_i]*np.exp(.5*-diffe) + (1.-muStates[layer_i])*np.exp(.5*diffe)
            flows += exK.sum()

        return flows*epsilon/nData

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

    def muTrain(self,vis,steps,eps,meanSteps):
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
        # Find meanfield estimates
        muStates = self.ExHidden(vis,meanSteps,sample=False)
        for ii in xrange(steps):
            dkdw = list_zeros_like(self.weights)
            for layer_i in xrange(self.n_layers):
                diffe = np.tile(self.bias[layer_i].copy(), (nData,1))
                # All layers except top
                if layer_i < (self.n_layers-1):
                    W_h = self.weights[layer_i].dot(muStates[layer_i+1].T).T
                    diffe += W_h
                # All layers except bottom (visible)
                if layer_i > 0:
                    vT_W = muStates[layer_i-1].dot(self.weights[layer_i-1])
                    diffe += vT_W
                diffe = (1.-2.*muStates[layer_i])*diffe
                
                # Bias update
                dkdbl = 0.5*(1.-2.*muStates[layer_i])*np.exp(.5*diffe).sum(0)
                self.bias[layer_i] -= eps*dkdbl/float(nData)

                # Weights update
                # All layers except top
                if layer_i < (self.n_layers-1):
                    dkdw[layer_i] += .5*(1.-2.*muStates[layer_i]).T.dot(muStates[layer_i+1]*diffe)
                # All layers except bottom (visible)
                if layer_i > 0:
                    dkdw[layer_i-1] += .5*muStates[layer_i-1].T.dot((1.-2*muStates[layer_1])*diffe)

            for layer_i in xrange(self.n_layers-1):
                self.weights[layer_i] -= eps*dkdw[layer_i]/float(nData)
