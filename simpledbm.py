from __future__ import division
from __future__ import print_function
import numpy as np

class sdbm(object):
    def __init__(self,layers,npl,weights=None,bias=None,state=None):
        if weights is None:
            self.weights = np.random.randn(layers-1,npl,npl)
        else:
            self.weights = weights
        if bias is None:
            self.bias = np.random.randn(layers,npl)
        else:
            self.bias = bias
        if state is None:
           self.state = np.random.randint(2,size=(layers,npl))
        else:
            self.state = state
           
        self.layers = layers
        self.npl = npl

    def pretrain(self,data):
        pass
    def train(self,data):
        pass

    def dedwDiff(self,state,layer,layerFlip,position):
        stateFlip = np.copy(state)
        stateFlip[layerFlip,position] = int(not state[layerFlip,position])
        return -(np.outer(state[layer],state[layer+1])-np.outer(stateFlip[layer],stateFlip[layer+1]))
    def dedbDiff(self,state,layer,position):
        stateFlip = np.copy(state)
        stateFlip[layer,position] = int(not state[layer,position])
        return -(state[layer]-stateFlip[layer])
    def eDiff(self,state,layerFlip,position):
        stateFlip = np.copy(state)
        stateFlip[layerFlip,position] = int(not state[layerFlip,position])
        return self.energy(self.weights,self.bias,state)-self.energy(self.weights,self.bias,stateFlip)

    def mpfTrain(self,vis,steps,eps,stepsSample):
        nData = vis.shape[0]
        fullState = np.array([self.sampleHidden(vis1,stepsSample) for vis1 in vis])
        for ii in xrange(steps):
            dw = np.zeros_like(self.weights)
            db = np.zeros_like(self.bias)
            #update weights and biases
            for state in fullState:
                #for visible
                for kk in xrange(self.npl):
                    db[0] = db[0]+self.dedbDiff(state,0,kk)*np.exp(.5*(self.eDiff(state,0,kk)))
                for jj in xrange(self.layers-1):
                    for kk in xrange(self.npl):
                        dw[jj] = dw[jj]+self.dedwDiff(state,jj,jj,kk)*np.exp(.5*(self.eDiff(state,jj,kk)))
                        ep = self.eDiff(state,jj+1,kk)
                        dw[jj] = dw[jj]+self.dedwDiff(state,jj,jj+1,kk)*np.exp(.5*(ep))
                        db[jj+1] = db[jj+1]+self.dedbDiff(state,jj+1,kk)*np.exp(.5*(ep))
            self.weights = self.weights+eps*dw/nData
            self.bias = self.bias+eps*db/nData
            
    def sampleHidden(self,vis,steps):
        stateUp = np.copy(self.state)
        stateUp[0] = vis
        for ii in xrange(steps):
            rands = np.random.rand(self.layers-1,self.npl)
            for jj in xrange(1,self.layers-1):
                for kk in xrange(self.npl):
                    terms = self.bias[jj,kk]+np.dot(stateUp[jj-1],self.weights[jj-1,:,kk])+np.dot(stateUp[jj+1],self.weights[jj,kk])
                    prob = 1/(1+np.exp(terms))
                    if rands[jj-1,kk] <= prob:
                        stateUp[jj,kk] = 1
                    else:
                        stateUp[jj,kk] = 0
            for kk in xrange(self.npl):
                top = self.layers-1
                terms = self.bias[top,kk]+np.dot(stateUp[top-1],self.weights[top-1,:,kk])
                prob = 1/(1+np.exp(terms))
                if rands[top-1,kk] <= prob:
                    stateUp[top,kk] = 1
                else:
                    stateUp[top,kk] = 0
        return stateUp

    def sampleFull(self,steps):
        pass

    def energy(self,weights,bias,state):
        ebias = np.sum([np.dot(bias[ii],state[ii]) for ii in xrange(self.layers)])
        eweights = np.sum([np.dot(state[ii],np.dot(weights[ii],state[ii+1])) for ii in xrange(self.layers-1)])
        return ebias+eweights
    def curEnergy(self):
        ebias = np.sum([np.dot(self.bias[ii],self.state[ii]) for ii in xrange(self.layers)])
        eweights = np.sum([np.dot(self.state[ii],np.dot(self.weights[ii],self.state[ii+1])) for ii in xrange(self.layers-1)])
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

