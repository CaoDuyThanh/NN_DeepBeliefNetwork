import theano
import theano.tensor as T
import numpy
import pickle
import cPickle

class RBMHiddenLayer:
    def __init__(self,
                 rng,
                 theanoRng,
                 input,
                 numVisible,
                 numHidden,
                 learningRate,
                 activation = T.nnet.sigmoid,
                 persistent = None,
                 kGibbsSample = 1,
                 W = None,
                 hBias = None,
                 vBias = None):
        # Set parameters
        self.Rng          = rng
        self.TheanoRng    = theanoRng
        self.Input        = input
        self.NumVisible   = numVisible
        self.NumHidden    = numHidden
        self.LearningRate = learningRate
        self.Activation   = activation
        self.Persistent   = persistent
        self.kGibbsSample = kGibbsSample
        self.W            = W
        self.hBias        = hBias
        self.vBias        = vBias

        self.createModel()

    def createModel(self):
        if self.W is None:
            wBound = numpy.sqrt(6.0 / (self.NumVisible + self.NumHidden))
            self.W = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = (self.NumVisible, self.NumHidden)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.hBias is None:
            hBound = numpy.sqrt(6.0 / self.NumHidden)
            self.hBias = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -hBound,
                        high =  hBound,
                        size = (self.NumHidden,)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.vBias is None:
            vBound = numpy.sqrt(6.0 / self.NumVisible)
            self.vBias = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -vBound,
                        high =  vBound,
                        size = (self.NumVisible)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        self.Params = [self.W, self.hBias, self.vBias]

        # Calculate Gibbs Sample
        if self.Persistent is None:
            chainStart = self.sampleHGivenV(self.Input)
        else:
            chainStart = self.Persistent

        (
            [
                vSample,
                hSample
            ],
            updates
        ) = theano.scan(fn           = self.gibbsSampleHvH,
                        outputs_info = [None, chainStart],
                        n_steps      = self.kGibbsSample)
        chainEnd = vSample[-1]
        updates[self.Persistent] = hSample[-1]

        # Calculate cost function
        self.Cost = T.mean(self.freeEnergy(self.Input)) - T.mean(self.freeEnergy(chainEnd))
        self.MonitoringCost = self.getPseudoLikeLihoodCost(updates)

        # Calculate grad
        self.Grads = T.grad(self.Cost, self.Params)

        for (param, grad) in zip(self.Params, self.Grads):
            updates[param] = param - self.LearningRate * grad

        self.Updates = updates

    def propUp(self, visible):
        return self.Activation(T.dot(visible, self.W) + self.hBias)

    def sampleHGivenV(self, visible):
        h1Mean = self.propUp(visible)
        h1Sample = self.TheanoRng.binomial(size  = h1Mean.shape,
                                           n     = 1,
                                           p     = h1Mean,
                                           dtype = theano.config.floatX)
        return h1Sample

    def propDown(self, hidden):
        return self.Activation(T.dot(hidden, self.W.T) + self.vBias)

    def sampleVGivenH(self, hidden):
        v1Mean = self.propDown(hidden)
        v1Sample = self.TheanoRng.binomial(size  = v1Mean.shape,
                                           n     = 1,
                                           p     = v1Mean,
                                           dtype = theano.config.floatX)
        return v1Sample

    def freeEnergy(self, visible):
        bv  = - T.dot(visible, self.vBias.T)
        cWv = - T.sum(T.log(1 + T.exp(T.dot(visible, self.W) + self.hBias)), axis = 1)
        return bv + cWv

    def gibbsSampleHvH(self, hidden):
        v1Sample = self.sampleVGivenH(hidden)
        h1Sample = self.sampleHGivenV(v1Sample)
        return [v1Sample, h1Sample]

    def gibbsSampleVhV(self, visible):
        h1Sample = self.sampleHGivenV(visible)
        v1Sample = self.sampleVGivenH(h1Sample)
        return [h1Sample, v1Sample]

    def getPseudoLikeLihoodCost(self, updates):
        bitIndex = theano.shared(value = 0, name = 'bitIndex')

        xi = T.round(self.Input)
        energyXi = self.freeEnergy(xi)
        xiFlip = T.set_subtensor(xi[:, bitIndex], 1 - xi[:, bitIndex])
        energyXiFlip = self.freeEnergy(xiFlip)

        # Use log likelihood to calculate cost
        cost = T.mean(self.NumVisible * T.log(self.Activation(energyXiFlip - energyXi)))

        updates[bitIndex] = (bitIndex + 1) % self.NumVisible

        return cost

    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow = True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self.Params]