import theano
import numpy
import pickle
import cPickle
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer:
    def __init__(self,
                 rng,                   # Random seed
                 input,                 # Data input
                 numIn,                 # Number neurons of input
                 numOut,                # Number reurons out of layer
                 activation = T.tanh,   # Activation function
                 corruption = 0.0,
                 W = None,
                 b = None
                 ):
        # Set parameters
        self.Rng = rng;
        self.Input = input
        self.NumIn = numIn
        self.NumOut = numOut
        self.Activation = activation
        self.Corruption = corruption
        self.W = W
        self.b = b

        self.createModel()

    def createModel(self):
        # Create shared parameters for hidden layer
        if self.W is None:
            """ We create random weights (uniform distribution) """
            # Create boundary for uniform generation
            wBound = 4 * numpy.sqrt(6.0 / (self.NumIn + self.NumOut))
            self.W = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low = -wBound,
                        high = wBound,
                        size = (self.NumIn, self.NumOut)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )
        self.WTranspose = self.W.T

        if self.b is None:
            """ We create zeros bias """
            # Create bias
            self.b = theano.shared(
                numpy.zeros(
                    shape=(self.NumOut,),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        # Parameters of model
        self.Params = [self.W, self.b]

        # Calculate Input corruption
        inputCorrupt = self.getCorruptedInput(self.Input, self.Corruption)
        self.Output = self.Activation(T.dot(inputCorrupt, self.W) + self.b)

    def getCorruptedInput(self, input, corruptionLevel):
        theano_rng = RandomStreams(self.Rng.randint(2 ** 30))
        return theano_rng.binomial(size=input.shape, n=1,
                                   p=1 - corruptionLevel,
                                   dtype=theano.config.floatX) * input

    def LoadModel(self, file):
        self.W.set_value(cPickle.load(file), borrow = True)
        self.b.set_value(cPickle.load(file), borrow = True)

    def SaveModel(self, file):
        pickle.dump(self.W.get_value(borrow = True), file, -1)
        pickle.dump(self.b.get_value(borrow = True), file, -1)
