import Utils.DataHelper as DataHelper
import os
from PIL import Image
from Utils import CostFHelper
from Utils.FilterHelper import *

# Import layers
from Layers.RBMHiddenLayer import *
from Layers.HiddenLayer import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# HYPER PARAMETERS
DATASET_NAME = '../Dataset/mnist.pkl.gz'
BATCH_SIZE = 10
VALIDATION_FREQUENCY = 50000
VISUALIZE_FREQUENCY = 5000

# NETWORKS HYPER PARAMETERS
HIDDEN_LAYERS_SIZES = [1000, 1000, 1000]
NUM_CHAINS = 15
NUM_OUT = 10

# PRETRAIN HYPER PARAMETERS
PRETRAINING_SAVE_PATH = '../Pretrained/pretrain_stage.pkl'
PRETRAINING_EPOCH = 100
PRETRAINING_LEARNING_RATE = 0.01
PRETRAINING_SAVE_FREQUENCY = 5000

# FINE-TUNING HYPER PARAMETERS
FINETUNING_SAVE_PATH = '../Pretrained/training_stage.pkl'
FINETUNING_EPOCH = 50
FINETUNING_LEARNING_RATE = 0.1

def DBN():
    #########################################
    #      LOAD DATASET                     #
    #########################################
    # Load datasets from local disk or download from the internet
    # We only load the images, not label
    datasets = DataHelper.LoadData(DATASET_NAME)
    trainSetX, trainSetY = datasets[0]
    validSetX, validSetY = datasets[1]
    testSetX, testSetY = datasets[2]

    nTrainBatchs = trainSetX.get_value(borrow=True).shape[0]
    nValidBatchs = validSetX.get_value(borrow=True).shape[0]
    nTestBatchs = testSetX.get_value(borrow=True).shape[0]
    nTrainBatchs //= BATCH_SIZE
    nValidBatchs //= BATCH_SIZE
    nTestBatchs //= BATCH_SIZE

    #########################################
    #      CREATE MODEL                     #
    #########################################
    '''
    MODEL ARCHITECTURE
       INPUT    ->    HIDDEN LAYER    ->    HIDDEN LAYER    ->     HIDDEN LAYER    ->    OUTPUT
    ( 28x28 )       ( 1000 neurons )      ( 1000 neurons )       ( 1000 neurons )    ( 10 outputs )
    '''
    # Create random state
    rng = numpy.random.RandomState(12345)
    theanoRng = RandomStreams(rng.randint(2 ** 30))

    # Create shared variable for input
    Index = T.lscalar('Index')
    LearningRate = T.scalar('LearningRate', dtype = 'float32')
    X = T.matrix('X')
    X2D = X.reshape((BATCH_SIZE, 28 * 28))
    Y = T.ivector('Y')

    rbmLayers = []
    hiddenLayers = []
    for idx in range(len(HIDDEN_LAYERS_SIZES)):
        if idx == 0:
            inputSize = 28 * 28
        else:
            inputSize = HIDDEN_LAYERS_SIZES[idx - 1]

        if idx == 0:
            layerInput = X2D
        else:
            layerInput = hiddenLayers[-1].Output

        persistentChain = theano.shared(
            numpy.zeros((BATCH_SIZE, HIDDEN_LAYERS_SIZES[idx]), dtype=theano.config.floatX),
            borrow=True
        )
        rbm = RBMHiddenLayer(
            rng          = rng,
            theanoRng    = theanoRng,
            input        = layerInput,
            numVisible   = inputSize,
            numHidden    = HIDDEN_LAYERS_SIZES[idx],
            learningRate = LearningRate,
            persistent   = persistentChain,
            kGibbsSample = NUM_CHAINS
        )

        rbmLayers.append(rbm)

        hiddenLayer = HiddenLayer(
            rng          = rng,
            input        = layerInput,
            numIn        = inputSize,
            numOut       = HIDDEN_LAYERS_SIZES[idx],
            W            = rbm.W,
            b            = rbm.hBias
        )
        hiddenLayers.append(hiddenLayer)
    # Create last layer
    inputSize = HIDDEN_LAYERS_SIZES[-1]
    layerInput = hiddenLayers[-1].Output
    hiddenLayer = HiddenLayer(
        rng          = rng,
        input        = layerInput,
        numIn        = inputSize,
        numOut       = NUM_OUT,
        activation   = T.nnet.softmax
    )
    hiddenLayers.append(hiddenLayer)

    ################################################
    #           Calculate cost function            #
    ################################################
    # Create train functions to train rbm layer (pre-training stage)
    rbmLayerFuncs = []
    for idx, rbmLayer in enumerate(rbmLayers):
        # Cost function
        trainFunc = theano.function(
            inputs  = [Index, LearningRate],
            outputs = [rbmLayer.MonitoringCost],
            updates = rbmLayer.Updates,
            givens  = {
                X: trainSetX[Index * BATCH_SIZE : (Index + 1) * BATCH_SIZE]
            }
        )
        rbmLayerFuncs.append([trainFunc])

    # Create train function to train entire model (fine-tuning stage)
    params = [] # Get all params from all rbm layers
    [params.extend(hiddenLayer.Params) for hiddenLayer in hiddenLayers]
    output = hiddenLayers[-1].Output
    cost = CostFHelper.CrossEntropy(output, Y)
    grads = T.grad(cost, params)
    updates = [(param, param - LearningRate * grad)
               for (param, grad) in zip(params, grads)]
    # Train function
    trainFunc = theano.function(
        inputs   = [Index, LearningRate],
        outputs  = [cost],
        updates  = updates,
        givens   = {
            X: trainSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: trainSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
        }
    )
    # Test function
    pred = T.argmax(output)
    error = CostFHelper.Error(pred, Y)
    testFunc = theano.function(
        inputs  = [Index],
        outputs = [error],
        givens  = {
            X: testSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
            Y: testSetY[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE],
        }
    )
    fineTuningFunc = [trainFunc, testFunc]

    #########################################
    #      PRETRAINING STAGE                #
    #########################################
    # Load old model before training
    if os.path.isfile(PRETRAINING_SAVE_PATH):
        file = open(PRETRAINING_SAVE_PATH)
        [rbmLayer.LoadModel(file) for rbmLayer in rbmLayers]
        file.close()

    print ('Start pre-training stage....')
    iter = 0
    costRbm = []
    for idx, rbmLayerFunc in enumerate(rbmLayerFuncs):
        print ('Train layer %d ' % (idx))
        for epoch in range(PRETRAINING_EPOCH):
            for trainBatchIdx in range(nTrainBatchs):
                iter += BATCH_SIZE
                costAELayer = rbmLayerFunc[0](trainBatchIdx, PRETRAINING_LEARNING_RATE)

                if iter % VISUALIZE_FREQUENCY == 0:
                    print ('Epoch = %d, iteration = %d ' % (epoch, iter))
                    print ('      CostRBMLayer = %f ' % (costAELayer[0]))

                if iter % PRETRAINING_SAVE_FREQUENCY == 0:
                    file = open(PRETRAINING_SAVE_PATH, 'wb')
                    [rbmLayer.SaveModel(file) for rbmLayer in rbmLayers]
                    file.close()
                    print('Save model !')

        # Save after training layer
        file = open(PRETRAINING_SAVE_PATH, 'wb')
        [rbmLayer.SaveModel(file) for rbmLayer in rbmLayers]
        file.close()
        print('Save model !')

    print ('Start pre-training stage. Done!')

    #########################################
    #      FINE-TUNING STAGE                #
    #########################################
    # Load save model
    if os.path.isfile(PRETRAINING_SAVE_PATH):
        file = open(PRETRAINING_SAVE_PATH)
        [rbmLayer.LoadModel(file) for rbmLayer in rbmLayers]
        file.close()
    shape = [(28, 28), (50, 20), (50, 20)]
    for idx, rbmLayer in enumerate(rbmLayers):
        image = Image.fromarray(tile_raster_images(
            X=rbmLayer.Params[0].get_value(borrow=True).T,
            img_shape=shape[idx], tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('filters_corruption_30_%d.png' % (idx))

    # Training
    print ('Start fine-tuning stage...')
    for epoch in range(FINETUNING_EPOCH):
        iter += BATCH_SIZE
        cost = fineTuningFunc[0](trainBatchIdx, FINETUNING_LEARNING_RATE)

        if iter % VISUALIZE_FREQUENCY == 0:
            print ('Epoch = %d, iteration = %d ' % (epoch, iter))
            print ('      Cost fine-tuning = %f ' % (cost[0]))

        if iter % VALIDATION_FREQUENCY == 0:
            validError = 0
            print ('Validate current model ')
            for validBatchIdx in range(nValidBatchs):
                validError += fineTuningFunc[1](validBatchIdx)[0]
            validError /= (nValidBatchs)

            if (validError < bestError):
                bestError = validError
                print ('Save model ! Sum cost = %f ' % (validError))
                file = open(FINETUNING_SAVE_PATH, 'wb')
                [hiddenLayer.SaveModel(file) for hiddenLayer in hiddenLayers]
                file.close()
    print ('Start fine-tuning stage. Done!')

if __name__ == '__main__':
    DBN()