import cPickle as pickle
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import theano
from lasagne import updates, layers, objectives, regularization, utils
from theano import tensor as T

import params_io as io
from build_cg import build_computation_graph
from load import mnist

MODE = 'TRAIN'  # MODE IS 'TRAIN' OR 'TEST'
BEST_MODEL_PATH = 'models/best'
LAST_MODEL_PATH = 'models/last'


# -----------------------HELPER FUNCTIONS-------------------------------------------
def iterate_minibatches(inputs, targets, labeled, batchsize, shuffle=False):
    # this function create mini batches for train/validation/test
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], labeled[excerpt]

def repeat_col(col, n_col):
    # repeat a column vector n times
    return np.repeat(col, n_col, axis=1)


def run_test(test_function, testX, testY, prefix='test'):
    # run test on an image set
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(testX, testY,
                                     np.zeros((testY.shape[0], 1)),
                                     500, shuffle=False):
        inputs, targets, labeled = batch
        err, _, _, _, acc = test_function(inputs, targets, labeled)
        test_err += err
        test_acc += acc
        test_batches += 1
    average_test_score = test_err / test_batches
    test_accuracy = test_acc / test_batches
    print("  " + prefix + " loss:\t\t{:.6f}".format(average_test_score))
    print("  " + prefix + " accuracy:\t\t{:.6f} %".format(
        test_accuracy * 100))
    return average_test_score, test_accuracy
#-----------------------PARAMETERS----------------------------#

print('Loading data')

# Load index of labeled images in train set
with open('labeled_index.pkl', 'r') as f:
    loaded_obj = pickle.load(f)
VALIDATION_SIZE = loaded_obj[0]
labeled_idx = loaded_obj[1]

# Load image and label of train, validation, test set
trX, vlX, teX, trY, vlY, teY = mnist(onehot=True, ndim=2)

print('Building computation graph')

# Set the dimension here, 1 list = 1 stack, 2 list = 2 stacks, etc...
IM_SIZE = trX.shape[1]
dimensions = [[1500, 3, 100]]  # example of 1 stack
#dimensions = [[1500,3,500],[1000,3,300]] # example of 2 stacks
input_shape = [None, IM_SIZE]
input_var = T.fmatrix('input_var')
target_var = T.fmatrix('target_var')
labeled_var = T.fmatrix('labeled_var')
unsupervised_graph, supervised_graph, features = build_computation_graph(input_var, input_shape, dimensions)
# Train graph has dropout
reconstruction = layers.get_output(unsupervised_graph)
prediction = layers.get_output(supervised_graph)
# Test graph has no dropout so deterministic = True
test_reconstruction = layers.get_output(unsupervised_graph, deterministic=True)
test_prediction = layers.get_output(supervised_graph, deterministic=True)

# Get all trainable params
params = layers.get_all_params(unsupervised_graph, trainable=True) + \
         layers.get_all_params(supervised_graph, trainable=True)
params = utils.unique(params)

# Get regularizable params
regularization_params = layers.get_all_params(unsupervised_graph, regularizable=True) + \
         layers.get_all_params(supervised_graph, regularizable=True)
regularization_params = utils.unique(regularization_params)

# Set learning ratio for unsupervised, supervised and weights regularization
lr = (1.0, 1, 1e-4)

# Creating loss functions
# Train loss has to take into account of labeled image or not
loss1 = objectives.squared_error(reconstruction, input_var)
loss2 = objectives.squared_error(prediction, target_var)
l2_penalties = regularization.apply_penalty(regularization_params, regularization.l2)
loss = lr[0]*loss1.mean() +\
       lr[1]*(loss2*repeat_col(labeled_var, 10)).mean() +\
       lr[2]*l2_penalties.mean()
# Test loss means 100% labeled
test_loss1 = objectives.squared_error(test_reconstruction, input_var)
test_loss2 = objectives.squared_error(test_prediction, target_var)
test_loss = lr[0]*test_loss1.mean() +\
            lr[1]*test_loss2.mean() +\
            lr[2]*l2_penalties.mean()

# Update function to train
updates_function = updates.adam(loss, params, 0.00001)

# Compile train function
train_fn = theano.function([input_var, target_var, labeled_var], loss, updates=updates_function,
                           allow_input_downcast=True,
                           on_unused_input='ignore')
# Compile test prediction function
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                  dtype=theano.config.floatX)
# Compile a second function computing the validation loss and accuracy:
#val_fn = theano.function([input_var, target_var, labeled_var], [loss2*lr[1], test_acc], allow_input_downcast=True)
val_fn = theano.function([input_var, target_var, labeled_var],
                         [test_loss,
                          lr[0]*test_loss1.mean(),
                          lr[1]*test_loss2.mean(),
                          lr[2]*l2_penalties.mean(),
                          test_acc], allow_input_downcast=True,
                         on_unused_input='ignore')

if MODE == 'TEST':
    # load saved best model
    io.read_model_data([unsupervised_graph, supervised_graph], BEST_MODEL_PATH)
    run_test(val_fn, teX, teY)
elif MODE == 'TRAIN':
    # if last model exists, load last model:
    if os.path.isfile(LAST_MODEL_PATH + '.' + io.PARAM_EXTENSION):
        choice = input(
            'PREVIOUS MODEL FOUND, CONTINUING TRAINING OR OVERRIDE OR END TRAINING? (ANSWER: "CONTINUE", "OVERRIDE", "END")\n')
        if choice == 'CONTINUE':
            best_validation_acc = io.read_model_data([unsupervised_graph, supervised_graph], BEST_MODEL_PATH)
        elif choice == 'OVERRIDE':
            best_validation_acc = 0
        else:
            sys.exit('Terminated by user choice.')
    print 'Training...'
    # number of epochs to train
    num_epochs = 10000
    train_loss = []
    train_loss1 = []
    train_loss2 = []
    train_regularize = []
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in iterate_minibatches(trX, trY, labeled_idx, 500, shuffle=True):
            inputs, targets, labeled = batch
            train_err = train_fn(inputs, targets, labeled)

        train_err = 0
        train_acc = 0
        train_batches = 0
        train_loss.append(0)
        train_loss1.append(0)
        train_loss2.append(0)
        train_regularize.append(0)
        for batch in iterate_minibatches(trX, trY, labeled_idx, 500, shuffle=True):
            inputs, targets, labeled = batch
            err, _loss1, _loss2, _regularize, acc = val_fn(inputs, targets, labeled)
            train_loss[-1] += err
            train_loss1[-1] += _loss1
            train_loss2[-1] += _loss2
            train_regularize[-1] += _regularize
            train_err += err
            train_acc += acc
            train_batches += 1
        train_loss[-1] /= train_batches
        train_loss1[-1] /= train_batches
        train_loss2[-1] /= train_batches
        train_regularize[-1] /= train_batches
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.6f} %".format(
            train_acc / train_batches * 100))

        valid_err, valid_acc = run_test(val_fn, vlX, vlY, "validation")
        # save last model
        io.write_model_data([unsupervised_graph, supervised_graph], best_validation_acc, LAST_MODEL_PATH)
        # if best model is found, save best model
        if valid_acc > best_validation_acc:
            best_validation_acc = valid_acc
            io.write_model_data([unsupervised_graph, supervised_graph], best_validation_acc, BEST_MODEL_PATH)
            print('NEW BEST MODEL FOUND!')
            run_test(val_fn, teX, teY, "test")

    # plot losses graph
    plt.clf()
    plt.plot(train_loss, 'r-')
    plt.plot(train_loss1, 'g-')
    plt.plot(train_loss2, 'b-')
    plt.plot(train_regularize, 'k-')
    plt.show()
