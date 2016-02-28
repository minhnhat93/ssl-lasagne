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
from parameters import run_parameters
from path_settings import BEST_MODEL_PATH, LAST_MODEL_PATH, WRONG_SAMPLES_PATH, DATA_PATH


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
    wrong_samples = []
    wrong_classification = []
    for batch in iterate_minibatches(testX, testY,
                                     np.zeros((testY.shape[0], 1)),
                                     run_parameters.batch_size,
                                     shuffle=False):
        inputs, targets, labeled = batch
        err, _, _, _, classification, acc, _wrong_samples = test_function(inputs, targets, labeled)
        test_err += err
        test_acc += acc
        wrong_samples += (_wrong_samples.nonzero()[0] + test_batches * run_parameters.batch_size).tolist()
        _wrong_classification = classification[_wrong_samples.nonzero()]
        wrong_classification += _wrong_classification.tolist()
        test_batches += 1
    average_test_score = test_err / test_batches
    test_accuracy = test_acc / test_batches
    print("  " + prefix + " loss:\t\t{:.6f}".format(average_test_score))
    print("  " + prefix + " accuracy:\t{:.6f} %".format(
        test_accuracy * 100))
    return average_test_score, test_accuracy, wrong_samples, wrong_classification


# -----------------------LOAD IMAGES AND LABELS----------------------------#
print('Loading data')

# Load index of labeled images in train set
with open(os.path.join(DATA_PATH,'labeled_index.pkl'), 'r') as f:
    labeled_idx = pickle.load(f)

# Load image and label of train, validation, test set
trX, vlX, teX, trY, vlY, teY = mnist(onehot=True, ndim=2)
IM_SIZE = trX.shape[1]

#-----------------------SET PARAMETERS-------------------------#
losses_ratio = run_parameters.losses_ratio
supervised_cost_fun = run_parameters.supervised_cost_fun

# -----------------------CREATE RUN FUNCTIONS------------------#
# Creating the computation graph
print('Building computation graph')
input_var = T.fmatrix('input_var')
target_var = T.fmatrix('target_var')
labeled_var = T.fmatrix('labeled_var')
unsupervised_graph, supervised_graph, features = build_computation_graph(input_var, run_parameters)
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

# Creating loss functions
# Train loss has to take into account of labeled image or not
loss1 = objectives.squared_error(reconstruction, input_var)
if supervised_cost_fun == 'squared_error':
    loss2 = objectives.squared_error(prediction, target_var) * repeat_col(labeled_var, 10)
elif supervised_cost_fun == 'categorical_crossentropy':
    loss2 = objectives.categorical_crossentropy(prediction, target_var) * labeled_var.T
l2_penalties = regularization.apply_penalty(regularization_params, regularization.l2)
loss = losses_ratio[0] * loss1.mean() + \
       losses_ratio[1] * loss2.mean() + \
       losses_ratio[2] * l2_penalties.mean()
# Test loss means 100% labeled
test_loss1 = objectives.squared_error(test_reconstruction, input_var)
if supervised_cost_fun == 'squared_error':
    test_loss2 = objectives.squared_error(test_prediction, target_var)
elif supervised_cost_fun == 'categorical_crossentropy':
    test_loss2 = objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = losses_ratio[0] * test_loss1.mean() + \
            losses_ratio[1] * test_loss2.mean() + \
            losses_ratio[2] * l2_penalties.mean()

# Update function to train
# sgd_lr = theano.shared(utils.floatX(0.001))
# sgd_lr_decay = utils.floatX(0.9)
updates_function = updates.adam(loss, params, run_parameters.update_lr)

# Compile train function
train_fn = theano.function([input_var, target_var, labeled_var], loss, updates=updates_function,
                           allow_input_downcast=True, on_unused_input='ignore')
# Compile test prediction function
classification = T.argmax(test_prediction, axis=1)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                  dtype=theano.config.floatX)
test_wrong = T.neq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1))
# Compile a second function computing the validation loss and accuracy:
#val_fn = theano.function([input_var, target_var, labeled_var], [loss2*lr[1], test_acc], allow_input_downcast=True)
val_fn = theano.function([input_var, target_var, labeled_var],
                         [test_loss,
                          losses_ratio[0] * test_loss1.mean(),
                          losses_ratio[1] * test_loss2.mean(),
                          losses_ratio[2] * l2_penalties.mean(),
                          classification, test_acc, test_wrong],
                         allow_input_downcast=True, on_unused_input='ignore')

# ----------------------------RUN-----------------------------------#
MODE = input('"TEST" OR "TRAIN"?\n')
if MODE == 'TEST':
    # load saved best model
    io.read_model_data([unsupervised_graph, supervised_graph], BEST_MODEL_PATH)
    run_test(val_fn, teX, teY)
elif MODE == 'TRAIN':
    # if last model exists, load last model:
    best_validation_acc = 0
    if os.path.isfile(LAST_MODEL_PATH + '.' + io.PARAM_EXTENSION):
        choice = input(
            'PREVIOUS MODEL FOUND, CONTINUE TRAINING OR OVERRIDE OR END? (ANSWER: "CONTINUE", "OVERRIDE", "END")\n')
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
        # if epoch % 1000 == 0:
        #     sgd_lr *= sgd_lr_decay
        start_time = time.time()
        for batch in iterate_minibatches(trX, trY, labeled_idx, run_parameters.batch_size, shuffle=True):
            inputs, targets, labeled = batch
            train_err = train_fn(inputs, targets, labeled)
        train_err = 0
        train_acc = 0
        train_batches = 0
        train_loss.append(0)
        train_loss1.append(0)
        train_loss2.append(0)
        train_regularize.append(0)
        train_wrong_samples = []
        train_wrong_classification = []
        for batch in iterate_minibatches(trX, trY, labeled_idx, run_parameters.batch_size, shuffle=True):
            inputs, targets, labeled = batch
            err, _loss1, _loss2, _regularize, train_classification, acc, _wrong_samples = val_fn(inputs, targets,
                                                                                                 labeled)
            train_loss[-1] += err
            train_loss1[-1] += _loss1
            train_loss2[-1] += _loss2
            train_regularize[-1] += _regularize
            train_err += err
            train_acc += acc
            train_wrong_samples += (_wrong_samples.nonzero()[0] + train_batches * run_parameters.batch_size).tolist()
            _wrong_classification = train_classification[_wrong_samples.nonzero()]
            train_wrong_classification += _wrong_classification.tolist()
            train_batches += 1
        train_loss[-1] /= train_batches
        train_loss1[-1] /= train_batches
        train_loss2[-1] /= train_batches
        train_regularize[-1] /= train_batches
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t{:.6f} %".format(
            train_acc / train_batches * 100))

        valid_err, valid_acc, valid_wrong_samples, valid_wrong_classification = run_test(val_fn, vlX, vlY, "validation")
        # save backup model every 10 epochs
        if epoch % 10 == 0:
            io.write_model_data([unsupervised_graph, supervised_graph], [best_validation_acc], LAST_MODEL_PATH)
        # if best model is found, save best model
        if valid_acc > best_validation_acc:
            print('NEW BEST MODEL FOUND!')
            best_validation_acc = valid_acc
            io.write_model_data([unsupervised_graph, supervised_graph], [best_validation_acc], BEST_MODEL_PATH)
            _, _, test_wrong_samples, test_wrong_classification = run_test(val_fn, teX, teY, "test")
            with open(WRONG_SAMPLES_PATH, 'w') as f:
                pickle.dump([train_wrong_samples, train_wrong_classification,
                             valid_wrong_samples, valid_wrong_classification,
                             test_wrong_samples, test_wrong_classification], f)
    # plot losses graph
    plt.clf()
    plt.plot(train_loss, 'r-')
    plt.plot(train_loss1, 'g-')
    plt.plot(train_loss2, 'b-')
    plt.plot(train_regularize, 'k-')
    plt.show()
