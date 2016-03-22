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
from path_settings import BEST_MODEL_PATH, LAST_MODEL_PATH, WRONG_SAMPLES_PATH, DATA_PATH, \
    UNSUPERVISED_PRETRAIN_PATH, SUPERVISED_PRETRAIN_PATH
from sparse import get_all_sparse_layers
from utils import normalize_zero_one


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
with open(os.path.join(DATA_PATH, 'labeled_index.pkl'), 'r') as f:
    labeled_idx = pickle.load(f)

# Load image and label of train, validation, test set
trX, vlX, teX, trY, vlY, teY = mnist(onehot=True, normalize_axes=None, ndim=2)
IM_SIZE = trX.shape[1]

# -----------------------SET PARAMETERS-------------------------#
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
reconstruction, prediction = layers.get_output([unsupervised_graph, supervised_graph])
# Test graph has no dropout so deterministic = True
test_reconstruction, test_prediction = layers.get_output([unsupervised_graph, supervised_graph], deterministic=True)
if run_parameters.clip_unsupervised_output is not None:
    reconstruction = T.clip(reconstruction, run_parameters.clip_unsupervised_output[0],
                            run_parameters.clip_unsupervised_output[1])
    test_reconstruction = T.clip(test_reconstruction, run_parameters.clip_unsupervised_output[0],
                                 run_parameters.clip_unsupervised_output[1])

# Get all trainable params
params = layers.get_all_params(unsupervised_graph, trainable=True) + \
         layers.get_all_params(supervised_graph, trainable=True)
# params = layers.get_all_params(supervised_graph)[-2:]
params = utils.unique(params)

# Get regularizable params
regularization_params = layers.get_all_params(unsupervised_graph, regularizable=True) + \
                        layers.get_all_params(supervised_graph, regularizable=True)
regularization_params = utils.unique(regularization_params)

# Creating loss functions
# Train loss has to take into account of labeled image or not
if run_parameters.unsupervised_cost_fun == 'squared_error':
    loss1 = objectives.squared_error(reconstruction, input_var)
elif run_parameters.unsupervised_cost_fun == 'categorical_crossentropy':
    loss1 = objectives.categorical_crossentropy(reconstruction, input_var)
if supervised_cost_fun == 'squared_error':
    loss2 = objectives.squared_error(prediction, target_var) * repeat_col(labeled_var, 10)
elif supervised_cost_fun == 'categorical_crossentropy':
    loss2 = objectives.categorical_crossentropy(prediction, target_var) * labeled_var.T
l2_penalties = regularization.apply_penalty(regularization_params, regularization.l2)
sparse_layers = get_all_sparse_layers(unsupervised_graph)
sparse_layers_output = layers.get_output(sparse_layers, deterministic=True)
# sparse_regularizer = reduce(lambda x, y: x + T.clip((T.mean(abs(y)) - run_parameters.sparse_regularize_factor) * y.size,
#                                                     0, float('inf')),
#                             sparse_layers_output, 0)
sparse_regularizer = reduce(lambda x, y: x + T.clip(T.mean(y, axis=1) - run_parameters.sparse_regularize_factor, 0,
                                                    float('inf')).sum() * y.shape[1],
                            sparse_layers_output, 0)

loss = losses_ratio[0] * loss1.mean() + \
       losses_ratio[1] * loss2.mean() + \
       losses_ratio[2] * l2_penalties.mean() + \
       losses_ratio[3] * sparse_regularizer
# Test loss means 100% labeled

if run_parameters.unsupervised_cost_fun == 'squared_error':
    test_loss1 = objectives.squared_error(test_reconstruction, input_var)
elif run_parameters.unsupervised_cost_fun == 'categorical_crossentropy':
    test_loss1 = objectives.categorical_crossentropy(test_reconstruction, input_var)
if supervised_cost_fun == 'squared_error':
    test_loss2 = objectives.squared_error(test_prediction, target_var)
elif supervised_cost_fun == 'categorical_crossentropy':
    test_loss2 = objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = losses_ratio[0] * test_loss1.mean() + \
            losses_ratio[1] * test_loss2.mean() + \
            losses_ratio[2] * l2_penalties.mean() + \
            losses_ratio[3] * sparse_regularizer

# Compute gradient in case of gradient clipping
if run_parameters.clip_gradient is not None:
    grad = T.grad(loss, params)
    if run_parameters.clip_gradient[0] is True:  # softclip
        grad = [updates.norm_constraint(g, run_parameters.clip_gradient[1], range(g.ndim)) for g in grad]
    else:
        grad = [T.clip(g, run_parameters.clip_gradient[0], run_parameters.clip_gradient[1]) for g in grad]
    loss = grad

# Update function to train
# sgd_lr = run_parameters.update_lr
sgd_lr = theano.shared(utils.floatX(run_parameters.update_lr))
sgd_lr_decay = utils.floatX(0.5)
sgd_lr_decay_threshold = utils.floatX(1.0)
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
# val_fn = theano.function([input_var, target_var, labeled_var], [loss2*lr[1], test_acc], allow_input_downcast=True)
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
    if os.path.isfile(LAST_MODEL_PATH):
        choice = input(
            'PREVIOUS MODEL FOUND, CONTINUE TRAINING OR OVERRIDE OR END? (ANSWER: "CONTINUE", "OVERRIDE", "END")\n')
        if choice == 'CONTINUE':
            best_validation_acc, old_lr = io.read_model_data([unsupervised_graph, supervised_graph], LAST_MODEL_PATH)
            sgd_lr.set_value(old_lr)
        elif choice == 'OVERRIDE':
            best_validation_acc = 0
        else:
            sys.exit('Terminated by user choice.')
    if run_parameters.load_pretrain_unsupervised:
        io.read_model_data([unsupervised_graph], UNSUPERVISED_PRETRAIN_PATH)
    if run_parameters.load_pretrain_supervised:
        io.read_model_data([supervised_graph], SUPERVISED_PRETRAIN_PATH)
    print 'Training...'
    # number of epochs to train
    num_epochs = run_parameters.num_epochs
    train_loss = []
    train_loss1 = []
    train_loss2 = []
    train_regularize = []
    last_loss = float('inf')
    num_iter = 1
    for epoch in range(num_epochs):
        start_time = time.time()
        # NORMALIZE DICTIONARY AFTER 1 EPOCH:
        if run_parameters.normalize_dictionary_after_epoch is not None:
            for sparse_layer in sparse_layers[0:len(sparse_layers) / 2]:
                D_ref = sparse_layer.get_dictionary()
                D = D_ref.eval()
                sparse_layer.set_dictionary(normalize_zero_one(D, run_parameters.normalize_dictionary_after_epoch))
        # RUN TRAIN
        for batch in iterate_minibatches(trX, trY, labeled_idx, run_parameters.batch_size, shuffle=True):
            inputs, targets, labeled = batch
            sgd_lr.set_value(1 / num_iter)
            num_iter += 1
            train_err = train_fn(inputs, targets, labeled)
        train_err = 0
        train_acc = 0
        train_batches = 0
        train_wrong_samples = []
        train_wrong_classification = []
        for batch in iterate_minibatches(trX, trY, labeled_idx, run_parameters.batch_size, shuffle=True):
            inputs, targets, labeled = batch
            err, _loss1, _loss2, _regularize, train_classification, acc, _wrong_samples = val_fn(inputs, targets,
                                                                                                 labeled)
            train_loss.append(err)
            train_loss1.append(_loss1)
            train_loss2.append(_loss2)
            train_regularize.append(_regularize)
            train_err += err
            train_acc += acc
            train_wrong_samples += (_wrong_samples.nonzero()[0] + train_batches * run_parameters.batch_size).tolist()
            _wrong_classification = train_classification[_wrong_samples.nonzero()]
            train_wrong_classification += _wrong_classification.tolist()
            train_batches += 1
        # if train_err > last_loss * sgd_lr_decay_threshold:
        #     sgd_lr.set_value((sgd_lr * sgd_lr_decay).eval())
        last_loss = train_err
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t{:.6f} %".format(
            train_acc / train_batches * 100))

        valid_err, valid_acc, valid_wrong_samples, valid_wrong_classification = run_test(val_fn, vlX, vlY, "validation")
        # save backup model every 10 epochs
        if epoch % 10 == 0:
            io.write_model_data([unsupervised_graph, supervised_graph], [best_validation_acc, sgd_lr.get_value()],
                                LAST_MODEL_PATH)
        # if best model is found, save best model
        if valid_acc > best_validation_acc:
            print('NEW BEST MODEL FOUND!')
            best_validation_acc = valid_acc
            io.write_model_data([unsupervised_graph, supervised_graph], [best_validation_acc, sgd_lr.get_value()],
                                BEST_MODEL_PATH)
            _, _, test_wrong_samples, test_wrong_classification = run_test(val_fn, teX, teY, "test")
            with open(WRONG_SAMPLES_PATH, 'w') as f:
                pickle.dump([train_wrong_samples, train_wrong_classification,
                             valid_wrong_samples, valid_wrong_classification,
                             test_wrong_samples, test_wrong_classification], f, pickle.HIGHEST_PROTOCOL)
    # plot losses graph
    plt.clf()
    plt.plot(train_loss, 'r-')
    plt.plot(train_loss1, 'g-')
    plt.plot(train_loss2, 'b-')
    plt.plot(train_regularize, 'k-')
    plt.show()
