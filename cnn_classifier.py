# Author: Graham Roberts et al.
import argparse
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn import metrics
from PIL import Image
import matplotlib.pyplot as plt

#dictionaries map artifact types to pefixes. only usedx for IO
unlabeledStyles = {'dislocation': 'Dis', 'precipitate_void': 'Precip'}
labeledStyles = {'dislocation': 'disloc', 'precipitate': 'precip', 'void': 'void'}

#division matrix multiplied by unlabeled data to normalize
normalization_matrix = np.full((1024, 1024), (1.0 / 255.0))

#csv2matrix
#  function takes arguments specifying a csv file of data
#  args:
#  style: key to Styles dict for IO lookup
#  num:  quadrant of image 1-4
#  labels: boolean value True->labeled data, False->unlabeled data
#  Returns: numpy matrix representation of csv file
def csv2matrix(style, num, labels=False):
    if labels:
        fn = 'csvs/{0}{1}Out.csv'.format(num, labeledStyles[style])
    else:
        fn = 'csvs/{0}unlabeled{1}MidOut.csv'.format(num, unlabeledStyles[style])
    m = np.genfromtxt(fn, delimiter=',')
    return (m)

#load_data
#  function depreciated use load_data_with_dev_at
def pload_data(target):
    train_true = [csv2matrix(target, 1, True)] + \
                 [csv2matrix(target, 2, True)]
    unlabeled_train = create_unlabeled_train()
    dev_true = csv2matrix(target, 3, labels=True)
    unlabeled_dev = create_unlabeled_dev()
    dim = [len(dev_true[0]), 2, 1]
    return train_true, unlabeled_train, dev_true, unlabeled_dev, dim

#load_data_with_dev_at
#  function populates matrices with data from a directory of csvs
#  args:
#  target: the type of defect to be found
#  quadrant: the quadrant to be used as dev
#  returns:
#  several matrices of data labeled and unlabeled for train, dev, and test
#  quadrant 1 is always test
def load_data_with_dev_at(target, quadrant):
    if quadrant >= 4:
        quadrant = 1
    if quadrant == 1:
        train_true = [csv2matrix(target, 2, True)] + \
                     [csv2matrix(target, 3, True)]
        unlabeled_train = create_unlabeled_train(2,3)
        dev_true = csv2matrix(target, 4, labels=True)
        unlabeled_dev = create_unlabeled_dev(4)
        dim = [len(dev_true[0]), 2, 1]
    elif quadrant == 2:
        train_true = [csv2matrix(target, 2, True)] + \
                     [csv2matrix(target, 4, True)]
        unlabeled_train = create_unlabeled_train(2,4)
        dev_true = csv2matrix(target, 3, labels=True)
        unlabeled_dev = create_unlabeled_dev(2)
        dim = [len(dev_true[0]), 2, 1]
    else:
        train_true = [csv2matrix(target, 3, True)] + \
                     [csv2matrix(target, 4, True)]
        unlabeled_train = create_unlabeled_train(3,4)
        dev_true = csv2matrix(target, 2, labels=True)
        unlabeled_dev = create_unlabeled_dev(2)
        dim = [len(dev_true[0]), 2, 1]
    test_true = csv2matrix(target, 1, True)
    unlabeled_test = create_unlabeled_dev(1)
    return train_true, unlabeled_train, dev_true, unlabeled_dev, test_true, unlabeled_test, dim

#create_unlabeled_train
#  function creates a two layer image of unlabeled data
#  args:
#     1 & q2: the numbers of the quadrants used for train
#  returns:
#     list of np tensors
#  notes:
#     stacks dislocation unlabeled data twice, update to use both or an option of which image to use
def create_unlabeled_train(q1,q2):
    """Create the matrices used for training the model."""
    disloc_one = np.multiply(csv2matrix('dislocation', q1), normalization_matrix)
    disloc_two = np.multiply(csv2matrix('dislocation', q2), normalization_matrix)
    precip_void_one = np.multiply(csv2matrix('dislocation', q1), normalization_matrix)
    precip_void_two = np.multiply(csv2matrix('dislocation', q2), normalization_matrix)

    return [np.stack((disloc_one, precip_void_one), axis=2)] + \
           [np.stack((disloc_two, precip_void_two), axis=2)]

#create_unlabeled_dev
#  function creates a tensor of unlabeled data for the dev quadrant
#  args:
#     uad: the number of the quadrant to be used for dev
#  returns: an np tensor of unlabeled data
#  notes:
#     tacks dislocation image twice, update to use both images or take a choice of image as an argument
#     lso used to populate test
def create_unlabeled_dev(quad):
    """Create the matrices used to evaluate the trained model."""
    disloc = np.multiply(csv2matrix('dislocation', quad), normalization_matrix)
    precip_void = np.multiply(csv2matrix('dislocation', quad), normalization_matrix)

    return np.stack((disloc, precip_void), axis=2)

#activation dict
#a dictionary that maps text representations of activation functions to their tensorflow implimentations to be used in nn kernel construction
activation = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'identity': tf.identity, 'sigmoid': tf.nn.sigmoid}

# parse_all_args
#  function to parse args
#  notes:
#     replace arg for model num with arg defining file of architecture specs
#     train and dev need to somehow take multiple files and know which overlap and which are dev
#     also need to specify which quad is test
#     mbdt depreciated , maybe update to a boolean option as to whether or not to impliment any filtering
def parse_all_args():
    """
    Parses arguments

    :return: the parsed arguments object
    """
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("target",
                        help="The type of anomaly to identify")
    parser.add_argument("fn",
                        help="The filename of the architecture model")
    parser.add_argument("-lr",
                        type=float,
                        help="The learning rate (a float) [default = 0.001]",
                        default=0.001)
    parser.add_argument("-patience",
                        type=int,
                        help="The number of non-improving epochs to run",
                        default=10)
    parser.add_argument("-mb", type=int,
                        help="The minibatch size (an int) [default = 32]",
                        default=32)
    parser.add_argument("-epochs",
                        type=int,
                        help="The number of epochs to train for [default = 5]",
                        default=5)
    parser.add_argument('-giveup', type=int,\
                        help="Quit if we haven't exceeded best performance in giveup*patience epochs (int) [default=3]", default=3)
    parser.add_argument('-beta', type=float,\
                        help='The lr decrease factor (float) [default=0.8]', default=0.8)
    parser.add_argument("-mbdt",
                        type=float,
                        help="minibatch density threshold. Only minibatches with a higher percentage of positive labels are trained on [default=0.0]",
                        default=0.0)
    parser.add_argument("-image",
                        action='store_true',
                        help='Toggle image output')
    return parser.parse_args()


# build graph
#  function builds graph
#  args:
#     layers: a list of dictionaries of layers and their parameters, built by build_dict
#        each dictionary in the list contains the information required to build a layer
#     dim: a list of dimensions
#     mb: the dimension of the minibatches
#  variables:
#     x: a tensor for unlabeled data
#     y_true: a tensor for labeled data
#     weights: a matrix of weights, multiplied by cross entropy to correct for sparse data
#     threshold_test: a threshold used for binary classification
#     a: a list of layers used to feed into the next layer, can also be used to address specific layers if desired
#     K: a kernel for a convolutional layer.
#     w: weight matrix of kernel
#     b: bias of kernel
#     z:convolutional layer with kernel K stride 1 and SAME padding
#     z2: z + bias
#     aout: any identiy clone of the output of the final layer: useful when analyzing the probablistic return of the network
#     cross_entropy: sigmoid cross entropy loss function
#     area_under_roc: scalar area of ROC curve NOTE update operation must also be ran to fill with a nonzero value
#     update_op_roc: update operation that fills area_under_roc with a value
#     summary_roc: calls update op must fetch with area_under_roc if that is desired, may be useful in plotting area_under_roc as a function of epoch, nothing useful implimented yet
#     ce: weighted cross entropy [cross_entropy X weights] NOTE rename something more meaningful
#     obj: objective function redfuce mean of cross entropy
#     objw: objective funtion reduce mean of weighted cross entropy
#     train_step: adam optimizer train step of objective function depending on which version is commented at execution supports both weighed and unweighted NOTE rename the weighted option so both exist and can be run optionally
#     pred: a matrix of prediction 1 if probability greater than threshold 0 else
#     acc: number of pixel with predictions matching labels / numbre of pixels in batch
#     init: a function that initializes global and local variables (area_under_roc uses locals) must be fetched once to finish construction
def build_graph(layers, dim, mb):
    lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
    x = tf.placeholder(dtype=tf.float32,
                       shape=(None, mb, mb, dim[1]),
                       name='x')
    y_true = tf.placeholder(dtype=tf.float32,
                            shape=(None, mb, mb, dim[2]),
                            name='y_true')
    weights = tf.placeholder(dtype=tf.float32, shape=(None, mb, mb, dim[2]), name='weights')
    threshold_test = tf.placeholder(dtype=tf.float32,
                               shape=(None, mb, mb, dim[2]),
                               name='threshold_test')
    a = [x]
    x_in = tf.identity(x, name='x_in')
    for i, layer in enumerate(layers):
        ## No need for max pool since output dimension MUST equal input dimension
        with tf.variable_scope('variable_scope{}'.format(i + 1)):
            k = tuple([int(j) for j in layer['k'].split(',')])
            #         l=layer['l']
            #         lprime=layer['lprime']
            #         s=layer['s']
            #         p=layer['p']
            f = layer['f']
            if f == 'relu':
                b_const = 0.1
            else:
                b_const = 0.0
            act = activation[f]

            ##Create tf parameter tensors
            W = tf.get_variable(name='W',
                                shape=k,
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='b',
                                shape=(k[-1]),
                                initializer=tf.constant_initializer(b_const))
            z = tf.nn.conv2d(a[i], W, strides=[1, 1, 1, 1], padding="SAME")
            z2 = tf.contrib.layers.batch_norm(z, center=True, scale=True, is_training=True,scope='bn')
            z2 = z2 + b
            a.append(act(z))
    aout = tf.identity(a[-1], name='aout')
    #   wout=tf.identity(weights,name='wout')
    y = tf.cast(y_true, dtype=tf.int32, name='y')
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=a[-1], labels=y_true)
    area_under_roc, update_op_roc = tf.metrics.auc(labels=y_true, predictions=a[-1], num_thresholds=200, name='roc_area')
    summary_roc = tf.summary.scalar("aucroc_op", update_op_roc)
    obj_roc = tf.identity(area_under_roc, name='aucroc')
    ce = tf.multiply(weights,cross_entropy,name='ce')
    objw = tf.reduce_mean(ce, name='objw')
    obj = tf.reduce_mean(cross_entropy, name='obj')
    pred = tf.cast(tf.greater(a[-1], threshold_test), tf.int32, name='prediction')
    #train_step = tf.train.AdamOptimizer(lr).minimize(obj, name='train_step')
    train_step = tf.train.AdamOptimizer(lr).minimize(objw, name='train_step')
    acc = tf.reduce_mean(tf.cast(tf.equal(y, pred), tf.float32), name='acc')
    #   acc = tf.multiply(tf.reduce_mean(cross_entropy-1),-1,name='acc')
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return init


# build_dict
# function do take a file specifying the architecture of the model and return a dictionary of parameters
# args:
#  fn: the filename of an architechture file specified as an argument to the program
# return:
#  model: a list of dictionaries, each dict contains the parameters for a leyer mapped to by their argument
def build_dict(fn):
    model = []
    arch = open(fn, 'r')
    for l in arch.readlines():
        params = {}
        specs = l.rstrip().split('|')
        print(specs)
        for s in specs:
            p = s.split(':')
            params[p[0]] = p[1]
        model += [params]
    return model


# not currently relevant
# computes a threshold such that predicting any value > threshold produces most accurate results
def find_best_threshold(labels, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    tpr_diff=np.diff(tpr)
    fpr_diff=np.diff(fpr)
    dtdf=tpr_diff/fpr_diff
    best_idx=np.argmin(dtdf)
    p = np.sum(labels)
    total = len(labels)
    n = total - p
    tp = tpr * p
    fp = fpr * n
    tn = n - fp
    acc = np.add(tp, tn) / total
 #   best_idx = np.argmax(acc)
    return thresholds[best_idx]

#init_graph
# functions builds and initializes graph
# args: dim a vector of layer depths, depreciated to arbitrary hardcoded values elsewhere I believe
#  mb: the size of minibatches are mb X mb
#  layers: a list of dicts of layer parameters supplied by build_dict
#  args: system arguments
#  other arguments present in signature, but not called?
# build graph and fetches init
# returns:
#  sess: the tensorflow session object of the graph
def init_graph(y_train, x_train, y_dev, x_dev, dim, mb, layers, args):
    # threshold = 0.5 * np.ones((1, mb, mb, 1))  # threshold = 0.5 not updated
    #  num_class=np.sum(y_train)+np.sum(y_dev)
    #   num_tot=dim[0]**2*3
    #   base_const=math.log(num_tot/(num_tot-num_class))
    #   delta_mult=math.log(num_tot/num_class)-base_const
    init = build_graph(layers, dim, mb)

    sess = tf.Session()
    sess.run(fetches=[init])
    return sess

#display_output_image
# args:
#  ce: the output matrix named ce because that was the matrix of interest when this function was written, should be renamed
#  label: a string used to label the image? doesn't seem to be referenced
# anyway it cast 0-1 values to grayscale and shows an image
def display_output_image(ce, label=''):
    """Use PIL to display the CE matrix as an image."""
    ce = np.squeeze(ce)
    mult = np.zeros(ce.shape)
    mult.fill(255)
    ce = np.multiply(ce, mult)
    ce_image = Image.fromarray(ce)
    ce_image.show()

#store_new_output
# args:
#  existing: the current vector of pixels
#  new: the results of the most recent minibatch to be appended to the array
# Somehow we get output that matches somewhat what we expect not quite sure how this logic works
def store_new_output(existing, new):
    """Create a long array of ce results from minibatches"""
    if len(existing):
        existing.append(new)
        return existing
    else:
        return [new]


#single_array_from_all_outputs
# after all minibatches this turns existing from store_new_output into a single matrix to dislpay
# args:
#  stored: a list of minbatch matrices
# returns:
#  a matrix of all minibatches
def single_array_from_all_outputs(stored):
    """Generate the square image from the minibatched CE"""
    row_len = int(math.sqrt(len(stored)))
    each_row = []
    for i in range(row_len):
        each_row.append(stored[i * row_len: (i + 1) * row_len])
        each_row[i] = [np.squeeze(x[0]) for x in each_row[i]]
    long_rows = [np.concatenate(x, axis=1) for x in each_row]
    return np.concatenate(long_rows, axis=0)

#all_output_debugging
# I have no idea what this does. It looks like Tim wrote this when he was confused abput the shapes he was passing around when displaying images
# He seems to have gotten that working and this isn't current;y called
def all_output_debugging(all_ces):
    """Learn about the shape of my minibatch compatible image generation idea."""
    print('Debugging CE stuff...')
    print('len(all_ces): {}  len(all_ces[0]): {}  len(all_ces[0][0]): {}'.format(
        len(all_ces),
        len(all_ces[0]),
        len(all_ces[0][0])
    ))

#run_graph
# This function is the main body of the function
# I believe it was copied out of main to keep main short, but this should be broken into several functions as well
# args:
#  y_train: the labels for the training set
#  x_train: the unlabeled data for the training set
#  y_dev: the labels for the dev set
#  x_dev: the unlabeled data for the dev set
#  y_test: the labels for the test set
#  x_test: the unlabeled data for the test set
#  dim: a listn of layer dimensions
#  mb: the minibatch size might be redundant to pass as an argument if args is also an argument
#  layers: probably the dict of layers I don't think this is ever referenced
#  args: The arguments passed to the program
#  sess: the tensorflow session
# I'll add some mid-function comment blocks, because this function is way too long for a single top comment block to be of use
def run_graph(y_train, x_train, y_dev, x_dev, y_test, x_test, dim, mb, layers, args, sess):
# threshold is the threshold value used for actual binary final prediction
#  there is some code that allows this value to be updated depending on the findings with the roc curve.
#  It is difficult to algorithmically determine the BEST threshold, maybe make this value an argument
# lr_0 the base learning rate, reset to after succesful iteration, learning rate is updated as patience wears thin
# lr the learning rate value passed to the graph
# epochs_waited the number of epochs since an increase in performance
# beta argument in program call, used to update the learning rate as the solution converges
# x_batches a list of minibatches of unlabeled data
# y_bathces a list of mminibatches of labels for corresponding minibatches in x_batches
# batch densities a list of the densities of positive classifications in the minibatches computed as sum/size AKA. number positive / number total
# weights: a list of weight mask for weighted cross entropy calculation positive classifications are weighted more important to correct for sparsity and discourage trivial solution
#  weights are defined as follows 1/batch sparsity for pixels with negative labels, 1/batch density for positive labels
#  batch density should be smaller, so the weights will be larger
#  mask created as a matrix of entirely 1/batch sparsity plus an element wise (1/[batch density]-1/[batch sparsity]) * ylabel *1 for positive classification, zero for negative*
# batch_density: not to be confused with batch_densities. it is the density of positive classifications in the full batch
# delta: the difference in weights between the negative and positive classifications
# batch_sparsity: percent negative classifications in the full batch
# fpr: a dictionary of false positive rates
# tpr: a dictionary of true positive rates
# roc_auc: a dictionary of the area under the roc curves
    threshold = 0.1 * np.ones((1, mb, mb, 1))
    lr_0=args.lr
    lr=lr_0
    epochs_waited=0
    beta=args.beta
    x_batches = []
    y_batches = []
    batch_densities = []
    weights=[]
    batch_density = np.sum(y_train)/np.size(y_train)
    batch_sparsity = 1. - batch_density
    delta = 1./batch_density - 1./batch_sparsity
    base_weight = 1./batch_sparsity
    fpr = dict()
    roc_auc = dict()
    tpr = dict()
    # Creates minibatches of corresponding x_data and y_labels and densities
    # splits the total training set into mb X mb minibatches, calculates their densities, populates lists
    for i in range(len(x_train)):
        for j in range(int(np.floor(dim[0] / mb))):
            for k in range(int(np.floor(dim[0] / mb))):
                x_batch = x_train[i][j * mb:(j + 1) * mb, k * mb:(k + 1) * mb, :]
                y_batch = y_train[i][j * mb:(j + 1) * mb, k * mb:(k + 1) * mb]
                density = np.sum(y_batch) / mb ** 2
                weights_batch= delta*np.array(y_batch)+base_weight
                x_batches.append(x_batch)
                y_batches.append(y_batch)
                weights.append(weights_batch)
                batch_densities.append(density)
    badcount = 0
    epoch = 0
    best = -10.
    model_string=""
    # the training loop
    # terminates if the number of epochs since an improvement equals a given giveup dumber specified as an argument to the program
    #    or the total number of epochs trained equals a given number of total epochs specified as an argument to the program
    while (badcount+args.patience*epochs_waited) < (args.patience*args.giveup) and epoch < args.epochs:
        all_outputs = []
        mbxt = []
        mbyt = []
        mbwt = []
        mb_num=0
        # loop populates a training sequence with minibatches
        # generates a random number in [0,1] for each minibatch
        # density also in [0,1]
        # if random number < density batch is included
        # probability of inclusion equal to density, thus denser minibatches are more likely to be included
        # continues if fewer than ten minibatches are selected should probably make this number an argument
        # also should make this behaviour optional
        while(mb_num < 10):
           for i in range(len(batch_densities)):
              rd=np.random.random()
              if(rd<batch_densities[i]):
                 mbxt.append(x_batches[i])
                 mbyt.append(y_batches[i])
                 mbwt.append(weights[i])
                 mb_num+=1
        epoch += 1
        mbxt=np.array(mbxt)
        mbyt=np.array(mbyt)
        mbwt=np.array(mbwt)
        idx = np.random.permutation(mb_num)
        mbxt = mbxt[idx]
        mbyt = mbyt[idx]
        mbwt = mbwt[idx]
        #       weights=weights[idx]
        #thresh_guess = 0
        j = 0
        saver = tf.train.Saver() #Adds model saving
        # iterates through all selected minibatches in a random order
        #trains model on each
        for l in range(mb_num):
            x_t = np.reshape(mbxt[l], (-1, mb, mb, dim[1]))
            y_t = np.reshape(mbyt[l], (-1, mb, mb, dim[2]))
            w_t = np.reshape(mbwt[l], (-1, mb, mb, dim[2]))
            label_vec = np.reshape(y_t, (mb * mb))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            tf.control_dependencies(update_ops)
            [_] = sess.run(fetches=['train_step'],
                    feed_dict={'x:0': x_t, 'y_true:0': y_t, 'threshold_test:0': threshold, 'lr:0': lr, 'weights:0':w_t})
   # second run doesn't fetch train step so model is not updated, diagnostic information fetched, but no longer referenced
   # useful if updating threshold?
            [x_in, aout, yy, my_obj, my_acc] = sess.run(fetches=['x_in:0', 'aout:0', 'y:0', 'obj:0', 'acc:0'],
                                                        feed_dict={'x:0': x_t, 'y_true:0': y_t,
                                                            'threshold_test:0': threshold, 'lr:0': lr, 'weights:0':w_t})
            # print('train_acc={0}\t\ttrain_obj={1}\t\tmbt={2}'.format(my_acc, my_obj,np.sum(mbyt[l])/mb**2))
            guess_vec=np.reshape(aout,(mb*mb))
  #          thresh_guess+=find_best_threshold(label_vec, guess_vec)
  #          j+=1
  #      threshold_value=thresh_guess/j
  #      threshold=threshold_value*np.ones((1,mb,mb,1))
            fpr, tpr, roc_auc = metrics.roc_curve(label_vec, guess_vec)
            roc_auc = metrics.auc(fpr, tpr)
        i = 0
        # loops over dev set
        # calculates the value of the objective function and the accuracy for each iteration
        # of these values, each the mean of a minibatch is the mean of the full batch
        dev_acc = 0
        dev_obj = 0
        for j in range(int(np.floor(dim[0] / mb))):
            for k in range(int(np.floor(dim[0] / mb))):
                x_d = np.reshape(x_dev[mb * j:mb * (j + 1), mb * k:mb * (k + 1)], (-1, mb, mb, dim[1]))
                y_d = np.reshape(y_dev[mb * j:mb * (j + 1), mb * k:mb * (k + 1)], (-1, mb, mb, dim[2]))
                w_d = np.reshape((y_dev[mb*j:mb*(j+1),mb*k:mb*(k+1)]),(-1, mb, mb, dim[2]))*delta+base_weight
                _, my_dev_acc, my_dev_obj, output = sess.run(fetches=['aucroc_op:0', 'aucroc:0', 'obj:0', 'aout:0'],
                                                          feed_dict={'x:0': x_d, 'y_true:0': y_d,
                                                              'threshold_test:0': threshold, 'lr:0': lr, 'weights:0':w_d})
                all_outputs = store_new_output(all_outputs, output)
                dev_acc += my_dev_acc
                dev_obj += my_dev_obj
                i += 1
        dev_acc /= i
        dev_obj /= i

        check_if_should_bail(dev_acc, best)
        # check for improvement saves best and resets variables on improvement
        # otherwise updates counters and decrements learning rate if neccesary
        # displays diagnostic information of model performance, and an image
        if dev_acc > best:
            model_string="models/model_%s%d.ckpt" % (args.target, epoch)
            saver.save(sess, "models/model_%s%d.ckpt" % (args.target, epoch), global_step=epoch)
            badcount = 0
            lr=lr_0
            epochs_waited=0
            best = dev_acc
            big_output = single_array_from_all_outputs(all_outputs)
        else:
            badcount += 1
            if(badcount>=args.patience):
                epochs_waited+=1
                badcount=0
                lr*=beta
        print('epoch {0}: roc={1} obj={2} badcount={3} best={4} {5}'.format(epoch, dev_acc, dev_obj, badcount, best, dev_acc-best))
    print("best = {:10f}, saved in file {}".format(best, model_string))
    if args.image:
        display_output_image(big_output)

    test_image_outputs=[]
    test_predictions_outputs=[]
    i=0
    # iterates over minibatches of test set and computes objective and accuracy
    # identical logic to dev set, but only ran after conversion
    test_acc=0
    test_obj=0
    saver.restore(sess, 'models/model_dislocation1.ckpt-1')
    for j in range(int(np.floor(dim[0] / mb))):
        for k in range(int(np.floor(dim[0] / mb))):
            x_test_mb = np.reshape(x_test[mb * j:mb * (j + 1), mb * k:mb * (k + 1)], (-1, mb, mb, dim[1]))
            y_test_mb = np.reshape(y_test[mb * j:mb * (j + 1), mb * k:mb * (k + 1)], (-1, mb, mb, dim[2]))
            w_test_mb = np.reshape(y_test[mb * j:mb * (j + 1), mb * k:mb * (k + 1)], (-1, mb, mb, dim[2]))*delta+base_weight

            my_test_acc, my_test_obj, a_output, pred_output= sess.run(fetches=['acc:0', 'obj:0', 'aout:0', 'prediction:0'],
                                                        feed_dict={'x:0': x_test_mb, 'y_true:0': y_test_mb,
                                                            'threshold_test:0': threshold, 'lr:0': lr, 'weights:0':w_test_mb})
            test_image_outputs = store_new_output(test_image_outputs, a_output)
            test_predictions_outputs = store_new_output(test_predictions_outputs, pred_output)
            test_acc += my_test_acc
            test_obj += my_test_obj
            i += 1
    test_acc/=i
    test_obj/=i
    # optionally display images of binary classification and grayscale probablistic classification, useful when observing model certainty
    if args.image:
        big_test_image_output = single_array_from_all_outputs(test_image_outputs)
        big_test_predictions_output = single_array_from_all_outputs(test_predictions_outputs)
        display_output_image(big_test_image_output)
        display_output_image(big_test_predictions_output)
    print('TEST: dev={0} obj={1}'.format(test_acc, test_obj))

    # plot the roc curve
    # I think this uses the most recent minibatch of the training set
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def check_if_should_bail(acc, best):
    """If the last accuracy and the best accuracy are within 0.000001, it probably
       means the model is returning garbage. Fail fast!
    """
    if abs(best - acc) < 0.000001:
        print('ABORT DUE TO IDENTICAL ACCURACIES')
        exit(0)


# main
# builds graph and initializes it
# uses k-fold training
# quad 1 is always test
# quad 2 (third iteration) is the dev set of all recorded results
# note it looks like we rebuild the batches from the images on the hard drive every single iteration (3 more times than neccesary)
def main():
    args = parse_all_args()
    # for quads in range (1,4):
    mb = args.mb
    layers = build_dict(args.fn)

    y_train, x_train, y_dev, x_dev, y_test, x_test, dim = load_data_with_dev_at(args.target, 1)
    sess = init_graph(y_train, x_train, y_dev, x_dev, dim, mb, layers, args)
    for quads in range(1, 4):
        y_train, x_train, y_dev, x_dev, y_test, x_test, dim = load_data_with_dev_at(args.target, quads)
        run_graph(y_train, x_train, y_dev, x_dev, y_test, x_test, dim, mb, layers, args, sess)


if __name__ == "__main__":
    main()
