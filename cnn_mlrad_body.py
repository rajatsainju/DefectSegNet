#cnn_mlrad_body
#The main portion of the cnn_mlrad_project
# Author: Graham Roberts et al.
# Updated: 04 July 2019
import argparse
import os
import time
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn import metrics
from PIL import Image
import time
import re
from matplotlib import pyplot as plt
import cv2

#custom modules
import cnn_dataloader as dl
import cnn_datastructs as ds

#activation dict
#a dictionary that maps text representations of activation functions to their tensorflow implimentations to be used in nn kernel construction
activation = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'identity': tf.identity, 'sigmoid': tf.nn.sigmoid}

# create max pool layer
# adds a max pool layer to the graph
def create_max_pool_layer(layer, regularizer, args, a, i, is_training):
   k = tuple([int(j) for j in layer['kernel'].split(',')])
   s = int(layer['strides'])
   in_layers = [int(j) for j in layer['input tensors'].split(',')]
   conn_list = []
   shortcut_names=[]
   for il in in_layers:
      conn_list += [a[il]]
   input_tensor = tf.concat([connect_tensor for connect_tensor in conn_list], axis = 3)
   if 'padding' in layer:
     p = layer['padding']
   else:
     p = 'SAME'
   z = tf.layers.max_pooling2d(input_tensor, pool_size = k, strides = s)
   return(z)

#custom pad
# creates padding around an image so that it mai9ntains its original size after being convolved

def custom_pad(input_tensor, k):
   paddings = [[0,0],[0,0],[0,0],[0,0]]
   padding_num0 = max(0,int((k[0]-1)/2))
   if k[0] %2 == 0:
       paddings[1][0] = padding_num0 + 1
       paddings[1][1] = padding_num0
   else:
       paddings[1][0] = padding_num0
       paddings[1][1] = padding_num0
   padding_num1 = max(int((k[1]-1)/2),0)
   if k[1] %2 == 0:
       paddings[2][0] = padding_num1 + 1
       paddings[2][1] = padding_num1
   else:
       paddings[2][0] = padding_num1
       paddings[2][1] = padding_num1
   paddings_tensor = tf.constant(paddings)
   input_tensor = tf.pad(input_tensor, paddings_tensor, mode='SYMMETRIC')
   return(input_tensor)

# adds a convolutional leyer to the graph
def create_conv_layer(layer, regularizer, args, a, i, is_training):
    k = tuple([int(j) for j in layer['kernel'].split(',')])
    nf = int(layer['filters'])
    f = layer['activation function']
    s = tuple([int(j) for j in layer['strides'].split(',')])
    in_layers = [int(j) for j in layer['input tensors'].split(',')]
    conn_list = []
    shortcut_names = []
    for il in in_layers:
         conn_list += [a[il]]
    input_tensor = tf.concat([connect_tensor for connect_tensor in conn_list], axis = 3)
    if f == 'relu':
        b_const = 0.1
    else:
        b_const = 0.0
    act = activation[f]
    if 'padding' in layer:
      input_tensor = custom_pad(input_tensor, k)
      p = 'VALID'
    else:
      p = 'SAME'
    z = tf.layers.conv2d(input_tensor, kernel_size = k, filters = nf, strides=s, padding=p, activation=act, use_bias=True, kernel_regularizer = regularizer)
    return(z)

# add a convolutional transpose layer
# useful for increasing the sample size of the image
#def create_conv_transpose(layer, regularizer, a, i, ds_count):
def create_conv_transpose(layer, regularizer, args, a, i, is_training):
   k = tuple([int(j) for j in layer['kernel'].split(',')])
   s = tuple([int(j) for j in layer['strides'].split(',')])
   f = int(layer['filters'])
   in_layers = [int(j) for j in layer['input tensors'].split(',')]
   conn_list = []
   shortcut_names = []
   for il in in_layers:
         conn_list += [a[il]]
   input_tensor = tf.concat([connect_tensor for connect_tensor in conn_list], axis = 3)
   if 'padding' in layer:
      p = layer['padding']
   else:
      p = 'SAME'
   z = tf.layers.conv2d_transpose(input_tensor, f, k, s, padding=p, kernel_regularizer = regularizer)
   return(z)

# adds a dense layer
#def create_dense_layer(layer, regularizer, a, i, ds_count):
def create_dense_layer(layer, regularizer, args, a, i, is_training):
   in_layers = [int(j) for j in layer['input tensors'].split(',')]
   conn_list = []
   for il in in_layers:
         il = in_layers[ind]
         conn_list += [a[il]]
   input_tensor = tf.concat([connect_tensor for connect_tensor in conn_list], axis = 3)
   f = layer['activation function']
   act = activation[f]
   if 'size' in layer:
      unit_shape = layer['size']
   else:
      unit_shape = input_tensor.shape()
   z = tf.layers.dense(input_tensor, unit_shape, activation = act, kernel_regularizer = regularizer)
   return(z)
   
def create_dropout_layer(layer, regularizer, args, a, i, is_training):
   in_layers = [int(j) for j in layer['input tensors'].split(',')]
   if len(in_layers) > 1:
      input_tensor = tf.concat([a[j] for j in in_layers], axis=3)
   else:
      input_tensor = a[in_layers[0]]
   rate = float(args['dropout_rate'])
   z = tf.layers.dropout(input_tensor, rate=rate, training = is_training)
   return(z)

layer_constructor_dict = {'convolution': create_conv_layer,
                          'max pool': create_max_pool_layer,
                          'convolution transpose': create_conv_transpose,
                          'dense': create_dense_layer,
                          'dropout': create_dropout_layer}
# Creates a layer and adds it to the graph
# reads the input layers dictionaty
# depending on the type calls a different function to build the graph
def create_layer(layer, a, i, args, dim, is_training, regularizer):
   z = layer_constructor_dict[layer['type']](layer, regularizer, args, a, i, is_training)
   if args['batch_norm'] == True:
       #z = tf.contrib.layers.batch_norm(z, center=True, scale=True, is_training=True,scope='bn_{0}'.format(i))
       z = tf.contrib.layers.batch_norm(z, center=True, scale=True, is_training=is_training,scope='bn_{0}'.format(i))
   a.append(z)
   return(a)

# build graph
#  function builds graph
#  args:
#     layers: a list of dictionaries of layers and their parameters, built by build_dict
#        each dictionary in the list contains the information required to build a layer
#     dim: a list of dimensions
#     edge_size: the dimension of the minibatches
#     objectives: The layers of each target
#     targets: The targets to be identified
#     args: The arguments passed to the program
#  variables:
#     x: a tensor for unlabeled data
#     y_true: a tensor for labeled data
#     weight: a scalar of weight, multiplied by cross entropy to correct for sparse data
#     a: a list of layers used to feed into the next layer, can also be used to address specific layers if desired
#     The following aspects exist within a dictionary for each desired target
#        cross_entropy: sigmoid cross entropy loss function
#        area_under_roc: scalar area of ROC curve NOTE update operation must also be ran to fill with a nonzero value
#        update_op_roc: update operation that fills area_under_roc with a value
#        summary_roc: calls update op must fetch with area_under_roc if that is desired, may be useful in plotting area_under_roc as a function of epoch, nothing useful implimented yet
#        ce: weighted cross entropy [cross_entropy X weights] NOTE rename something more meaningful
#        obj: objective function redfuce mean of cross entropy
#        train_step: adam optimizer train step of objective function depending on which version is commented at execution supports both weighed and unweighted NOTE rename the weighted option so both exist and can be run optionally
#     init: a function that initializes global and local variables (area_under_roc uses locals) must be fetched once to finish construction
#def build_graph(layers, dim, mb, raw_objectives, targets, args):
def build_graph(layers, dim, edge_size, objectives, targets, args):
    ds_count = 0
    shortcut_names = []
    l2_body = tf.contrib.layers.l2_regularizer(scale=args['reg_scale'])
    l2_last = tf.contrib.layers.l2_regularizer(scale=args['last_reg_scale'])
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
    x = tf.placeholder(dtype=tf.float32,
                       shape=(None, edge_size, edge_size, dim),
                       name='x')
    y_true = {}
    for t in targets:
       y_true[t] = tf.placeholder(dtype=tf.float32, shape=(None, edge_size, edge_size, 1), name='y_true_{0}'.format(t))
    a = [x]
    x_in = tf.identity(x, name='x_in')
    residual_weight = tf.placeholder(dtype=tf.float32, shape=[], name='residual_weight')
    index_of_layer=0
    for i, layer in enumerate(layers):
        with tf.variable_scope('variable_scope{}'.format(i + 1)):
            if i ==len(layers)-1:
               l2 = l2_body
            else:
               l2 = l2_last
            a = create_layer(layer, a, i, args, dim, is_training, l2)
        index_of_layer = i
    index_of_layer += 1
       
    aout = tf.identity(a[-1], name='aout')
    objective_list = []
    output_list = []
    weight_val = tf.placeholder(dtype=tf.float32, shape=[], name = 'ce_weight')
    weight_mats = ds.init_target_dict(targets)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
       for t in targets:
         weight_mats[t] = tf.placeholder(dtype=tf.float32, shape = (None, edge_size, edge_size, 1), name = 'weights_{0}'.format(t))
         cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = a[objectives[t]], labels=y_true[t], name='objective_{0}'.format(t))
         ce = tf.multiply(y_true[t], cross_entropy, name='objective_{0}'.format(t))
         #ce = tf.add(tf.multiply(tf.multiply(weight_mats[t], cross_entropy, name='objective_{0}'.format(t)), ce_weight), cross_entropy)
         area_under_roc, update_op_roc = tf.metrics.auc(labels=y_true[t], predictions=a[objectives[t]], num_thresholds=200, name='roc_area')
         summary_roc = tf.summary.scalar("aucroc_op_{0}".format(t), update_op_roc)
         obj_roc = tf.identity(area_under_roc, name='aucroc_{0}'.format(t))
         objective_list.append(tf.train.AdamOptimizer(lr).minimize(tf.reduce_mean(ce), name='train_step_{0}'.format(t)))
         output_list.append(tf.identity(a[objectives[t]], name='out_{0}'.format(t)))
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return init, ds_count, shortcut_names




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
def init_graph(dim, edge_size, layers, objectives, targets, args):
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config = cfg)
    init, ds_count, shortcut_names = build_graph(layers, dim, edge_size, objectives, targets, args)
    sess.run(fetches=[init])
    return sess, shortcut_names


def make_train_weight_matrix(y_t, alpha):
   expanded_mat = np.zeros((y_t.shape[0], y_t.shape[1]+2, y_t.shape[2]+2, y_t.shape[3]))
   expanded_mat[:,0,0,:] = y_t[:,0,0,:]
   expanded_mat[:,0,-1,:] = y_t[:,0,-1,:]
   expanded_mat[:,-1,0,:] = y_t[:,-1,0,:]
   expanded_mat[:,-1,-1,:] = y_t[:,-1,-1,:]
   expanded_mat[:,1:-1,0,:] = y_t[:,:,0,:]
   expanded_mat[:,1:-1,-1,:] = y_t[:,:,-1,:]
   expanded_mat[:,0,1:-1,:] = y_t[:,0,:,:]
   expanded_mat[:,-1,1:-1,:] = y_t[:,-1,:,:]
   expanded_mat[:,1:-1,1:-1,:] = y_t[:,:,:,:]
   for i in range(expaned_mat.shape[0]):
      plt.imshow(expanded_mat[i,:,:,0])
      plt.title('expanded_mat %d'%(i))
      plt.show()
   weight_mat = np.zeros(y_t.shape)
   weight_mat = weight_mat+expanded_mat[:,:-2,:-2,:]
   weight_mat = weight_mat+expanded_mat[:,1:-1,:-2,:]
   weight_mat = weight_mat+expanded_mat[:,2:,:-2,:]
   weight_mat = weight_mat+expanded_mat[:,:-2,1:-1,:]
   weight_mat = weight_mat+expanded_mat[:,1:-1,1:-1,:]
   weight_mat = weight_mat+expanded_mat[:,2:,1:-1,:]
   weight_mat = weight_mat+expanded_mat[:,:-2,2:,:]
   weight_mat = weight_mat+expanded_mat[:,1:-1,2:,:]
   weight_mat = weight_mat+expanded_mat[:,2:,2:,:]
   weight_mat = weight_mat.astype(float)*alpha/9.
   weight_mat = weight_mat+np.ones(weight_mat.shape)
   for i in range(weight_mat.shape[0]):
      plt.imshow(weight_mat[i,:,:,0])
      plt.title('weight mat %d'%(i))
      plt.show()
   return(weight_mat)

# run train loop
# Iterates over a random set of the train set
# uses a depth af mb_depth from the args
# iterates tran_num times
def run_train_loop(sess, x_train, y_train, targets, dim, lr, weight, time_stamp, args, eval_train_recent = False):
#def run_train_loop(sess, x_train, y_train, targets, dim, lr, weight, time_stamp, args, shortcut_names, eval_train_recent = False):
   rw = args['residual_weight']
   edge_size = args['segment_size']
   mbyt = {}
   mb_depth = args['mb_depth']
   outputs = {}
   labels = {}
   for l in range(args['train_num']):
      x_t, mbyt = dl.augment_train(x_train, y_train, targets, (1024, 1024), args)
      fd ={'x:0':x_t, 'lr:0':lr, 'is_training:0':True, 'residual_weight:0':rw}
      train_fetches = []
      weight_matrices = ds.init_target_dict(targets)
      for t in targets:
         y_t = mbyt[t]
         labels[t] = np.reshape(y_t, (-1,edge_size,edge_size))
         if args['weight_kernel']:
            weight_matrices[t] = make_train_weight_matrix(y_t, weight)
         else:
            weight_matrices[t] = np.ones(y_t.shape)+weight*y_t[:,:,:,:]
         train_fetches.append('train_step_{0}'.format(t))
         fd['y_true_{0}:0'.format(t)] = y_t
         fd['weights_{0}:0'.format(t)] = weight_matrices[t]
    #  fd['ce_weight:0'] = weight
      sess.run(fetches=train_fetches, feed_dict=fd)
      if eval_train_recent:
         print('eval recent')
         eval_fd = fd.copy()
         eval_fd['is_training:0'] = False
         eval_fetches = ['out_{0}:0'.format(t) for t in targets]
         return_list = sess.run(fetches=eval_fetches, feed_dict = fd)
         outputs = ds.init_target_dict(targets)
         for k in range(len(targets)):
            t = targets[k]
            outputs[t] = np.reshape(return_list[k], (-1,edge_size,edge_size,1))
         train_recent_stats = ds.calc_stats(outputs, mbyt, targets, 512*512)
   return(outputs, mbyt)

#loads the entire dev, test, or half of train into the graph
#receives output prediction in {0,1}
#identifies roc data
def run_eval_loop(sess, x, y, targets, edge_size, lr, dim, time_stamp, args):
   not_training = False
   outputs = ds.init_target_dict(targets) 
   labels = ds.init_target_dict(targets)
#   auc = ds.init_target_dict(targets)
   fetches = ['out_{0}:0'.format(t) for t in targets]
   x_d = ds.image_to_tensor(np.array(x), edge_size)
   val_fd={'x:0':x_d,'lr:0':0,'is_training:0':True,'residual_weight:0':args['residual_weight']}
   y_d = {}
   for t in targets:
      y_d[t] = ds.image_to_tensor(np.array(y[t]), edge_size)
      val_fd['y_true_{0}:0'.format(t)] = y_d[t]
   print(fetches)
   return_list = sess.run(fetches=fetches, feed_dict=val_fd)
   for k in range(len(targets)):
      t=targets[k]
      outputs[t] = return_list[k]
      labels[t] = y_d[t]
   
   return(outputs, labels, x_d)


def init_saver(sess, args):
    saver = tf.train.Saver()
    if args['checkpoint_fn'] is not None:
       try:
         saver.restore(sess, args['checkpoint_fn'])
         print("loaded {0} succesfully or at least didn't not work".format(args['checkpoint_fn']))
       except:
         print('Error reloading checkpoint from {0} continuing from scratch'.format(args['checkpoint_fn']))
    #saver.restore(sess, args['checkpoint_fn'])
    return(saver)

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
def run_graph(y_train, x_train, y_val, x_val, y_test, x_test, dim, args, sess, saver, time_stamp, targets):
# lr_0 the base learning rate, reset to after succesful iteration, learning rate is updated as patience wears thin
# lr the learning rate value passed to the graph
# epochs_waited the number of epochs since an increase in performance
# beta argument in program call, used to update the learning rate as the solution converges
# x_batches a list of minibatches of unlabeled data
# y_bathces a list of mminibatches of labels for corresponding minibatches in x_batches
# batch densities a list of the densities of positive classifications in the minibatches computed as sum/size AKA. number positive / number total
    os.makedirs("../results/"+args['experiment_filename'].split(".")[0]+"/{0}".format(time_stamp))
    output_dirname = "../results/%s/%s"%(args['experiment_filename'].split('.')[0],time_stamp)
    iu_outfile=open("../results/"+args['experiment_filename'].split(".")[0]+"/{0}/iu.txt".format(time_stamp),'w')    
    edge_size = args['segment_size']
    threshold = 0.5 * np.ones((1, edge_size, edge_size, 1))
    best_beat=False
    rw = args['residual_weight']
    lr_0=args['lr']
    lr=lr_0
    epochs_waited=0
    lr_decay=args['lr_decay']
    weight = args['weight']
    badcount = 0
    epoch = 0
    best = 0
    model_string=model_string="../results/"+args['experiment_filename'].split(".")[0]+"/{0}/model_{0}.ckpt".format(time_stamp)
    save_path = saver.save(sess, model_string)
    # the training loop
    # terminates if the number of epochs since an improvement equals a given giveup dumber specified as an argument to the program
    #    or the total number of epochs trained equals a given number of total epochs specified as an argument to the program
    while (badcount+args['patience']*epochs_waited) < (args['patience']*args['giveup']) and epoch < args['epochs']:
        epoch += 1
#        if epoch == 2:
#            first_saver = tf.train.Saver()
#            first_saver.save(sess, '../results/Early_checkpoint.ckpt')
        train_modded_outputs, train_modded_labels = run_train_loop(sess, x_train, y_train, targets, dim, lr, weight, time_stamp, args, eval_train_recent=args['eval_train_recent'])
        if args['eval_train_all']:
           train_outputs, train_labels_eval, train_ims = run_eval_loop(sess,x_train, y_train, targets, edge_size, lr, dim, time_stamp, args)
           train_stats = ds.calc_stats(train_outputs, train_labels_eval, targets, 1024*1024)


        val_outputs, val_labels, val_ims = run_eval_loop(sess, x_val, y_val, targets, edge_size, lr, dim, time_stamp, args)
        val_iu, val_std_iu = ds.IU(val_outputs, val_labels, targets)
        iu_outfile.write('{0}\t{1}'.format(epoch,'\t'.join(['{0} {1}'.format(key, value) for key, value in val_iu.items()])))
        if args['eval_train_all']:
           iu_outfile.write('\t{0}'.format('\t'.join(['train_{0} {1}'.format(key, value) for key, value in train_stats['iu_scores'].items()])))
        if args['eval_train_recent']:
           iu_train_recent, iu_train_std_recent = ds.IU(train_modded_outputs, train_modded_labels, targets)
           iu_outfile.write('\t{0}'.format('\t'.join(['recent_train_{0} {1}'.format(key, value) for key, value in iu_train_recent.items()])))
        iu_outfile.write('\n')
        master_iu = np.mean([val_iu[t] for t in targets])


        if master_iu > best:
           best_beat=True
#           model_string="../results/"+args['experiment_filename'].split(".")[0]+"/{0}/model_{0}.ckpt".format(time_stamp)
           save_path = saver.save(sess, model_string)
           print('save path: {0}'.format(save_path))
           badcount = 0
           lr=lr_0
           epochs_waited=0
           best = master_iu
        else:
            badcount += 1
            if(badcount>=args['patience']):
                print('\n\nRESTORING TO CHECKPOINT\n\n')
                epochs_waited+=1
                badcount=0
                lr*=args['lr_decay']
                weight = weight*float(args['weight_decay'])
                saver.restore(sess, save_path)
        if args['verbose']:
            print("NEW IU: %s\tBEST IU: %s"%(master_iu, best))
    saver.restore(sess, save_path)
    test_outputs, test_labels, test_ims = run_eval_loop(sess, x_test, y_test, targets, edge_size, lr, dim, time_stamp, args)
    new_val_output, new_val_labels, new_val_ims = run_eval_loop(sess, x_val, y_val, targets, edge_size, lr, dim, time_stamp, args)
    test_stats = ds.calc_stats(test_outputs, test_labels, targets, 1024*512, should_plot = False)
    test_im_d = {}
    for t in targets:
      test_im_d[t] = test_ims
      tempim = np.zeros((test_ims.shape[1], test_ims.shape[0]*test_ims.shape[2]))
      tempout = np.zeros((test_ims.shape[1], test_outputs[t].shape[0]*test_outputs[t].shape[2]))
    test_label_arr = dl.desegment_output(test_labels, targets, (512, 1024))
    test_out_arr = dl.desegment_output(test_outputs, targets, (512,1024))
    test_in_arr = dl.desegment_output(test_im_d, targets, (512, 1024))
    f_stats = ds.calc_stats(new_val_output, new_val_labels, targets, 1024*512)
    for t in targets:
        for i in range(len(test_out_arr[t])):
            np.savetxt('%s/csv_label_%s_%d.csv'%(output_dirname, t, i), test_label_arr[t][i])
            np.savetxt('%s/csv_output_%s_%d.csv'%(output_dirname, t, i), test_out_arr[t][i])
    if args['verbose']:
        print('TEST: auc {0} fscore {1} iu {2}'.format(np.mean([test_stats['auc'][t] for t in targets]), np.mean([test_stats['fscores'][t] for t in targets]), np.mean([test_stats['iu_scores'][t] for t in targets])))

    print('returning from run')
    return(f_stats, test_stats)


# builds graph and initializes it
# args are a dictionary not a namespace
#def execute(args, train_unlabeled, train_labeled, dev_unlabeled, dev_labeled, test_unlabeled, test_labeled, train_sets, organized_train_unlabeled, organized_train_labeled, layers, objectives, targets)
#def execute(args, layers, objectives, targets, dim, train_unlabeled, train_labeled, dev_unlabeled, dev_labeled, test_unlabeled, test_labeled):
def execute(args, layers, objectives, targets, train_unlabeled, train_labeled, val_unlabeled, val_labeled, test_unlabeled, test_labeled):
    edge_size = args['segment_size']
    dim = len(args['image_types'])
    time_stamp = time.strftime("%d%b_%H.%M.%S")
    sess, shortcut_names = init_graph(dim, edge_size, layers, objectives, targets, args)
    saver = init_saver(sess, args)
    val_stats, test_stats = run_graph(train_labeled, train_unlabeled, val_labeled, val_unlabeled, test_labeled, test_unlabeled, dim, args, sess, saver, time_stamp, targets)

    if args['verbose']:
        print(time_stamp)
    dl.write_outlog(args, time_stamp, targets, test_stats, val_stats)
    sess.close()
    tf.reset_default_graph()
    print('executed')
    return(time_stamp, val_stats, test_stats)
