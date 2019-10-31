#cnn_datastruct
#helpful dataa structures and statistics functions for the cnn_mlrad project
# Author: Graham Roberts et al.
#Updated: 04 July 2019
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
import calc_precision_recall_stats as cprs
from matplotlib import pyplot as plt

#init_target_dict
#creates a dictionary over targets
#ARGS:
#  targets: a list of targets for 
#RETURNS:
#  a dictionary dict[tagets]
def init_target_dict(targets):
   return_dict = {}
   for t in targets:
      return_dict[t] = []
   return(return_dict)


# build_arch_dict
# function do take a file specifying the architecture of the model and return a dictionary of parameters
# args:
#  fn: the filename of an architechture file specified as an argument to the program
# return:
#  model: a list of dictionaries, each dict contains the parameters for a leyer mapped to by their argument
def build_arch_dict(fn):
    model = []
    arch = open(fn, 'r')
    for l in arch.readlines():
        params = {}
        specs = l.rstrip().split('|')
        for s in specs:
            p = s.split(':')
            params[p[0]] = p[1]
        model += [params]
    return model

# build objectives
# function parses the file of target file ie targets_list.txt
# finds which targets are desired and which tensors are to be used for classification
# return a list of targets and a dictionary mapping them to layers in the graph
def build_objectives(targets_list):
   objs = {}
   targets = []
   for t in targets_list:
      splits = t.split(':')
      target = splits[0]
      objs[target]=int(splits[1])
      targets.append(target)
   return(objs,  targets)

#single_array_from_all_outputs
# after all minibatches this turns existing from store_new_output into a single matrix to dislpay
# args:
#  stored: a list of minbatch matrices
# returns:
#  a matrix of all minibatches
def single_array_from_all_outputs(stored, square=False):
#    """Generate the square image from the minibatched CE"""
    if not square:
       num_columns = int(math.sqrt(2*len(stored)))
       num_rows = num_columns/2
    else:
       num_columns = math.sqrt(len(stored))
       num_rows = num_columns
    num_rows = int(num_rows)
    num_columns = int(num_columns)
    each_row = []
    for i in range(num_rows):
        row=np.concatenate([np.squeeze(stored[j]) for j in range(i*num_columns,(i + 1)*num_columns)],axis=1)
        each_row.append(row)
    return np.concatenate(each_row, axis=0)


#format_output_dictionary
#ARGS: 
#  input_dictionary: a dictionary over sets containing dictionaries over targets containg list of output matrices
#  targets: The list of target string names
#  sets: the list of image sets targetted by dev and test subset of {1,9,17,25}
#
#FUNCTION:
#  formats each list into large single matrix
#
#RETURNS:
#  output_dictionary: a dictioary over sets of dictionaries over targets pointing to output matrix
def format_output_dictionary(input_dictionary, targets, sets):
   output_dictionary = init_target_set_dict(targets, sets)
   for set_num in sets:
      for t in targets:
          output_dictionary[set_num][t] = single_array_from_all_outputs(input_dictionary[set_num][t])
   return(output_dictionary)

def single_vector_from_all_outputs(arr):
   single_tensor = np.concatenate(arr, axis=0)
   single_vector = np.resize(single_tensor, np.size(single_tensor))
   return(single_vector)

def image_to_tensor(inim, edge_size):
   print(inim.shape)
   hlim = int(inim.shape[1]/edge_size)
   vlim = int(inim.shape[2]/edge_size)
   num = (hlim*vlim*inim.shape[0])
   outtens = np.zeros((num,edge_size,edge_size,1))
   for i in range(num):
      zz = int(i/(hlim*vlim))
      ii = i-(zz*hlim*vlim)
      xx = int(ii%hlim)
      yy = int((ii/hlim)%vlim)
      tempim= inim[zz,int(xx*edge_size):int((xx+1)*edge_size),int((yy)*edge_size):int((yy+1)*edge_size)]
      outtens[i,:,:,:] = tempim
   return(outtens)

def IU(outputs, labels, targets):
   iu_scores = {}
   std_iu_scores = {}
   for t in targets:
      max_maybe_iu = 0
      thresholds = np.linspace(np.min(outputs[t]),np.max(outputs[t]),100)
      for tt in thresholds:
         iu = []
         for i in range(outputs[t].shape[0]):
            pred = np.greater(outputs[t][i,:,:],tt)[:,:]
            inter = np.sum(np.logical_and(pred, labels[t][i,:,:]))
            union = np.sum(np.logical_or(pred, labels[t][i,:,:]))
            iu += [inter/union]
         maybe_iu = np.mean(iu)
         maybe_std_iu = np.std(iu)
         if maybe_iu > max_maybe_iu:
            max_maybe_iu = maybe_iu
            max_std_iu = maybe_std_iu
            best_thresh = tt
      iu_scores[t] = max_maybe_iu
      std_iu_scores[t] = maybe_std_iu 
   return(iu_scores, std_iu_scores)

def calc_stats(outputs, labels, targets, siz, should_plot = False):
   should_plot = False
   stats = {}
   stats['auc'] = init_target_dict(targets)
   stats['std_auc'] = init_target_dict(targets)
   stats['acc'] = init_target_dict(targets)
   stats['std_acc'] = init_target_dict(targets)
   stats['recall'] = init_target_dict(targets)
   stats['std_recall'] = init_target_dict(targets)
   stats['precision'] = init_target_dict(targets)
   stats['std_recall'] = init_target_dict(targets)
   stats['fscores'] = init_target_dict(targets)
   stats['std_fscores'] = init_target_dict(targets)
   stats['iu_scores'] = init_target_dict(targets)
   stats['std_iu_scores'] = init_target_dict(targets)
   target_index = 0
   for t in targets:
      sidesize = outputs[t].shape[1]
      index = 0
      set_fscores = []
      set_precision = []
      set_recall =[]
      set_iu_scores = []
      set_acc = []
      set_auc = []
#      for set_num in range(len(labels[t])):
      outvec = single_vector_from_all_outputs(outputs[t])
      labvec = single_vector_from_all_outputs(labels[t])
      numim = int(outvec.shape[0]/siz)
      maybe_thresholds = np.linspace(np.min(outputs[t]),np.max(outputs[t]),1000)
      max_maybe_iu = 0
      max_std_iu = 0
      outout = np.zeros((sidesize, outputs[t].shape[0]*sidesize))
      outpred = np.zeros((sidesize, outputs[t].shape[0]*sidesize))
      outconf = np.zeros((sidesize, outputs[t].shape[0]*sidesize, 3))
      cyan = np.array([0,255,255])
      yellow = np.array([255, 255, 0])
      red = np.array([255, 0, 0])
      best_thresh = 0
#      for i in range(outputs[t].shape[0]):
#          testarr = np.concatenate((outputs[t][i,:,:], labels[t][i,:,:]), axis=1)
#          np.savetxt('testarr_%s_%d.csv'%(t,i), testarr[:,:,0])
      for tt in maybe_thresholds:
         iu = []
         for i in range(outputs[t].shape[0]):
            pred = np.greater(outputs[t][i,:,:],tt)[:,:]
            inter = np.sum(np.logical_and(pred, labels[t][i,:,:]))
            union = np.sum(np.logical_or(pred, labels[t][i,:,:]))
            iu += [inter/union]
         maybe_iu = np.mean(iu)
         maybe_std_iu = np.std(iu)
         if maybe_iu > max_maybe_iu:
            max_maybe_iu = maybe_iu
            max_std_iu = maybe_std_iu
            best_thresh = tt
      if(should_plot):
         for i in range(outputs[t].shape[0]):
            confsquare = np.zeros((sidesize, sidesize, 3))
            pred = np.greater(outputs[t][i,:,:],best_thresh)[:,:,0]
            confsquare[np.where(np.logical_and(pred, labels[t][i,:,:,0]))] = cyan
            confsquare[np.where(np.logical_and(pred, np.logical_not(labels[t][i,:,:,0])))] = red
            confsquare[np.where(np.logical_and(np.logical_not(pred), labels[t][i,:,:,0]))] = yellow
            outconf[:,i*sidesize:(i+1)*sidesize,:] = confsquare[:,:,:].copy()
            outpred[:,i*sidesize:(i+1)*sidesize] = pred
            outout[:,i*sidesize:(i+1)*sidesize] = outputs[t][i,:,:,0]
         plt.subplot(311)
         plt.imshow(outout)
         plt.subplot(312)
         plt.imshow(outpred)
         plt.subplot(313)
         plt.imshow(outconf)
         plt.show()
      stats['iu_scores'][t] = max_maybe_iu
      stats['std_iu_scores'][t] = max_std_iu
      for i in range(numim):
         sub_outvec = outvec[i*siz:(i+1)*siz]
         sub_labvec = labvec[i*siz:(i+1)*siz]
         sub_precision, sub_recall, thresholds = metrics.precision_recall_curve(sub_labvec, sub_outvec)
         temp_f, temp_thresh, temp_precision, temp_recall = find_f(sub_precision, sub_recall, thresholds, 1)
         set_fscores += [temp_f]
         set_precision += [temp_precision]
         set_recall += [temp_recall]
         
         set_acc += [find_accuracy(sub_outvec, sub_labvec, temp_thresh)]
         set_auc += [metrics.roc_auc_score(sub_labvec, sub_outvec)]
      stats['auc'][t] = np.mean(set_auc)
      stats['std_auc'][t] = np.std(set_auc)
      stats['precision'][t] = np.mean(set_precision)
      stats['std_precision'] = np.std(set_precision)
      stats['recall'][t] = np.mean(set_recall)
      stats['std_recall'][t] = np.std(set_recall)
      stats['fscores'][t] = np.mean(set_fscores)
      stats['std_fscores'][t] = np.mean(set_fscores)
      stats['acc'][t] = np.mean(set_acc)
      stats['std_acc'][t] = np.std(set_acc)
   return(stats)

#find_accuracy
#ARGS:
#  predicted: a centor of probabilities [0,1]
#  true: A vector of ground truth labeled points
#  threshold: the prediction threshold
#
#FUNCTION:
#  percentage of points with probaility grater than threshold that agreee with labels
def find_accuracy(prediction, labels, threshold):
   prediction = np.greater(prediction, threshold)
   correct = np.equal(prediction, labels)
   return(np.sum(correct)/np.size(correct))

#finds the f_score at a given beta
#takes as arguments the precision, recall, and thresholds from sklearn.metrics.precision_recall_curve
#fourth argument is the beta at which to evaluate
#returns the maximum f-score
def find_f(precision, recall, thresholds, beta):
   with np.errstate(divide='ignore', invalid = 'ignore'):
      numerator = (1+float(beta)**2)*precision*recall
      denominator = float(beta)**2*precision+recall
      fscores = numerator/denominator
      indmax = np.nanargmax(fscores)
      threshmax = thresholds[indmax]
      if threshmax == 1.0:
         indmax = indmax-1
      tmax = thresholds[indmax]
      fmax = fscores[indmax]
      pmax = precision[indmax]
      rmax = recall[indmax]
   return(fmax, tmax, pmax, rmax)

