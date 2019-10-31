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
import cv2
import imutils
from matplotlib import pyplot  as plt

#custom modules
import cnn_datastructs as ds

#LOAD LABELED DATA
#ARGS:
#  sections a list of sections to load either [1,2,3] for training data, or val or test for val and test respectively
#  doses The doses to load hidose, lodose, or pristine or an appropriate combination
#  ims The image numbers to load, either [1,2,3]
#  targets The targets to load
#  path to dir The path to the data directory i.e., ../renamed_data_png/train_labeled
#RETURNS
#  The labeled data
def load_labeled_data(sections, doses, ims, targets, path_to_dir):
   labeled_data = {}
   for t in targets:
      labeled_data[t] = []
      for sect in sections:
         for dose in doses:
            for im in ims:
               imname = '%s/%s_%s%s_%s_lab.png'%(path_to_dir, sect, dose, im, t)
               try:
                  inim = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
                  labels = np.greater(inim, 0).astype(np.uint8)
                  container = np.zeros((labels.shape[0], labels.shape[1], 1))
                  container[:,:,0] = labels
                  labeled_data[t] += [container]
               except:
                  print("error loading image %s"%(imname))
   return(labeled_data) 

#LOAD UNLABELED DATA
#Loads the unlabeled data
#ARGS:
#  Sections the sections or quandrants to load
#  doses the doses hidose, lodose or pristine to load
#  ims the image to use from 1, 2, or 3
#  sources the source targets disloc, voids, or preips to load
#  path_to_dir the path to the directory containg data
#
#RETURNS
#  A list of unlabeledd data images
def load_unlabeled_data(sections, doses, ims, sources, path_to_dir):
   unlabeled_data_list = []
   for sect in sections:
      for dose in doses:
         for im in ims:
            subsource_list = []
            for s in sources:
               imname = '%s/%s_%s%s_%s.png'%(path_to_dir, sect, dose, im, s)
               try:
                  newim = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
                  container = np.zeros((newim.shape[0], newim.shape[1], 1))
                  container[:,:,0] = newim
                  subsource_list += [container]
               except:
                  print("Error loading image %s"%(imname))
            unlabeled_data_list += [np.dstack(subsource_list)]
   return(unlabeled_data_list)

def load_unlabeled_image(imname):
   unlabeled_data_list = []
   try:
      print(imname)
      newim = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
      container = np.zeros((newim.shape[0], newim.shape[1], 1))
      container[:,:,0] = newim
      return(np.dstack([container]))
   except:
      print('ERROR LOADING IMAGE %s'%(imname))
      return(None)

#LOAD PAIRED DATA
#calls load_unlabeled data and load labelledd data and keeps the paired data organized
#ARGS:
#  Sections the sections or quandrants to load
#  doses the doses hidose, lodose or pristine to load
#  ims the image to use from 1, 2, or 3
#  sources the source targets disloc, voids, or preips to load
#  path_to_dir the path to the directory containg data
#
#RETURNS both the labeled and unlabeld data
def load_paired_data(sections, doses, ims, sources, targets, path_to_dir):
   unlabeled_data = load_unlabeled_data(sections, doses,ims, sources, path_to_dir)
   labeled_data = load_labeled_data(sections, doses, ims, targets, '%s_labels'%(path_to_dir))
   return(unlabeled_data, labeled_data)
   
#LOAD ALL
#calls load paired data for train, test and val
#RETRUNS ALL of the data
def load_all(args, targets):
   unlabeled_train, labeled_train = load_paired_data(['1','2','3'], args['train_doses'], args['train_images'], args['image_types'], targets, "%s/train"%(args['path_to_directory']))
   unlabeled_val, labeled_val = load_paired_data(['val'], args['val_doses'], args['val_images'], args['image_types'], targets, "%s/val"%(args['path_to_directory']))
   unlabeled_test, labeled_test = load_paired_data(['test'], args['test_doses'], args['test_images'], args['image_types'], targets, "%s/test"%(args['path_to_directory']))
   return(unlabeled_train, labeled_train, unlabeled_val, labeled_val, unlabeled_test, labeled_test)

# Desegment output
#  Turns a list of minibatches into images of the original size they came from
#  useful for visualization
#  ARGS:
#     Output a dict of minibatches for each target
#     targets the targets
#     im size the size of the inital image 512*1024 for val and test 1024*1024 for train
#  RETURNS
#     ann array of images
def desegment_output(output, targets, im_size):
   print(im_size)
   subimsize = output[targets[0]].shape
   top_count = int(round(im_size[0]/subimsize[1]))
   side_count = int(round(im_size[1]/subimsize[2]))
 #  images = {}
   imarr = {}
   each_im_size = np.size(output[targets[0]][0])
   out_im_size = im_size[0]*im_size[1]
   images_per = int(out_im_size/each_im_size)
   for t in targets:
    #  images[t] = [np.zeros(im_size) for entry in range(int(len(output[t])/images_per))]
      imarr[t] = [np.zeros(im_size) for entry in range(int(len(output[t])/images_per))]
      for i in range(output[t].shape[0]):
         tempim = np.zeros((subimsize[0], subimsize[1]))
         tempim = output[t][i][:,:,0]
         print('TEMPIM SHAPE')
         print(tempim.shape)
         j = int(i % top_count)
         k = int((i / top_count)% side_count)
         l = int(i/(top_count*side_count))
         imarr[t][l][j*subimsize[1]:(j+1)*subimsize[1],k*subimsize[2]:(k+1)*subimsize[2]] = tempim
   return(imarr)

#  Augment Train
#  selects a small subset of the training set and modifies them to create a larger variety of training data
#  ARGS:
#     unlabeld: the unlabeled data
#     labeled: the labeled data to keep it aligned through rotation and flip
#     targets: the targets
#     im_size: The size of the imagtes input
#     args: The args dictionary passed by the user
#  RETURNS:
#     an minibatch of augmented data to train on
def augment_train(unlabeled, labeled, targets, im_size, args):
   count = args['mb_depth']
   edge_size = args['segment_size']
   with np.errstate(divide='ignore', invalid = 'ignore'):
      dim = 1
      tstart = time.time()
      im_num = len(unlabeled)
      ims = np.random.randint(0, high = im_num, size = count)
      angles = np.random.randint(0, high = 360, size = count)
      ix = np.zeros(count)
      iy = np.zeros(count)
      minibatch_tensor = np.zeros((count, edge_size, edge_size, dim))
      minibatch_label_tensors = {}
      for t in targets:
         minibatch_label_tensors[t]= np.zeros((count, edge_size, edge_size, dim))
      for i in range(count):
         xx, yy = rand_point(angles[i], im_size[0], im_size[1], edge_size = edge_size)
         rotim = imutils.rotate_bound(unlabeled[ims[i]], angles[i])
         minibatch_tensor[i,:,:,:] = np.reshape(rotim[xx:xx+edge_size,yy:yy+edge_size], (1,edge_size, edge_size, dim))
         for t in targets:
            rotlab = imutils.rotate_bound(labeled[t][ims[i]], angles[i])
            rotlab = np.greater(rotlab, np.mean(rotlab))*np.max(rotlab)
            minibatch_label_tensors[t][i,:,:,:] = np.reshape(rotlab[xx:xx+edge_size,yy:yy+edge_size], (1,edge_size, edge_size, dim))
      cleanim = minibatch_tensor[0,:,:,0].copy()
      minibatch_tensor = adjust_contrast(minibatch_tensor, args['contrast'])
      minibatch_tensor = add_normal_noise(minibatch_tensor, args['noise_scale'])
      minibatch_tensor, minibatch_label_tensors = flip_some(minibatch_tensor, minibatch_label_tensors, targets, args['flip_chance'])
      minibatch_tensor = drop_some(minibatch_tensor, args['drop_chance'])
   return(minibatch_tensor, minibatch_label_tensors)

#WRITE OUTLOG
#Wrrites the performance and all of the hyperparameters if a experiment needs to be repeated or analyzed
# ARGS:
#     args: the argument array passed from the command line and updated from a configuration
#     time_stamp: The unique time_stamp of en experiment run
#     test_stats: The performance on the test set measured by a variety of kmetrics
#     val_stats:  The stats on the val data with a variety of metrics
def write_outlog(args, time_stamp, targets, test_stats, val_stats):
   outfile=open("../results/"+args['experiment_filename'].split(".")[0]+"/{0}/writeup.txt".format(time_stamp),'w')    
   outfile.write('test_iu: {0}'.format(np.mean([test_stats['iu_scores'][t] for t in targets])))
   outfile.write('\n'.join('{0} {1}'.format(key, value) for key, value in args.items()))
   outfile.write('\n')
   outconf = open('../results/%s/%s/config.txt'%(args['experiment_filename'], time_stamp), 'w')
   for key, value in args.items():
      print(key)
      print(value)
      outconf.write('%s: %s\n'%(key, str(value)))
   print('TEST IU SCORES')
   print(test_stats['iu_scores'])
   print('VAL IU SCORES')
   print(val_stats['iu_scores'])
   for t in targets:
      outfile.write('%s_test_iu: %d\n'%(t, test_stats['iu_scores'][t]))
      outfile.write('%s_test_f: %d\n'%(t, test_stats['fscores'][t]))
      outfile.write('%s_test_auc: %d\n'%(t, test_stats['auc'][t]))
      outfile.write('%s_test_acc: %d\n'%(t, test_stats['acc'][t]))
      outfile.write('%s_test_precision: %d\n'%(t, test_stats['precision'][t]))
      outfile.write('%s_test_recall: %d\n'%(t, val_stats['precision'][t]))
      outfile.write('%s_iu: %d\n'%(t, val_stats['iu_scores'][t]))
      outfile.write('%s_f: %d\n'%(t, val_stats['fscores'][t]))
      outfile.write('%s_auc: %d\n'%(t, val_stats['auc'][t]))
      outfile.write('%s_acc: %d\n'%(t, val_stats['acc'][t]))
      outfile.write('%s_precision: %d\n'%(t, val_stats['precision'][t]))
      outfile.write('%s_recall: %d\n'%(t, val_stats['precision'][t]))
   outfile.close()
   return
  
#RAND POINT
#calculates an anchor point for random rotation so a minibatch may be taken without exceeding to original boundaries
#ARGS:
#  inangle: the angle of rotation
#  maxx the maximum x value
#  maxy the maximum y value
#  enge_size: the size of the small square images within ewach mimnibatch
#Returns:
#  The x nd y points of the corner to draw a minibatch from
def rand_point(inangle, maxx, maxy, minx = 0, miny = 0, edge_size = 128):
   corns = np.array([[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy]]) - np.array([(minx+maxx)/2.,(miny+maxy)/2.])
   angle = np.deg2rad(inangle % 90)
   cos = np.cos(angle)
   sin = np.sin(angle)
   tan = np.tan(angle)
   xtrans = np.array([cos, -sin])
   ytrans = np.array([sin, cos])
   xc = np.sum(corns*xtrans, axis = 1)
   yc = np.sum(corns*ytrans, axis = 1)
   xc = xc - np.min(xc)
   yc = yc - np.min(yc)
   ymax = np.max(yc)
   xmax = np.max(xc)
   rangex = np.arange(xmax-edge_size)
   pushedx = rangex+edge_size
   #These next line create the top-near bounds
   neartop = np.nan_to_num((yc[3]-rangex/tan)*(rangex<=xc[0]))
   neartop += np.nan_to_num(((rangex-xc[0])*tan)*(rangex>xc[0]))
   neartop = np.round(neartop).astype(int)
   #These next line create the top-far bounds
   fartop = np.nan_to_num((yc[3]-pushedx/tan)*(pushedx<=xc[0]))
   fartop += np.nan_to_num(((pushedx-xc[0])*tan)*(pushedx>xc[0]))
   fartop = np.round(fartop).astype(int)
   #The calculate the bottom border
   nearbot = np.nan_to_num((yc[3]+rangex*tan)*(rangex <= xc[2]))
   nearbot += np.nan_to_num((yc[2] - (rangex-xc[2])/tan) * ( rangex > xc[2]))
   nearbot = np.round(nearbot).astype(int)
   #The calculate the bottom border
   farbot = np.nan_to_num((yc[3]+pushedx*tan)*(pushedx <= xc[2]))
   farbot += np.nan_to_num((yc[2] - (pushedx-xc[2])/tan) * ( pushedx > xc[2]))
   farbot = np.round(farbot).astype(int)

   topbound = np.maximum(neartop, fartop)
   botbound = np.minimum(nearbot, farbot)-edge_size
   #This calculates the indeces that dont violate the vertical spacing
   xindallowed = (botbound-topbound)> 0
   xallowed = rangex[xindallowed]
   topallowed = topbound[xindallowed]
   botallowed = botbound[xindallowed]
   randind = np.random.randint(0, high = len(xallowed))
   sx = int(xallowed[randind])
   stop = topallowed[randind]
   sbot = botallowed[randind]
   sy = int(np.random.randint(stop, high = sbot))
   return(sx, sy)


def adjust_contrast(im, value):
    factor = (259 * (255 + value))/(255 * (259 - value))
    table = np.array([(i/255.0 * factor)*255.0 for i in np.arange(0,256)]).astype(np.uint8)
    im = cv2.LUT(im.astype(np.uint8), table)
    return(im)
  
#Add normal noise
#Adds random noise drawn from a normal distribution
#ARGS:
# the image to add noise to in place
# the scale of the noise
# The image is returned but the object pointed to has been modified
def add_normal_noise(im, scale):
    noisearr = np.random.normal(loc=0, scale=scale, size=im.shape).astype(np.uint8)
    im+=noisearr
    return(im)

#Flip some
#ARGS:
#imvec: the vector of miniature images, some of which will be flipped
#labvecs: The vector of labels to be flipped in the same way to maintain label allignment
#targets: the targets
#chance: The chance each image will be flipped
def flip_some(imvec, labvecs, targets, chance):
    ax0_select = np.less(np.random.random(len(imvec)), chance).astype(np.uint8)
    output_vec = imvec
    outlab_vecs = {}
    rotated_labs = {}
    for t in targets:
        rotated_labs[t] = np.flip(labvecs[t].copy(), axis=1)
        outlab_vecs[t] = labvecs[t]
    rotated_arr = np.flip(imvec.copy(), axis=1)
    for i in range(imvec.shape[0]):
        rotated_arr[i,:,:] *= ax0_select[i]
        output_vec[i,:,:,] *= (1-ax0_select[i])
        for t in targets:
            outlab_vecs[t][i,:,:] *= (1-ax0_select[i])
            rotated_labs[t][i,:,:] *= ax0_select[i]
    output_vec += rotated_arr
    for t in targets:
        outlab_vecs[t] += rotated_labs[t]
    return(output_vec, outlab_vecs)

#drop some 
#replaces some percentage of pixels with background
#ARGS:
#  imvec: The vector of input images
#  chance: The percentage of pixels to be replaced
def drop_some(imvec, chance):
    immean = np.mean(imvec)
    imstdev = np.std(imvec)
    immax = np.max(imvec)
    immin = np.min(imvec)
    chances = np.random.random(imvec.shape)
    new_pix = (np.random.normal(size = imvec.shape, loc = immean, scale = imstdev)*np.less(chances, chance)).astype(np.uint8)
    old_pix = (imvec*np.greater(chances, chance)).astype(np.uint8)
    old_pix += new_pix
    old_pix = np.clip(old_pix, immin, immax)
    return(old_pix)

def mean_dict(data, sets, targets):
   return(np.mean([np.mean([data[set_num][t] for t in targets]) for set_num in sets]))

def std_dict(data, sets, targets, dof):
   return(np.std([np.mean([data[set_num][t] for t in targets]) for set_num in sets]))

def format_write_statement(args, targets, time_stamp, val_stats, test_stats):
    out_string ='time_stamp:%s'%(time_stamp)
    for t in targets:
        out_string += '|%s_test_iu:%f|%s_test_std_iu:%f|%s_val_iu:%f|%s_val_std_iu:%f'%(t, test_stats['iu_scores'][t],t, test_stats['std_iu_scores'][t], t, val_stats['iu_scores'][t], t, val_stats['std_iu_scores'][t])
    out_string += '|lr:%d|weight:%d|lr_decay:%d|reg_scale:%d|last_reg_scale:%d|dropout_rate:%d|weight_decay:%d|drop_shortcut_rate:%d'%(args['lr'], args['weight'], args['lr_decay'], args['reg_scale'], args['last_reg_scale'], args['dropout_rate'],args['weight_decay'],args['drop_shortcut_rate'])
    print(out_string)
    out_string += '\n'
    return(out_string)
