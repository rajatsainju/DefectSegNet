#calc_precision_recall_stats
#Script to calculate the precision, recall, f1, f2, f0.5, auc, and accuracy and add it to a log file
#Requires an existing log file and a path to the directory containg the raw output images
#creates a new log file. It might be helpfel to overwrite the old one with the new one to save space and avoid confusion, but this is not done automatically for safety.
#Author: Graham Roberts
#Current as of: 2 July 2018

import numpy as np
from PIL import Image
import argparse
from sklearn import metrics
from matplotlib import pyplot as plt

def parse_all_args():
   parser=argparse.ArgumentParser()
   parser.add_argument('log_file', help = 'the filename of the logs')
   parser.add_argument('results_dir', help = 'the directory containing the source images')
   parser.add_argument('target', help = 'the target type')
   parser.add_argument('output_filename', help = 'the file name to save the output to')
   return(parser.parse_args())

#intersection_over_union
def calc_confusion_matrix(invec, labvec, threshold):
   pvec = np.greater(invec, threshold*np.ones(invec.shape))
   tp = 0
   fp = 0
   fn = 0
   tn = 0
   for i in range(len(invec)):
      p = pvec[i]
      l = labvec[i]
      if p and l:
         tp += 1
      elif p and not l:
         fp += 1
      elif not p and l:
         fn += 1
      else:
         tn += 1
   return(tp, fp, fn, tn)

def calc_confusion_arrays(invec, labvec, thresholds):
   tp_vec = np.zeros(thresholds.shape)
   fp_vec = np.zeros(thresholds.shape)
   fn_vec = np.zeros(thresholds.shape)
   tn_vec = np.zeros(thresholds.shape)
   for i in range(len(thresholds)):
      tp, fp, fn, tn = calc_confusion_matrix(invec, labvec, thresholds[i])
      tp_vec[i] = tp
      fp_vec[i] = fp
      fn_vec[i] = fn
      tn_vec[i] = tn
   return(tp_vec, fp_vec, fn_vec, tn_vec)

#find_f_stats
#finds the f_score at a given beta
#takes as arguments the precision, recall, and thresholds from sklearn.metrics.precision_recall_curve
#fourth argument is the beta at which to evaluate
#returns the maximum f-score
def find_f(precision, recall, thresholds, beta):
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

def find_iu(imvec, labvec, threshold):
   tp, fp, fn, tn = calc_confusion_matrix(imvec, labvec, threshold)
   iu_vec = tp/(tp+fn+fp)
   return(iu_vec)

def main():
   args = parse_all_args()
   lfn = args.log_file
   rdir = args.results_dir
   target = args.target
   ofn = args.output_filename
   infile = open(lfn, 'r')
   outfile = open(ofn, 'w')
   lines = infile.readlines()
   for line in lines:
      params = {}
      for p in line.split('|'):
         pp = p.split(':')
         params[pp[0]] = pp[1].strip()
      ts = params['time_stamp']
      inim_1 = Image.open('{0}/{1}/output_set17_{2}.png'.format(rdir, ts, target))
      inim_2 = Image.open('{0}/{1}/output_set25_{2}.png'.format(rdir, ts, target))
#      inim = Image.open('{0}/{1}/output_set17_{2}.png'.format(rdir, ts, target))
      labim_1 = Image.open('{0}/{1}/labels_set17_{2}.png'.format(rdir, ts, target))
      labim_2 = Image.open('{0}/{1}/labels_set25_{2}.png'.format(rdir, ts, target))
#      labim = Image.open('{0}/{1}/labels_set17_{2}.png'.format(rdir, ts, target))
      inarr_1 = np.array(inim_1)/255.
      inarr_2 = np.array(inim_2)/255.
#      inarr = np.array(inim)/255.
      labarr_1 = np.array(labim_1)/255.
      labarr_2 = np.array(labim_2)/255.
#      labarr = np.array(labim)/255.
      arrsize = np.size(inarr_1)
      invec_1 = np.reshape(inarr_1, arrsize)
      invec_2 = np.reshape(inarr_2, arrsize)
#      invec = np.reshape(inarr, arrsize)
      labvec_1 = np.reshape(labarr_1, arrsize)
      labvec_2 = np.reshape(labarr_2, arrsize)
#      labvec = np.reshape(labarr, arrsize)
      invec = np.concatenate((invec_1, invec_2), axis = 0)
      print('invec_1 shape {0} invec shape {1}'.format(invec_1.shape, invec.shape))
      imarr = np.reshape(invec, (int(np.sqrt(invec.shape[0])),int(np.sqrt(invec.shape[0]))))
      plt.imshow(imarr, cmap='hot')
      plt.show()
#      invec = np.concatenate((invec, invec), axis = 0)
#      labvec = np.concatenate((labvec, labvec), axis = 0)
      labvec = np.concatenate((labvec_1, labvec_2), axis = 0)
      np.savetxt('whatshappening_voids_invec.csv',invec)
      np.savetxt('whatshappeneinng_voids_calc_labvec.csv',labvec)
      temp_precision, temp_recall, thresholds = metrics.precision_recall_curve(labvec, invec)
      tp, fp, fn, tn = calc_confusion_arrays(invec, labvec, thresholds)
      print('precision first {0} last {1} recall first {2} last {3}'.format(temp_precision[0], temp_precision[-1], temp_recall[0], temp_recall[-1]))
      precision = temp_precision[1:]
      recall = temp_recall[1:]
      iu_vec = tp/(tp+fn+fp)
      print(np.max(iu_vec))
      acc_vec = (tp+tn)/(tp+fp+fn+tn)
      iu_argmax = np.nanargmax(iu_vec)
      iu = iu_vec[iu_argmax]
      iu_thresh = thresholds[iu_argmax]
      iu_acc = acc_vec[iu_argmax]
      p_iu = precision[iu_argmax]
      r_iu = recall[iu_argmax]
      aupr = metrics.average_precision_score(labvec, invec)
      f1, t1, p1, r1 = find_f(precision, recall, thresholds, 1)
      maybe_tp, maybe_fp, maybe_fn, maybe_tn = calc_confusion_matrix(invec, labvec, t1)
      found_iu = find_iu(invec, labvec, t1)
      print('maybe iu {0} found iu {1} iu_vec {2}'.format(maybe_tp/(maybe_tp+maybe_fn+maybe_fp), found_iu, iu))
      f2, t2, p2, r2 = find_f(precision, recall, thresholds, 2)
      f5, t5, p5, r5 = find_f(precision, recall, thresholds, 0.5)
      predarray = np.greater(invec, t1*np.ones(2*arrsize))
      #iu = metrics.jaccard_similarity_score(labvec, predarray)
      acc = np.sum(np.equal(predarray, labvec).astype(int))/(2*arrsize)
      params['{0}_f_one'.format(target)] = f1
      params['{0}_f_precision'.format(target)] = p1
      params['{0}_f_recall'.format(target)] = r1
      params['{0}_f_threshold'.format(target)] = t1
      params['{0}_f_accuracy'.format(target)] = acc
      params['{0}_IU'.format(target)] = iu
      params['{0}_IU_threshold'.format(target)] = iu_thresh
      params['{0}_IU_precision'.format(target)] = p_iu
      params['{0}_IU_recall'.format(target)] = r_iu
      params['{0}_IU_accuracy'.format(target)] = iu_acc
      params['{0}_aupr'.format(target)] = aupr
      #outfile.write('|'.join([':'.join(d.items()) for d in params.items()]))
      outfile.write('{0}\n'.format('|'.join(['{0}:{1}'.format(d[0],d[1]) for d in params.items()])))

#main()
