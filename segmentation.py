import argparse
import numpy as np
import cnn_mlrad_body as cmb
import cnn_dataloader as dl
import cnn_datastructs as ds
import tensorflow as tf
import os

#ISFLOAT
#a boolean test whther or not a value can be interpreted as a floating point number
#used when parsing config
def isFloat(val):
   try:
      float(val)
      return(True)
   except:
      return(False)

#ISINT
#a bolean test of whether or not a string can be interpreted as an integer
#used when parsing configuration
def isInt(val):
   try:
      int(val)
      return(True)
   except:
      return(False)


#UPDATE ARGS FINAL
#fills the args dictionary with options from the config file
#many are not really neccesary since the model isn't training, but they are present in the call signature of useful functions
#They are still present in the file for posterity
#Some options are neccesary for segmentation sizes etc
def update_args_final(args, lines):
   for line in lines:
      vals = line.split(': ')
      if isInt(vals[1]):
         args[vals[0]] = int(vals[1])
      elif isFloat(vals[1]):
         args[vals[0]] = float(vals[1])
      else:
         args[vals[0]] = vals[1]
   return(args)

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--image',
                        help = 'The input image or images in a space separated list to evaluate',
                        required = True,
                        nargs = '+')
   parser.add_argument('--target',
                        help = 'the target to predict on',
                        required = True,
                        type = str)
   parser.add_argument('--output',
                        help = 'The output directory',
                        default = 'results')
   return(vars(parser.parse_args()))

#EVALUATE_IMAGE
#ARGS:
# sess: the tensorflow session object
# image_tensor: The multidimensional tensor of images
# targets: the target types to be predicted, is a list style to support multiclass options although as of yet no satisfactory checkpoints have been saved and it is reccomended for single class ony
# dim: a value related to the dimenionality of the input, i.e., 3 for RGB or a single image. Tests stacking the three input images have not proven encouraging and dim is kept to 1 for all published checkpoints
# args: a dictionary of other examples
#
#Passes the image tesor through the pre-trained convolutional eural network for prediction
#
#RETURNS:
# the output probability maps
def evaluate_image(sess, image_tensor, targets, dim, args):
   not_training = False
   outputs = ds.init_target_dict(targets)
   labels = ds.init_target_dict(targets)
   fetches = ['out_%s:0'%(t) for t in targets]
   new_container = np.zeros((1,image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))
   new_container[0,:,:,:] = image_tensor
   x_d = ds.image_to_tensor(new_container, args['segment_size'])
   fd = {'x:0':x_d, 'lr:0':0, 'is_training:0':True, 'residual_weight:0':args['residual_weight']}
   return_list = sess.run(fetches = fetches, feed_dict = fd)
   for k in range(len(targets)):
      t = targets[k]
      outputs[t] = return_list[k]
   return(outputs)
   
#LOAD PARAMETERS
#PARSE ARGUMENTS FROM COMMAND LINE
#CREATE CONFIG,
#CREATE OJECTIVES
#CREATE TARGETS
#IDENTIFIES CHECKPOINT FILE
def load_parameters():
   args = parse_args()
   savefile = 'best_checkpoint_%s'%(args['target'])
   layers = ds.build_arch_dict('%s/architecture.txt'%(savefile))
   infile = open('%s/targets.txt'%(savefile), 'r')
   targ_list = []
   for line in infile.readlines():
     targ_list += [line.replace(' ',':')]
   objectives, targets = ds.build_objectives(targ_list)
   print(targets)
   dim = 1
   config = open('%s/config.txt'%(savefile), 'r')
   args = update_args_final(args,config.readlines())
   checkfile = os.path.join(savefile, 'checkpoint.ckpt')
   return(args, config, objectives, targets, dim, checkfile)

#INPUT IMAGE
#ARGS:
#  image: a string path to an image to load
#
#Loads the image directed to
#separates it into a tensor and identifies the name with the leading path removed
#
#RETURNS:
#  imname: the string name of the file with the path removed
#  image_tensor: The loaded image separated into small squares and stacked nto a tensor for the neural network
#  imsize: The original size of the image for reconstruction
def input_image(image):
   imname = os.path.split(image)[-1]
   image_tensor = dl.load_unlabeled_image(image)
   imsize = (image_tensor.shape[0], image_tensor.shape[1])
   return(imname, image_tensor, imsize)

#OUTPUT IMAGE
#ARGS:
#  sess: the tensorflow session object
#  image_tensor: The tensor of image segemnts returned by input_image()
#  targets: The list of targets to be evaluated
#  dim: for now 1, but room is left for future updates
#  args: the dictionary of arguemnts, and the hyperparameter configuration trained on used for image segentation
#  imsize: the original size of the image to reconstruct
#
#Evaluates the input image tensor with evaluate_image()
#Reshapes to the original size
#
#RETURNS:
#  output_array: the array of images resied to their original size with values returned by the network
def output_image(sess, image_tensor, targets, dim, args, imsize):
   outputs = evaluate_image(sess, image_tensor, targets, dim, args) 
   output_array = dl.desegment_output(outputs, targets, imsize)
   return(output_array)

#SCALE IMAGE
#Simply takes an image of likelihoods and maps the to [0,1] probability
def scale_image(image):
   scaledim = image.copy()
   scaledim -= np.min(scaledim)
   scaledim /= np.max(scaledim)
   return(scaledim)

#SEGMENTATION.PY
#
#Loads a pre-trained checkpoint and uses it to evaluate previously unseen image(s)
#For now takes a target from ['prec', 'void', 'disloc'] and uses our best results
#Also requires a list of images using their relative path they should e space separated if there are more than one
#Optionally a directory to save results to can be specified
def main():
   args, config, objctives, targets, dim, checkfile = load_parameters()
   FLAGS = tf.app.flags.FLAGS
   tf.app.flags.DEFINE_string('checkpoint_dir', checkfile, 'checkpoint_directory')
   if not os.path.exists(args['output']):
      os.makedirs(args['output'])
   with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph('%s.meta'%(checkfile))
#      new_saver.restore(sess, tf.train.latest_checkpoint('/'.join(checkfile.split('/')[:-1])))
      new_saver.restore(sess, checkfile)
      for image in args['image']:
         imname, image_tensor, imsize = input_image(image)
         output_array = output_image(sess, image_tensor, targets, dim, args, imsize)
         for t in targets:
            for i in range(len(output_array[t])):
               scaledim = scale_image(output_array[t][i])
               np.savetxt(os.path.join(args['output'], '%s_pred.csv'%(imname.strip('.png'))), scaledim)
   

if __name__ == '__main__':
   main()
