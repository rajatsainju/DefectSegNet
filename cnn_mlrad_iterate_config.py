#cnn mlrad iterate configurations
#Author Graham Roberts
#Current: 04 July 2014
import numpy as np
import cnn_mlrad_body as cmb
import cnn_datastructs as ds
import cnn_dataloader as dl
import argparse
import time

def parse_all_args():
#    Parses arguments
#
#    :return: the parsed arguments object
#    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",
                        help="The targets to identify and their labels, separated by a colon. space separated strings are accepted for multiclassification tasks.  e.g. disline:20",
                        nargs = '+',
                        required = True)

    parser.add_argument("--architecture_file",
                        help="The filename of the architecture model",
                        required = True)

    parser.add_argument("--path_to_directory",
                        help="The path to the directory containing data, should contain all images of one region at one dosage. Absolute path most likely not to cause issues, but relative path 'should' work",
                        required = True)

    parser.add_argument('--config_file',
                        help = 'the filename of saved configurations',
                        required = True)

    parser.add_argument("--image_types",
                        help='a string containing vpd or any combination of the three, at least one must be present',
                        type = str,
                        nargs = '+',
                        default=['v','p','d'])

    parser.add_argument("--lr",
                        type=float,
                        help="The learning rate (a float) [default = 0.001]",
                        default=0.001)

    parser.add_argument("--patience",
                        type=int,
                        help="The number of non-improving epochs to run",
                        default=10)

    parser.add_argument("--segment_size", type=int,
                        help="The miniimage edge size (an int) [default = 512]",
                        default=512)

    parser.add_argument("--epochs",
                        type=int,
                        help="The number of epochs to train for [default = 5]",
                        default=5)

    parser.add_argument('--giveup',
                        type=int,
                        help="Quit if we haven't exceeded best performance in giveup*patience epochs (int) [default=3]",
                        default=3)
 
    parser.add_argument('--lr_decay',
                        type=float,
                        help='The lr decrease factor (float) [default=0.8]',
                        default=0.8)

    parser.add_argument("--random_count",
                        help='The number of minibatches to be included in random sample, if absent all minibatches will be trained on once',
                        type=int,
                        default=0)

    parser.add_argument("--train_doses",
                        help = "The doses of images to train on element of {hi,lo, pristine}",
                        default = ['hi', 'lo'],
                        type = str,
                        nargs = '+')

    parser.add_argument("--train_images",
                        help = "the images to train on. Elements of {1, 2, 3}",
                        default = ['1', '2'],
                        type = str,
                        nargs = '+')

    parser.add_argument("--val_doses",
                        help = "The doses to validate on. Elements of {hi, lo, pristine}. Defaults to the same as train doses.",
                        default = None,
                        type = str,
                        nargs = '+')

    parser.add_argument("--val_images",
                        help = "The images to validate on. Elements of {1, 2, 3}. defaults to the same as train",
                        default = None,
                        type = str, 
                        nargs = '+')

    parser.add_argument("--test_doses",
                        help = "The doses to test against. Elements of {hi, lo, pristine. Defaults to hi",
                        default = ['hi'],
                        type = str,
                        nargs = '+')

    parser.add_argument("--test_images",
                        help = "The images to test on. Elements of {1, 2, 3}. Defaults to [1,2].",
                        default = ['1', '2'],
                        type = str,
                        nargs = '+')

    parser.add_argument('--residual_weight',
                        type=float,
                        help='The numerical factor by which the input is added to the last layer',
                        default=0.1)

    parser.add_argument('--train_num',
                        help = 'the number of training batches per epoch',
                        type = int,
                        default = 20)

    parser.add_argument('--batch_norm',
                        help = 'boolean whether ot not to batch normalize',
                        type = bool,
                        default = False)

    parser.add_argument('--weight',
                        help = 'the weight to assign positive classifications',
                        type = float,
                        default = 8.0)

    parser.add_argument('--mb_depth',
                        help = 'the number of minibatch images to pass through the nn stacked at a time',
                        type = int,
                        default=32)

    parser.add_argument('--experiment_filename',
                        help='the name of the logfile',
                        default = 'results_log.txt')

    parser.add_argument('--reg_scale',
                        help='the scale for L2 regularization',
                        default=0.1,
                        type=float)

    parser.add_argument('--last_reg_scale', 
                        help='the scale for L2 on the last layer', 
                        default=0.1, 
                        type=float)

    parser.add_argument('--dropout_rate', 
                        help = 'the probability that a point is dropped during training', 
                        default = 0.2)

    parser.add_argument('--eval_train_all', 
                        help = 'whether or not to evaluate quality on train statistics each epoch', 
                        default=False, 
                        type=bool)

    parser.add_argument('--eval_train_recent', 
                        help = 'whether or not to evaluate on the most recently trained on minibatces of train', 
                        default = False)

    parser.add_argument('--verbose', 
                        help='whether or not to print val scores each iteration', 
                        default = False)

    parser.add_argument('--initial_config', 
                        help='The iteration number in a file to start on', 
                        default=0)

    parser.add_argument('--weight_decay', 
                        help='the less than one value to decrease the weights by', 
                        default=1.0, 
                        type=float)

    parser.add_argument('--weight_kernel', 
                        help = 'whether or not to kernelize the weights', 
                        default = False, 
                        type=bool)

    parser.add_argument('--checkpoint_fn', 
                        help = 'the path to the checkpoint .ckpt file', 
                        default = None)

    parser.add_argument('--drop_shortcut_rate',
                        help = "The rate at which to drop shortcuts. I think this is removed but it's referenced in several places so I'll double check",
                        default = 1)
    parser.add_argument('--contrast',
                        help = "The max contrast value to adjust the training data by",
                        type = float,
                        default = 0)
    parser.add_argument('--noise_scale',
                        help = 'The scale of random gaussian noise',
                        type = float,
                        default = 0)
    parser.add_argument('--flip_chance',
                        help = 'The probability that any one layer will be flipped',
                        type = float,
                        default = 0)
    parser.add_argument('--drop_chance',
                        help = 'The chance any one pixel will e dropped and replaced with noise',
                        type = float,
                        default = 0)
    args = parser.parse_args()
    argd = set_doses(vars(args).copy())
    
    return(argd) 

def set_doses(argd):
    hi_dose_opts = ['hi', 'high', 'h', 'hidose', 'highdose']
    low_dose_opts = ['lo', 'low', 'l', 'lodose', 'lowdose']
    pristine_opts = ['pristine', 'prist', 'p']
    tdoses = []
    for dose in argd['train_doses']:
       if dose.lower() in hi_dose_opts and 'hidose'not in tdoses:
          tdoses += ['hidose']
       elif dose.lower() in low_dose_opts and 'lodose' not in tdoses:
          tdoses += ['lodose']
       elif dose.lower() in pristine_opts and 'pristine' not in tdoses:
          tdoses += ['pristine']
    argd['train_doses'] = tdoses
    if argd['val_doses'] == None:
      argd['val_doses'] = argd['train_doses']
    else:
      vdoses = []
      for dose in argd['val_doses']:
          if dose.lower() in hi_dose_opts and 'hidose'not in vdoses:
             vdoses += ['hidose']
          elif dose.lower() in low_dose_opts and 'lodose' not in vdoses:
             vdoses += ['lodose']
          elif dose.lower() in pristine_opts and 'pristine' not in vdoses:
             vdoses += ['pristine']
      argd['val_doses'] = vdoses
    if argd['val_images'] == None:
      argd['val_images'] = argd['train_images'] 
    tedoses = []
    for dose in argd['test_doses']:
       if dose.lower() in hi_dose_opts and 'hidose'not in tedoses:
          tedoses += ['hidose']
       elif dose.lower() in low_dose_opts and 'lodose' not in tedoses:
          tedoses += ['lodose']
       elif dose.lower() in pristine_opts and 'pristine' not in tedoses:
          tedoses += ['pristine']
    argd['test_doses'] = tedoses
    imtypes = []
    dislocopts = ['disloc', 'dislocations', 'dislocation', 'd']
    precipopts = ['precip', 'precips', 'precipitates', 'precipitate', 'p']
    voidopts = ['void', 'voids', 'v']
    for t in argd['image_types']:
       if t.lower() in dislocopts and 'disloc' not in imtypes:
          imtypes += ['disloc']
       elif t.lower() in precipopts and 'prec' not in imtypes:
       #elif t.lower() in precipopts and 'precip' not in imtypes:
          imtypes += ['prec']
          #imtypes += ['precip']
       elif t.lower() in voidopts and 'void' not in imtypes:
          imtypes += ['void']
    argd['image_types'] = imtypes
    return(argd)

   

      
def update_args(args, l):
  config={}
  specs = l.rstrip().split('|')
  for s in specs:
     p = s.split(':')
     config[p[0]] = p[1]
  keys = [key for key, value in config.items()]
  if 'lr' in keys:
     args['lr'] = float(config['lr'])
  if 'weight' in keys:
     args['weight'] = float(config['weight'])
  if 'reg_scale' in keys:
     args['reg_scale'] = float(config['reg_scale'])
  if 'last_reg_scale' in keys:
     args['last_reg_scale'] = float(config['last_reg_scale'])
  if 'lr_decay' in keys:
     args['lr_decay'] = float(config['lr_decay'])
  elif 'beta' in keys:
      args['lr_decay'] = float(config['beta'])
  if 'dropout_rate' in keys:
     args['dropout_rate'] = float(config['dropout_rate'])
  if 'weight_decay' in keys:
     args['weight_decay'] = float(config['weight_decay'])
  if 'contrast' in keys:
     args['contrast'] = float(config['contrast'])
  if 'noise_scale' in keys:
     args['noise_scale'] = float(config['noise_scale'])
  if 'drop_chance' in keys:
     args['drop_chance'] = float(config['drop_chance'])
  return(args)
   
def main():
   best_iu = 0
   best_line = None
   args_0 = parse_all_args()
   path_to_dir = args_0['path_to_directory']
   layers = ds.build_arch_dict(args_0['architecture_file'])
   objectives, targets = ds.build_objectives(args_0['target'])
   best_auc = 0
   dim = len(args_0['image_types'])
   check_file = None
   big_output = None
   time_stamp = time.strftime("%d%b_%H.%M.%S")
   train_unlabeled, train_labeled, val_unlabeled, val_labeled, test_unlabeled, test_labeled = dl.load_all(args_0, targets)
   outfile = open(args_0['experiment_filename'], 'a')
   config_file = open(args_0['config_file'], 'r')
   lines = config_file.readlines()
   errorlog = open('error_log.log', 'a')
   best_time_stamp = 'none'
   for i in range(int(args_0['initial_config']),len(lines)):
     l = lines[i]
     args = update_args(args_0.copy(),l)
     print('executing')
     time_stamp, val_stats, test_stats = cmb.execute(args, layers, objectives, targets, train_unlabeled, train_labeled, val_unlabeled, val_labeled, test_unlabeled, test_labeled)
     print('executed') 
     print(test_stats['iu_scores'])
     out_string = dl.format_write_statement(args, targets, time_stamp, val_stats, test_stats)
     outfile.write(out_string)
     outfile.flush()
     mean_test_iu = np.mean([test_stats['iu_scores'][t] for t in targets])
     if mean_test_iu > best_iu:
        best_iu = mean_test_iu
        best_time_stamp = time_stamp
        print("NEW BEST IU%s"%(best_iu))
   print("IU: %s\tTIMESTAMP: %s"%(best_iu, best_time_stamp))

if __name__ == '__main__':
   main()
