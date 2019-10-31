import numpy as np
import cv2
import imutils
import time
from matplotlib import pyplot as plt

def load_im(fn):
   return(cv2.cvtColor(cv2.imread(fn,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB))

def rot_border(angle, inxy):
   xmax = np.max(inxy[:,0])
   xmin = np.min(inxy[:,0])
   ymax = np.max(inxy[:,1])
   ymin = np.min(inxy[:,1])
   xmid = (xmin+xmax)/2
   ymid = (ymin+ymax)/2
   xy = inxy - np.array(xmid, ymid)
   sin = np.sin(np.deg2rad(angle))
   cos = np.cos(np.deg2rad(angle))
   xtrans = np.array([cos, -sin])
   ytrans = np.array([sin, cos])
   xx = np.sum((xy*xtrans),axis=1)
   yy = np.sum((xy*ytrans),axis=1)
   xx = np.round(xx + (np.max(xx)-np.min(xx))/2.).astype(int)
   yy = np.round(yy + (np.max(xx)-np.min(xx))/2.).astype(int)
#   xx = xx * np.greater(xx, xmin)
#   yy = yy * np.greater(yy, ymin)
#   xx = xx * np.less(xx, xmax) + xmax * np.greater(xx, xmax)
#   yy = yy * np.less(yy, ymax) + ymax * np.greater(yy, ymax)
   return(xx, yy)

def rand_sec(inangle, maxx, maxy, minx = 0, miny = 0, mb = 128, samplenum = 200):
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
   rangex = np.arange(xmax-mb)
   pushedx = rangex+mb
   #These next line create the top-near bounds
   neartop = (yc[3]-rangex/tan)*(rangex<=xc[0])
   neartop += ((rangex-xc[0])*tan)*(rangex>xc[0])
   neartop = np.round(neartop).astype(int)
   #These next line create the top-far bounds
   fartop = (yc[3]-pushedx/tan)*(pushedx<=xc[0])
   fartop += ((pushedx-xc[0])*tan)*(pushedx>xc[0])
   fartop = np.round(fartop).astype(int)
   #The calculate the bottom border
   nearbot = (yc[3]+rangex*tan)*(rangex <= xc[2])
   nearbot += (yc[2] - (rangex-xc[2])/tan) * ( rangex > xc[2])
   nearbot = np.round(nearbot).astype(int)
   #The calculate the bottom border
   farbot = (yc[3]+pushedx*tan)*(pushedx <= xc[2])
   farbot += (yc[2] - (pushedx-xc[2])/tan) * ( pushedx > xc[2])
   farbot = np.round(farbot).astype(int)

   topbound = np.maximum(neartop, fartop)
   botbound = np.minimum(nearbot, farbot)-mb
   #This calculates the indeces that dont violate the vertical spacing
   print(botbound-topbound)
   xindallowed = (botbound-topbound)> 0
   xallowed = rangex[xindallowed]
   topallowed = topbound[xindallowed]
   botallowed = botbound[xindallowed]
   randind = np.random.randint(0, high = len(xallowed), size=samplenum)
   sx = xallowed[randind].astype(int)
   stop = topallowed[randind]
   sbot = botallowed[randind]
   sy = np.zeros(samplenum).astype(int)
   for i in range(samplenum):
      sy[i] = np.random.randint(stop[i], high=sbot[i])
   return(sx, sy)

def plot_randsection(plot, sx, sy, mb=128):
   for i in range(len(sx)):
      xx = sx[i]
      yy = sy[i]
      xa = np.array([xx, xx+mb, xx+mb, xx, xx])
      ya = np.array([yy, yy, yy+mb, yy+mb, yy]) 
      plot.plot(xa, ya)
   return

def full_process(im, mb = 128, angle = 45, samplenum = 200):
   tbegin = time.time()
   rotim = imutils.rotate_bound(im, angle)
   sx, sy = rand_sec(angle, im.shape[0], im.shape[1], samplenum = samplenum, mb = mb)
   samples = np.zeros((samplenum, mb, mb, 3))
   for i in range(samplenum):
      xx = sx[i]
      yy = sy[i]
      samples[i] = rotim[xx:xx+mb,yy:yy+mb]
   tend = time.time()
   return(samples.astype(int), sx, sy, rotim, tend-tbegin)
       

def adjust_contrast(im, value):
    factor = (259 * (255 + value))/(255 * (259 - value))
    im = (factor*(im-128)+128).astype(np.uint8)
    return(im)

def add_normal_noise(im, scale):
    noisearr = np.random.normal(loc=0, scale=scale, size=im.shape).astype(np.uint8)
    im+=noisearr
    return

def flip_some(imvec, chance):
    ax0_select = np.less(np.random.random(len(imvec)), chance).astype(np.uint8)
    print(ax0_select)
    output_vec = imvec.copy()
    rotated_arr = np.flip(imvec.copy(), axis=1)
    for i in range(imvec.shape[0]):
        rotated_arr[i,:,:] *= ax0_select[i]
        output_vec[i,:,:] *= (1-ax0_select[i])
    output_vec += rotated_arr
    return(output_vec)

def drop_some(imvec, chance):
    immean = np.mean(imvec)
    imstdev = np.std(imvec)
    immax = np.max(imvec)
    immin = np.min(imvec)
    chances = np.random.random(imvec.shape)
    new_pix = np.random.normal(size = imvec.shape, loc = immean, scale = imstdev)*np.less(chances, chance).astype(int)
    old_pix = imvec.copy()*np.greater(chances, chance)
    old_pix += new_pix
    old_pix = np.clip(old_pix, immin, immax)
    return(old_pix)

"""contrast noise_scale flip_chance drop_chance"""
