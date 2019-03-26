# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:45:06 2016
aim to be the standard augmentation that supports as many transformation as possible
current supported transformations:
  channel swap, transpose
  mean substract, flip, crop, scale, rotate
@author: tb00083
"""

import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from numpy import matrix as mat
from PIL import Image

class ImageAugment:

  """
  SimpleTransformer is a simple class for preprocessing and deprocessing
  images for caffe.
  """

  def __init__(self, mean=0, scale=1.0, shape = (225,225), rot = None,zoom=None,verbose = True):
    self.mean_org = np.array(mean, dtype=np.float32).squeeze()
    
    self.flip = True
    self.tranpose = True
    self.channel_swap = True
    self.scale = np.float32(scale)
    self.outshape = shape
    self.rot = rot
    self.zoom = zoom
    
    #mean_org should have form CxHxW for color image or HxW for gray img or (3,)
    # for mean pixel or just a scalar
    if self.mean_org.ndim == 0: #scalar
      self.mean = self.mean_org 
    elif self.mean_org.ndim == 1: #mean pixel RGB
      assert len(self.mean_org)==3, 'ImageAugment error: not mean_pixel format'
      if verbose: print('ImageAugment: assume color images')
      self.mean = self.mean_org[:,None,None]
    elif self.mean_org.ndim == 2:  #gray mean img
      self.mean = self.mean_org[None,...]
      if verbose: print('ImageAugment: assume gray images')
      self.channel_swap = False
    else:  #color mean image
      self.mean = self.mean_org
      if verbose: print('ImageAugment: assume color images')
    
    if self.mean_org.ndim > 1: #need to crop mean
      x_offset = (self.mean_org.shape[-1] - self.outshape[-1])/2
      y_offset = (self.mean_org.shape[-2] - self.outshape[-2])/2
      self.mean = self.mean[...,y_offset:y_offset+self.outshape[-2],
               x_offset:x_offset+self.outshape[-1]]
    
    msg = 'tranpose: ' + str(self.tranpose) + '\nchannel swap: ' + str(self.channel_swap)
    if verbose: print(msg)
    
    
  def set_mean(self, mean):
    """
    Set the mean to subtract for centering the data.
    """
    self.mean = mean
      
  def set_shape(self, shape):
    self.outshape = shape

  def set_scale(self, scale):
    """
    Set the data scaling.
    """
    self.scale = scale
  
  def set_rotation(self, rot):
    """
    Set the rotation range (in degree)
    """
    self.rot = rot
    
  def set_zoom(self,zoom):
    """
    set zooming range (float)
    """
    self.zoom = zoom

  def preprocess(self, im):
    """
    HxWxC to CxHxW
    swap RGB to BGR
    mean substract
    scaling
    """
    im = np.float32(im)
    if im.ndim == 2:
      im = im[...,None] #gray scale image
    if self.tranpose:
      im = im.transpose((2,0,1)) #RGB HxWxC to CxHxW
    if self.channel_swap:
      im = im[::-1] # CxHxW RGB to BGR
    #mean substract
    im -= self.mean
    #scale
    im *= self.scale
    
    return im

  def deprocess(self, im):
    """
    inverse of preprocess()
    """
    im /= self.scale
    im += self.mean
    if self.channel_swap:
      im = im[::-1]
    if self.tranpose:
      im = im.transpose((1,2,0))
    
    return im
      
  def augment(self,im,control=''):
    """
    augmentation includes: random crop, random mirror, random rotation + preprocess
    """
    im = np.uint8(im.copy()) #don't modify destructively
    if control:
      rot_ang = control['rot']
      im = ndimage.rotate(im, rot_ang, \
                            mode='nearest',reshape = False)
      flip = control['flip']
    else:
      #random zooming
      if self.zoom:
        zf = np.random.uniform(self.zoom[0],self.zoom[1])
        im = imresize(im, zf)
      #random rotation
      if self.rot:
        rot_ang = np.random.randint(self.rot[0], high = self.rot[1]+1)
        im = ndimage.rotate(im, rot_ang, \
                            mode='nearest',reshape = False)
      #random flip
      flip = np.random.choice(2)*2-1
    #1 comment to disable flip 
#==============================================================================
#     if self.flip:
#       im = im[:, ::flip]
#==============================================================================
    
    #random crop
    y_offset = np.random.randint(im.shape[0] - self.outshape[0])
    x_offset = np.random.randint(im.shape[1] - self.outshape[1])
    im = im[y_offset:self.outshape[0]+y_offset,
              x_offset:self.outshape[1]+x_offset]
    return self.preprocess(im)
  
  def augment_deploy(self,ims, enrich = False, crop = True):
    """
    same as augment() but applied to deploy network only:
    action: subtract mean, apply scale, central crop
    enrich: want to augment the data?
            False: no augmentation
            True: online augmentation
    """
    if enrich:
      assert ims.shape[0] <=2, 'Img enrich enabled. Batchsize should be <=2'
      #crop:5; flip:2; rotation:3 (-5,0,5);scale:3 (0.95, 1, 1.05)
      #totally 5x2x3x3=90 augments
      out = []
      for i in range(ims.shape[0]):
        im = ims[i].squeeze()[::-1].transpose((1,2,0))
        im_f = im[:,::-1] #flip
        #resize
        z = [imresize(im,0.95), imresize(im,1.05), im] + \
            [imresize(im_f,0.95), imresize(im_f,1.05), im_f]
        #rotate
        r = [ndimage.rotate(im_, -5) for im_ in z] + z +\
            [ndimage.rotate(im_, 5) for im_ in z]
        #crop
        r = [crop_center(im_,240,240) for im_ in r]
        r2 = [[im_[:self.outshape[-2],:self.outshape[-1]],
               im_[:self.outshape[-2],-self.outshape[-1]:],
               im_[-self.outshape[-2]:,:self.outshape[-1]],
               im_[-self.outshape[-2]:,-self.outshape[-1]:],
               crop_center(im_,self.outshape[-2],self.outshape[-1])
               ] for im_ in r]
        r2 = sum(r2,[])
        r2 = [self.preprocess(im_) for im_ in r2]
        out.extend(r2)
      return np.array(out)
      
    else:
      if crop:
        ims = np.float32(ims)
        #central crop
        x_offset = (ims.shape[-1] - self.outshape[-1])/2
        y_offset = (ims.shape[-2] - self.outshape[-2])/2
        ims =  ims[...,y_offset:y_offset+self.outshape[-2],
                   x_offset:x_offset+self.outshape[-1]]
        
        ims = (ims - self.mean)*self.scale
        return ims
      else:
        ims = np.uint8(ims)
        out = np.zeros(ims.shape[:2] + tuple(self.outshape[-2:]), dtype=np.float32)
        for i in range(out.shape[0]):
          img = ims[i][::-1].transpose(1,2,0)
          img = np.array(Image.fromarray(np.uint8(img)).resize(out.shape[-2:], Image.BILINEAR), dtype=np.float32)
          out[i,...] = img.transpose(2,0,1)[::-1]
        return (out - self.mean)*self.scale

def crop_center(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def edge_rotate(im,ang):
  """
  rotate edge map using nearest neighbour preserving edge and dimension
  Assumption: background 255, foreground 0
  currently does not work as good as ndimage.rotate
  """
  ang_rad = np.pi / 180.0 * ang
  H,W = np.float32(im.shape)
  R = mat([[np.cos(ang_rad),-np.sin(ang_rad) ,0],
        [np.sin(ang_rad), np.cos(ang_rad),0],
        [0              ,0               ,1.0]])
  T0 = mat([[1.0,0,-W/2],[0,1.0,-H/2],[0,0,1.0]])
  M0 = T0.I * R * T0
  
  tl_x,tl_y = np.floor(warping([0,0],M0))
  tr_x,tr_y = np.floor(warping([W-1,0],M0))
  bl_x,bl_y = np.floor(warping([0,H-1],M0))
  br_x,br_y = np.floor(warping([W-1,H-1],M0))
  
  minx = np.min([tl_x,tr_x,bl_x,br_x])
  maxx = np.max([tl_x,tr_x,bl_x,br_x])
  miny = np.min([tl_y,tr_y,bl_y,br_y])
  maxy = np.max([tl_y,tr_y,bl_y,br_y])
  T1 = mat([[1.0,0.0,minx],
            [0.0,1.0,miny],
            [0.0,0.0,1.0]])
  M1 = M0.I * T1
  nW = int(maxx - minx+1)
  nH = int(maxy - miny+1)
  
  out = np.ones((nH,nW),dtype=np.float32)*255
  for y in xrange(nH):
    for x in xrange(nW):
      u,v = np.int64(warping([x,y],M1))
      if u>=0 and u<W and v>=0 and v<H and im[v,u]!=255:
        out[y,x]=0
  
  return out

def warping(p,M):
  """Given 2-D point p and  warping matrix M, find the new location of p
  p:type list
  M: type matrix
  """
  p0 = mat([p[0],p[1],1.0]).T
  out = M*p0
  out /= out[2]
  return out[0],out[1]