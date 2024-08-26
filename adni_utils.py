#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:34:27 2018

@author: pakitochus
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:47:12 2018

@author: pakitochus
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
from skimage.util import crop

def compute_size(imshape, downsample=1):
    size_x, size_y, size_z = np.mgrid[0:(imshape[0]-1+1e-8):downsample, 0:(imshape[1]-1+1e-8):downsample, 0:(imshape[2]-1+1e-8):downsample].shape[1:]
    return (size_x, size_y, size_z)


def load_img_arr(file, downsample=1, smooth=False, fsize=2):
    image = nib.load(file)
    if downsample==1:
        imarray =image.get_data()
        imarray[np.isnan(imarray)] = 0
    else:
        array = image.get_data()
        array[np.isnan(array)] = 0
        if smooth:
            array = ndimage.gaussian_filter(array, sigma=(fsize,fsize,fsize))
        xx, yy, zz = np.arange(array.shape[0]),np.arange(array.shape[1]),np.arange(array.shape[2])
        interpolator = RegularGridInterpolator((xx,yy,zz), np.squeeze(array), bounds_error=True)
        xi, yi, zi = np.mgrid[0:(array.shape[0]-1+1e-8):downsample, 
                              0:(array.shape[1]-1+1e-8):downsample, 
                              0:(array.shape[2]-1+1e-8):downsample]
        imarray = interpolator((xi, yi, zi)) 
    return imarray


def cut_img(imarray, final_size):
    coza = np.nanmean(np.nanmean(imarray,axis=1), axis=1)
    th = 0.05*(coza.max()-coza.min())+coza.min()
    ind_x = np.argwhere(coza>th)[[0, -1]].flatten()
    
    coza = np.nanmean(np.nanmean(imarray,axis=0), axis=1)
    th = 0.05*(coza.max()-coza.min())+coza.min()
    ind_y = np.argwhere(coza>th)[[0, -1]].flatten()
    
    coza = np.nanmean(np.nanmean(imarray,axis=1), axis=0)
    th = 0.05*(coza.max()-coza.min())+coza.min()
    ind_z = np.argwhere(coza>th)[[0, -1]].flatten()
    
    if np.diff(ind_y)>np.diff(ind_x):
        ind_x = ind_y-(ind_y.mean()-ind_x.mean()).astype(int)
        if ind_x[0]<0:
            ind_x = ind_x-ind_x[0]
    else:
        ind_y = ind_x-(ind_x.mean()-ind_y.mean()).astype(int)
        if ind_y[0]<0:
            ind_y = ind_y-ind_y[0]
    
    imarray = imarray[ind_x[0]:ind_x[1]+1,ind_y[0]:ind_y[1]+1,ind_z[0]:ind_z[1]+1]

    rest = np.array(imarray.shape)-np.array(final_size)
    diff = np.array([np.ceil(rest/2), np.floor(rest/2)]).T
    diff_crop = diff.copy()
    diff_crop[diff<0] = 0
    imfinal = crop(imarray, tuple(map(tuple, diff_crop)))
    diff_pad = -diff.copy()
    diff_pad[diff_pad<0] = 0
    imfinal = np.pad(imfinal, pad_width=tuple(map(tuple, diff_pad.astype(int))), mode='constant', constant_values=0)
    return imfinal 

def normalize(X, mode='integral'):
    out = np.zeros(X.shape)
    for i in range(X.shape[0]):
        out[i] = X[i]/X[i].mean()
    return out

def standardize(X):
    out = np.zeros(X.shape)
    for i in range(X.shape[0]):
        out[i] = (X[i]-X[i].mean)/X[i].std()
    return out

def smooth(X, fsize=5):
    out = np.zeros(X.shape)
    for i in range(X.shape[0]):
        out[i] = ndimage.gaussian_filter(X[i], sigma=(fsize,fsize,fsize))
    return out

def load_adni_stack(dbdir, imtype='gm', which='first', downsample=1, preprocessing=[None], fsize=3, lab='DX simple', final_size=None, num_data=None, num_data2=None):
    """
    imtype: type of imate ('gm', 'wm', 'norm'), 'orig' is not yet allowed.
    which: int (if session) or string ('first', 'last' 'all')
    preprocessing: list with 'norm', 'stand' and 'smooth'. 
    fsize: filter size, if 'smooth' is specified. 
    lab: options 'DX simple', 'DX Group', 'MMSE Total Score', 'GDSCALE Total Score', 
        'Global CDR', 'FAQ Total Score', 'NPI-Q Total Score', 'ADAS11', 'ADAS13'
    """
    imtypes = {'gm': 'GM', 'wm': 'WM', 'norm': 'Norm'} #, 'orig': 'Orig'
    datos = pd.read_csv(os.path.join(dbdir, 'dataset.csv'))
    if imtype.lower() in imtypes:
        imshape = (157, 189, 156)
        filter_data = datos.loc[datos.Tissue==imtypes[imtype.lower()],:].sort_values(by=['Subject', 'vCode'])
    else:
        raise TypeError("input not recognized")
    
    #filter_data.groupby('Subject').tail(1)
    
    if type(which) is str:
        if which=='first':
            select = filter_data.index.get_indexer(filter_data.groupby('Subject').head(1).index)
            N=select.shape[0]
        elif which=='last':
            select = filter_data.index.get_indexer(filter_data.groupby('Subject').tail(1).index)
            N=select.shape[0]
        elif which=='all':
            select = filter_data.index.get_indexer(filter_data.index)
            N=select.shape[0]
    elif type(which) is int:
        select = filter_data.index.get_indexer(filter_data.groupby('Subject').nth(which).index)
        N=select.shape[0]
    
    if num_data is not None:
        select = select[:num_data]
    
    if num_data2 is not None:
        select = select[num_data2:]

    N = select.shape[0]
    subjects = filter_data.loc[filter_data.index[select],:]
    labels = subjects[lab].values   
 
    if lab is 'DX':
        labels_act = [el.split(' to ')[-1] for el in labels.astype(str)]
        labels[np.array(labels_act)=='Dementia'] = 'AD'
        labels[np.array(labels_act)=='NL'] = 'CTL'
        labels[np.array(labels_act)=='MCI'] = 'MCI'
        labels = labels.astype(str)
    
    if final_size is None:
        final_size = compute_size(imshape, downsample)

        
    dsname = '%dx%dx%d'%final_size
    canonical_name = os.path.join(dbdir, 'data_all_'+imtype.lower()+'_'+dsname)
    name = os.path.join(dbdir, 'data_'+str(which)+'_'+imtype.lower()+'_'+dsname)
    #for smoothing
    sm = False
    if 'smooth' in preprocessing:
        sm=True
        canonical_name = canonical_name+'_s'+str(fsize)
        name = name+'_s'+str(fsize)
        
    canonical_name = canonical_name+'.npy'
    name = name+'.npy'

    if os.path.isfile(name):
        stack = np.load(name)[select]
    elif os.path.isfile(canonical_name):
        stack = np.load(canonical_name)[select]
    else:
        print("non canonic")
        stack = np.zeros((N, final_size[0], final_size[1], final_size[2])) 
        for ix,sub in enumerate(subjects['file']):
            print('Loading '+sub+'...')
            new_path = os.path.join(dbdir, sub)
            print(new_path)
            stack[ix,:,:,:] = cut_img(load_img_arr(new_path, downsample=downsample, smooth=sm, fsize=fsize), final_size)
        print('Saving dataset')
        np.save(name, stack)

    if 'norm' in preprocessing:
        stack = normalize(stack)
    elif 'stand' in preprocessing:
        stack = normalize(stack)
    return stack, labels, stack.shape[1:]


def parse_labels(labels, order=None):
    if order is None:
        order= np.unique(labels)
    labels_num = np.zeros(len(labels))
    for ix,el in enumerate(order):
        labels_num[labels==el] = ix
    return labels_num, order
    
