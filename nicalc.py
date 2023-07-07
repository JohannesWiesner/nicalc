#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate statistics from Nifti-Files.

@author: jwiesner
"""

import numpy as np
import pandas as pd

from nilearn.image import math_img
from nilearn.image import new_img_like
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

def calculate_img_similarity(ref_img,src_img,mask_img,similarity_type='spearmanr'):
    '''Calculate similarity / distance between to nifti-images.
    
    Parameters
    ----------
    ref_img : Niimg-like object, array_like
        First nifti file or an array-like object.
    src_img : Niimg-like object, array_like
        Second nifti file or an array-like object.
    mask_img : Niimg-like object or None
        Mask image which will be applied to both input images. If None, both
        ref_img and src_img must be provided as arrays (i.e. already masked nifti files).
    similarity_type : str, optional
        Can be 'spearmanr' for Spearman's rank correlation coefficient,  
        'pearsonr' for Pearson correlation coefficient, or 'cosine_similarity' for
        cosine similarity. Default='spearmanr'.
    Returns
    -------
    img_similarity : float
        The distance / similarity coefficient.

    '''
    
    if mask_img:
        ref_img_data = apply_mask(ref_img,mask_img).reshape(1,-1)
        src_img_data = apply_mask(src_img,mask_img).reshape(1,-1)
    else:
        ref_img_data = ref_img.reshape(1,-1)
        src_img_data = src_img.reshape(1,-1)

    if similarity_type == 'cosine_similarity':
        img_similarity = cosine_similarity(ref_img_data,src_img_data)
    
    elif similarity_type == 'spearmanr':
        img_similarity,_ = spearmanr(ref_img_data,src_img_data,axis=1)
        
    if similarity_type == 'pearsonr':
        ref_img_data = ref_img_data.reshape(-1)
        src_img_data = src_img_data.reshape(-1)
        img_similarity,_ = pearsonr(ref_img_data,src_img_data)
        
    return img_similarity

def get_similarity_matrix(img_dict,mask_img,similarity_type='spearmanr'):
    '''Compute similarity / distance between multiple images
    
    Parameters
    ----------
    img_dict : dict
        dictionary with where keys are the names of the images and values
        are either a nifti file or an array-like object.
    mask_img : TYPE
        Mask image which will be applied to images. If None, images must be 
        provided as arrays (i.e. already masked nifti files).
    similarity_type : TYPE, optional
        Can be 'spearmanr' for Spearman's rank correlation coefficient,  
        'pearsonr' for Pearson correlation coefficient, or 'cosine_similarity' for
        cosine similarity. Default='spearmanr'.

    Returns
    -------
    similarity_matrix : pd.DataFrame
        Similarity matrix for all images.

    '''
    
    # compute similarity between all images
    imgs = list(img_dict.values())
    img_similarities = [calculate_img_similarity(ref_img,src_img,mask_img,similarity_type=similarity_type) for (ref_img,src_img) in combinations(imgs,2)]
    
    # create empty data frame
    n_imgs = len(img_dict)
    similarity_matrix = pd.DataFrame(np.zeros((n_imgs,n_imgs)))
    
    # get combinations of image names
    img_names = img_dict.keys()
    img_name_combos = [(ref_name,src_name) for ref_name,src_name in combinations(list(img_names),2)]
    
    # fill data frame with similarity values
    similarity_matrix = similarity_matrix.reindex(img_names)
    similarity_matrix.columns = img_names
    
    for idx,combo in enumerate(img_name_combos):
        similarity_matrix.loc[combo[0],combo[1]] = img_similarities[idx]
    
    return similarity_matrix

def calculate_overlap(first_mask_img,second_mask_img,proportion_type='first'):
    
    # get image data
    first_mask_img_data = first_mask_img.get_fdata()
    second_mask_img_data = second_mask_img.get_fdata()
    
    # get overlap array
    overlap = np.logical_and(first_mask_img_data,second_mask_img_data)
    size_overlap = np.count_nonzero(overlap)
    
    # get other logical arrays which can be contrasted to overlap array
    both_imgs_data = np.logical_or(first_mask_img_data,second_mask_img_data)
    size_both_imgs = np.count_nonzero(both_imgs_data)
    size_first_mask_img = np.count_nonzero(first_mask_img_data)
    
    if proportion_type == 'first':
        overlap_proportion = (size_overlap / size_first_mask_img) * 100
    
    elif proportion_type == 'both':
        overlap_proportion = (size_overlap / size_both_imgs) * 100
    
    return overlap_proportion

def check_mask_atlas_overlap(mask_img,atlas_img):
    
    # resample atlas to mask image
    # TO-DO: both stat mask and atlas image contain integers. When one has to be
    # resampled to the other, continous resampling can not be used (labels
    # like 1.4 are not valid.). interpolation keyword seems to do the trick:
    # Using interpolation == 'nearest' seems to keep resampled labels as integers.
    atlas_img = resample_to_img(source_img=atlas_img,target_img=mask_img,interpolation='nearest')
    
    # get intersection of mask image and atlas image
    intersect_img = math_img('img_1 * img_2',img_1=atlas_img,img_2=mask_img)
    
    # get region labels and region sizes of atlas image and intersection image
    atlas_labels, atlas_sizes = np.unique(atlas_img.get_fdata(), return_counts=True)
    intersect_img_labels, intersect_img_counts = np.unique(intersect_img.get_fdata(), return_counts=True)
    
    # check which labels are not shared by atlas image and intersection image
    not_shared_labels = np.array([list(set(atlas_labels) - set(intersect_img_labels))]).reshape(-1,)
    not_shared_zeros = np.zeros(not_shared_labels.shape)
    
    # fill up labels and sizes arrays with not shared labels and corresponding
    # zero sizes
    intersect_img_labels_all = np.concatenate((intersect_img_labels,not_shared_labels))
    intersect_img_sizes_all = np.concatenate((intersect_img_counts,not_shared_zeros))
    
    # sort filled up intersection image sizes
    intersect_img_labels_all_sorted_indices = np.argsort(intersect_img_labels_all)
    intersect_img_sizes_all_sorted = intersect_img_sizes_all[intersect_img_labels_all_sorted_indices]
    
    # compare atlas sizes and intersection image sizes
    overlap_proportion = (intersect_img_sizes_all_sorted / atlas_sizes) * 100
    
    return overlap_proportion
    
def calculate_mask_spread(mask_img):
    
    mask_img_data = mask_img.get_fdata()
    voxel_coords = np.argwhere(mask_img_data == 1)
    
    if voxel_coords.size == 0:
        avg_distance = 0.0
    else:
        avg_distance = np.mean(pdist(voxel_coords))
    
    return avg_distance

def project_on_atlas(atlas_img,projection_dict):
    '''Project values from a dictionary that maps atlas indices to values onto
        the image and return projection image. The dictionary must not a
        0-key. Will be deprecated in favor of map_on_atlas
    

    Parameters
    ----------
    atlas_img : niimg-like object
        An atlas image.
    projection_dict : dict
        A dictionary where the keys are the atlas regions and the values
        are the values of those regions.

    Returns
    -------
    niimg-like object
        A niimg-like object with values mapped onto brain regions.

    '''

    # check if projection dictionary contains '0' as a key
    if 0 in projection_dict.keys():
        raise KeyError('Dictionary must not contain "0" key.')
            
    # get data from atlas and make sure that the atlas indices are integers
    atlas_img_data = atlas_img.get_fdata().astype(int)
    
    # map each idx-value combination on atlas data
    atlas_img_data_projections = np.zeros(atlas_img_data.shape)
        
    for key in projection_dict:
        atlas_img_data_projections[atlas_img_data == key] = projection_dict[key]
            
    # return nifti-file
    return new_img_like(atlas_img,atlas_img_data_projections)

def map_on_atlas(atlas_img,mapping_dict,background_label=0):
    '''Map values onto atlas regions using a dictionary

    Parameters
    ----------
    atlas_img : niimg-like object
        An atlas image.
    mapping_dict : dict
        A dictionary that maps atlas region indices onto values for each
        region. Must not contain the background label.
    background_label : int or float, optional
        Label used in atlas_img to represent background. Default=0.

    Returns
    -------
    niimg-like object
        A niimg-like object with values mapped onto brain regions.

    '''

    # mapping dict must not contain the background label as key
    if background_label in mapping_dict.keys():
        raise KeyError('Dictionary must not contain the background label as key')

    # get data from atlas and make sure that the atlas indices are integers
    atlas_img_data = atlas_img.get_fdata().astype(int)

    # map each idx-value combination on atlas data
    # use this approach: https://stackoverflow.com/a/16993364/8792159
    atlas_img_data = atlas_img.get_fdata().astype(int)
    u,inv = np.unique(atlas_img_data,return_inverse = True)
    atlas_img_data_mapping = np.array([mapping_dict.get(x,x) for x in u])[inv].reshape(atlas_img_data.shape)

    # return nifti-file
    return new_img_like(atlas_img,atlas_img_data_mapping)
