#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate statistics from Nifti-Files.

@author: jwiesner
"""

import numpy as np
import pandas as pd

from nilearn.image import math_img
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

def calculate_img_similarity(img_1,img_2,mask_img,similarity_type='cosine_similarity'):
                             
    img_1_data = apply_mask(img_1,mask_img).reshape(1,-1)
    img_2_data = apply_mask(img_2,mask_img).reshape(1,-1)

    if similarity_type == 'cosine_similarity':
        (img_similarity) = cosine_similarity(img_1_data,img_2_data)
    
    elif similarity_type == 'spearmanr':
        (img_similarity) = spearmanr(img_1_data,img_2_data,axis=1)
        
    if similarity_type == 'pearsonr':
        img_1_data = img_1_data.reshape(-1)
        img_2_data = img_2_data.reshape(-1)
        (img_similarity) = pearsonr(img_1_data,img_2_data)
        
    return img_similarity

def get_similarity_matrix(img_dict,mask_img,similarity_type='cosine_similarity'):
    
    # get combinations of stat image names
    img_names = img_dict.keys()
    img_name_combos = [(first_key,second_key) for first_key,second_key in combinations(list(img_names),2)]
    
    stat_imgs = list(img_dict.values())
    img_combinations = combinations(stat_imgs,2)
    img_similarities = [calculate_img_similarity(img_1,img_2,mask_img,similarity_type=similarity_type) for (img_1,img_2) in img_combinations]
    
    # create empty data frame
    n_imgs = len(img_dict)
    df = pd.DataFrame(np.zeros((n_imgs,n_imgs)))
    
    # FIXME: It would be more pretty to use reversed names in order
    # to add the correlation 'triangle in the bottom left corner' but this
    # somehow leads to a rotation of the y-labels when plotting.
    # img_names_reversed = list(img_names)[::-1]
    # df = df.reindex(img_names_reversed)
    # df.columns = img_names_reversed
    df = df.reindex(img_names)
    df.columns = img_names
    
    # fill data frame with similarity values
    for idx,combi in enumerate(img_name_combos):
        df.loc[combi[0],combi[1]] = img_similarities[idx]
    
    return df

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