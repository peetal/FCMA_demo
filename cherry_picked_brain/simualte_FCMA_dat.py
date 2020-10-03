#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:41:27 2020

@author: peetal
"""

# simulate 10 subjects' fMRI data over 9 small features (ROI) for FCMA demo

import os, random
from nilearn import image, plotting
import numpy as np  
from brainiak.utils import fmrisim as sim 

# specify directory
dat_dir = '/path/to/FCMA_demo/cherry_picked_brain'

# Specify the volume parameters
trDuration = 1  # seconds
numTRs = 400 # How many TRs will you generate?

# Set up stimulus event time course parameters
event_duration = 15  # How long is each event
isi = 5  # What is the time between each event

# Specify signal magnitude parameters
signal_change = 10 # How much change is there in intensity for the max of the patterns across participants
multivariate_pattern = 1  # Do you want the signal to be a z scored pattern across voxels (1) or a univariate increase (0)

print('Load template of average voxel value')
template_nii = image.load_img(os.path.join(dat_dir, 'sub_template.nii.gz'))
template = template_nii.get_fdata()
dimensions = np.array(template.shape[0:3])

print('Create binary mask and normalize the template range')
mask, template = sim.mask_brain(volume=template,
                                mask_self=True)
mask_cherry = image.load_img(os.path.join(dat_dir, 'cherry_pick_brain_mask.nii.gz')).get_fdata()

# Load the noise dictionary
print('Loading noise parameters')
with open(os.path.join(dat_dir, 'sub_noise_dict.txt'), 'r') as f:
    noise_dict = f.read()
noise_dict = eval(noise_dict)
noise_dict['matched'] = 0

# ------------------------
# Cherry pick 1000 voxels 
# ------------------------
#brain_index = np.where(mask == 1)
#brain_coordinates = np.array([[x for x in brain_index[0]], [x for x in brain_index[1]], [x for x in brain_index[2]]])
#lucky_voxel = random.sample(range(brain_coordinates.shape[1]), 400)

## two sets of voxels
#for i, voxel_set in enumerate([lucky_voxel[:200], lucky_voxel[200:400]]):
#    lucky_voxel_coordinates = [brain_coordinates[:, voxel] for voxel in voxel_set]
#    lucky_voxel_mask = np.zeros(dimensions)
#    for vox in lucky_voxel_coordinates: # vox = lucky_voxel_coordinates[0]
#        lucky_voxel_mask[vox[0],vox[1],vox[2]] = 1
#    lucky_voxel_mask_nii = image.new_img_like(template_nii, lucky_voxel_mask, affine = template_nii.affine)
#    lucky_voxel_mask_nii.to_filename(os.path.join(dat_dir, f'cherry_pick200voxel_cond{i}.nii.gz'))

## full mask
#lucky_voxel_coordinates = [brain_coordinates[:, voxel] for voxel in lucky_voxel]
#lucky_voxel_mask = np.zeros(dimensions)
#for vox in lucky_voxel_coordinates: # vox = lucky_voxel_coordinates[0]
#    lucky_voxel_mask[vox[0],vox[1],vox[2]] = 1
#lucky_voxel_mask_nii = image.new_img_like(template_nii, lucky_voxel_mask, affine = template_nii.affine)
#lucky_voxel_mask_nii.to_filename(os.path.join(dat_dir, 'cherry_pick_brain_mask.nii.gz'))


# stimfunction across two conditions for each subject 
stimfunc_all = []
for sid in range(8): # sid = 0
    # Create the stimulus time course of the conditions
    total_time = int(numTRs * trDuration)
    events = int(total_time / (event_duration + isi))
    onsets_A = []
    onsets_B = []
    randoized_label = np.repeat([1,2],int(events/2)).tolist()
    random.shuffle(randoized_label)
    for event_counter, cond in enumerate(randoized_label):
        
        # Flip a coin for each epoch to determine whether it is A or B
        if cond == 1:
            onsets_A.append(event_counter * (event_duration + isi))
        elif cond == 2:
            onsets_B.append(event_counter * (event_duration + isi))
            
    temporal_res = 1 # How many timepoints per second of the stim function are to be generated?
    
    # Create a time course of events 
    stimfunc_A = sim.generate_stimfunction(onsets=onsets_A,
                                           event_durations=[event_duration],
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                          )
    
    stimfunc_B = sim.generate_stimfunction(onsets=onsets_B,
                                           event_durations=[event_duration],
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                           )
    # stimfunc per subject
    stimfunc_ppt = np.concatenate((stimfunc_A, stimfunc_B), axis = 1)
    
    stimfunc_all.append(stimfunc_ppt)
    
    print('Load ROIs')
    nii_A = image.load_img(os.path.join(dat_dir, 'cherry_pick200voxel_cond0.nii.gz'))
    nii_B = image.load_img(os.path.join(dat_dir, 'cherry_pick200voxel_cond1.nii.gz'))
    ROI_A = nii_A.get_fdata()
    ROI_B = nii_B.get_fdata()
    
    # How many voxels per ROI
    voxels_A = int(ROI_A.sum())
    voxels_B = int(ROI_B.sum())
    
    # Create a pattern of activity across the two voxels
    print('Creating signal pattern')
  
    pattern_A = np.random.rand(voxels_A).reshape((voxels_A, 1))
    pattern_B = np.random.rand(voxels_B).reshape((voxels_B, 1))
   
    # Multiply each pattern by each voxel time course
    # Noise was added to the design matrix, to make the correlation pattern noise, so FCMA could be challenging. 
    weights_A = np.tile(stimfunc_A, voxels_A) * pattern_A.T + np.random.normal(0,1, size = np.tile(stimfunc_A, voxels_A).shape) 
    weights_B = np.tile(stimfunc_B, voxels_B) * pattern_B.T + np.random.normal(0,1, size = np.tile(stimfunc_B, voxels_B).shape) 
        
    # Convolve the onsets with the HRF
    # TR less than feature is not good, but b/c this is simulated data, can ignore this concer. 
    print('Creating signal time course')
    signal_func_A = sim.convolve_hrf(stimfunction=weights_A,
                                   tr_duration=trDuration,
                                   temporal_resolution=temporal_res,
                                   scale_function=1,
                                   )
    
    signal_func_B = sim.convolve_hrf(stimfunction=weights_B,
                                   tr_duration=trDuration,
                                   temporal_resolution=temporal_res,
                                   scale_function=1,
                                   )
    
    
    # Multiply the signal by the signal change 
    signal_func_A =  signal_func_A * signal_change #+ signal_func_B * signal_change
    signal_func_B =  signal_func_B * signal_change #+ signal_func_A * signal_change

    # Combine the signal time course with the signal volume
    print('Creating signal volumes')
    signal_A = sim.apply_signal(signal_func_A,
                                ROI_A)
    
    signal_B = sim.apply_signal(signal_func_B,
                                ROI_B)
    
    # Combine the two signal timecourses
    signal = signal_A + signal_B
    
    # spare the true noise. 
    #print('Generating noise')
    #noise = sim.generate_noise(dimensions=dimensions,
    #                           stimfunction_tr=np.zeros((numTRs, 1)),
    #                           tr_duration=int(trDuration),
    #                           template=template,
    #                           mask=mask_cherry,
    #                           noise_dict=noise_dict,
    #                           temporal_proportion = 0.5)
    
    
    
    brain = signal #+ noise
    brain_nii = image.new_img_like(template_nii, brain, template_nii.affine)
    out_path = os.path.join(dat_dir, '../simulated_data')
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path, exist_ok=True)
    brain_nii.to_filename(os.path.join(out_path, f"sub_{sid}_sim_dat.nii.gz"))
    
     
# write out the simulated epoch file 
sim.export_epoch_file(stimfunction = stimfunc_all,
                      filename = os.path.join(dat_dir, '../simulated_data/sim_epoch_file.npy'),
                      tr_duration = 1.0,
                      temporal_resolution = 1.0,)

