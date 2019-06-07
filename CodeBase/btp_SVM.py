#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 02:16:15 2019

@author: satish
"""

import csv
import numpy as np
import pandas as pd
import cv2
from sklearn.svm import SVC
from fastdtw import fastdtw
import pickle
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from sklearn.metrics import accuracy_score

FRAME_PREFIX = '/media/satish/My Passport/Natta/2_natta/7/Dancer6/Performance1/color_USB-VID_045E&PID_02BF-0000000000000000_'
ANNOTATION_FILE = '/home/satish/Desktop/BTP_new/S2_D6_Natta_Motions/D6_Final_Natta_P1_annotations/7.csv'

FRAME_PREFIX1 = '/media/satish/My Passport/Natta/2_natta/1/Dancer6/Performance2/color_USB-VID_045E&PID_02BF-0000000000000000_'
ANNOTATION_FILE1 = '/home/satish/Desktop/BTP_new/S2_D6_Natta_Motions/D6_Final_Natta_P2_annotations/1.csv'

FRAME_PREFIX_TEST = '/media/satish/My Passport/Natta/2_natta/7/Dancer6/Performance3/color_USB-VID_045E&PID_02BF-0000000000000000_'
ANNOTATION_FILE_TEST = '/home/satish/Desktop/BTP_new/S2_D6_Natta_Motions/D6_Final_Natta_P3_annotations/7.csv'


Max_frames = 0;
def parse_annotation_file(annotation_file):
    global Max_frames
    motion_framelist = []
    actual_output  = []
    
    f1 = open(annotation_file,'r')
    reader = csv.reader(f1)
    for row in reader:
        if row[0]!="":
            st = int(row[1])
            end = int(row[2])
            if end-st >10:
                Max_frames = max(Max_frames,end-st)
                motion_framelist.append([st,end])
                actual_output.append(row[3])
    
    return motion_framelist,actual_output

motion_framelist1,actual_output1 = parse_annotation_file(ANNOTATION_FILE)
#motion_framelist2,actual_output2 = parse_annotation_file(ANNOTATION_FILE1)

motion_framelist3,actual_output3 = parse_annotation_file(ANNOTATION_FILE_TEST)


def getHistogram2(orientation,magnitude,n_cellsx,n_cellsy,cx=8,cy=8,orientations=18):
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[cy // 2:cy * n_cellsy:cy, cx // 2:cx * n_cellsx:cx]
    for i in range(orientations-1):
        temp_ori = np.where(orientation <= 360 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 360 / orientations * i,
                            temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = (temp_ori > -1)
        temp_mag = np.where(cond2, magnitude*(1-np.abs(orientation-(20 * (i)))/20), 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, int(i)] = temp_filt[subsample]
        
        temp_mag = np.where(cond2, magnitude*(1-np.abs(orientation-(20 * (i+1)))/20), 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        if 20 * (i+1)!=360:
            orientation_histogram[:, :, int(i+1)] = temp_filt[subsample]
        else:
            orientation_histogram[:, :, 0] = temp_filt[subsample]
    return orientation_histogram


def getHistogram1(orientation,magnitude,n_cellsx,n_cellsy,cx=8,cy=8,orientations=18):
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[cy // 2:cy * n_cellsy:cy, cx // 2:cx * n_cellsx:cx]
    for i in range(orientations-1):
        temp_ori = np.where(orientation < 360 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 360 / orientations * i,
                            temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = (temp_ori > -1)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, int(i)] = temp_filt[subsample]
    return orientation_histogram




def hof(flow, orientations=18, pixels_per_cell=(8, 8),visualise=False,method=1):

    
    flow = np.atleast_2d(flow)

   

    if flow.ndim < 3:
        raise ValueError("Requires dense flow in both directions")
    
    if flow.dtype.kind == 'u':
        flow = flow.astype('float')

    gx = flow[:,:,0]
    gy = flow[:,:,1]



    
#    magnitude = sqrt(gx**2 + gy**2)
#    
#    orientation = arctan2(gy, gx) * (180 / pi) +180
    magnitude,orientation = cv2.cartToPolar(gx, gy,angleInDegrees=1)
#    print(np.min(orientation),np.max(orientation))
    
#    print(magnitude)
    sy, sx = flow.shape[:2]
#    print(sy,sx)
    cx, cy = pixels_per_cell

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y
    
    orientation_histogram = []
    
    if method==1:
        print("yo 1")
        orientation_histogram = getHistogram1(orientation,magnitude,n_cellsx,n_cellsy)
    else: 
        orientation_histogram = getHistogram2(orientation,magnitude,n_cellsx,n_cellsy)
    hof_image = None

    if visualise:
        from skimage import draw

        radius = min(cx, cy) // 2 - 1
        hof_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations-1):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = int(radius * cos(float(o) / orientations * np.pi))
                    dy = int(radius * sin(float(o) / orientations * np.pi))
                    rr, cc = draw.line(centre[0] - dy, centre[1] - dx,
                                            centre[0] + dy, centre[1] + dx)
                    hof_image[rr, cc] += orientation_histogram[y, x, o]


    normalised_blocks = orientation_histogram
    
#    for x in range(n_blocksx):
#        for y in range(n_blocksy):
#            block = orientation_histogram[y:y+by, x:x+bx, :]
#            eps = 1e-5
#            normalised_blocks[y, x, :] = block 
#            normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)

    
    if visualise:
        return normalised_blocks.ravel(), hof_image
    else:
        return normalised_blocks.ravel()



def getoptflow(motion_framelist,frame_prefix):
    global Max_frames
    max_vecSize = Max_frames * 86400
    motion_flows = []
    for j in range(len(motion_framelist)):
        print(j)
        st,end = motion_framelist[j]
        i = st
        flows_motion3=[]
        flow = None
        frame1 = cv2.imread(frame_prefix+str(i)+'.png');
        prv =  cv2.cvtColor( frame1, cv2.COLOR_RGB2GRAY )
        while i < end :
            i = i+1
            frame2 = cv2.imread(frame_prefix+str(i)+'.png');
            
            nxt = cv2.cvtColor( frame2, cv2.COLOR_RGB2GRAY )
    
            flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 12, 2, 2, 1.0, 1)
            flow = hof(flow,method=2)
#            print(temp.shape)
            flows_motion3.append(flow)
        flows_motion3 = np.array(flows_motion3)
        flows_motion3 = flows_motion3.ravel()
        flows_motion3 = np.pad(flows_motion3, (0,max_vecSize-flows_motion3.shape[0]), 'constant', constant_values=(1e-5))
            
        motion_flows.append(flows_motion3)
    return motion_flows

motion_flows1  = getoptflow(motion_framelist1,FRAME_PREFIX)   
#motion_flows2 = getoptflow(motion_framelist2,FRAME_PREFIX1)
motion_flows3 = getoptflow(motion_framelist3,FRAME_PREFIX_TEST)

motion_flows = motion_flows1 
actual_output = actual_output1


def frame_diff(frame11 , frame22):
    frame1 = frame11
    frame2 = frame22
    diff = np.subtract(frame1,frame2);
    diff = np.sum(diff)
    diff = np.sqrt(diff)
    dis = diff
    return dis



clf = SVC()
clf.fit(motion_flows,actual_output)

print("training done")

ans = clf.predict(motion_flows3)

print(accuracy_score(ans,actual_output3))



    
    

