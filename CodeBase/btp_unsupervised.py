#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 05:32:58 2018

@author: satish
"""

import csv
import numpy as np
np.set_printoptions(threshold=10000)
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
import cv2 
from fastdtw import fastdtw
import seaborn as sns
from sklearn import metrics
import scipy
import math
from scipy.sparse import csgraph
from numpy.linalg import norm 
from scipy.sparse.linalg import eigsh
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from tslearn.metrics import dtw







#HardDisk1
#
#FRAME_PREFIX = '/media/satish/Seagate Backup Plus Drive/2_natta/3/Dancer1/color_USB-VID_045E&PID_02BF-0000000000000000_'
#AUTUAL_OUTPUT = '/home/satish/Desktop/BTP_new/D1_Actual/D1_Natta_3_motion.csv'


#HardDisk2
FRAME_PREFIX = '/media/satish/My Passport/Natta/2_natta/4/Dancer6/Performance2/color_USB-VID_045E&PID_02BF-0000000000000000_'
AUTUAL_OUTPUT = '/home/satish/Desktop/BTP_new/D6_Final_Natta_P2_annotations/4.csv'



FRAME_FEATURESIZE = 0




#weighted binning
def getHistogram2(orientation,magnitude,n_cellsx,n_cellsy,cx=8,cy=8,orientations=36):
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

    


#interval binning
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



def hof(flow, orientations=9, pixels_per_cell=(8, 8),visualise=False,method=1):

    
    flow = np.atleast_2d(flow)

   

    if flow.ndim < 3:
        raise ValueError("Requires dense flow in both directions")
    
    if flow.dtype.kind == 'u':
        flow = flow.astype('float')

    gx = flow[:,:,0]
    gy = flow[:,:,1]



    magnitude,orientation = cv2.cartToPolar(gx, gy,angleInDegrees=1)
    sy, sx = flow.shape[:2]
#    print(sy,sx)
    cx, cy = pixels_per_cell

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y
    
    orientation_histogram = []
    
    if method==1:
        orientation_histogram = getHistogram1(orientation,magnitude,n_cellsx,n_cellsy,orientations=orientations)
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
    

    
    if visualise:
        return normalised_blocks.ravel(), hof_image
    else:
        return normalised_blocks.ravel()





motion_framelist = []



f1 = open(AUTUAL_OUTPUT,'r')
reader1 = csv.reader(f1)

Max_frames = 0;

actual_clusters = []
i = 0
for curr1 in reader1:
     st = int(curr1[1])
     end = int(curr1[2])
     if end - st >= 10 :
         Max_frames = max(Max_frames,end-st)
         print(i,end-st)
         motion_framelist.append([st,end])
         clus = curr1[3]
         actual_clusters.append(clus)
         i += 1

print(Max_frames)
print(actual_clusters)
NUM_CLUSTERS = len(np.unique(actual_clusters))







    
motion_flows = []
color_frame=[]

for j in range(len(motion_framelist)):
    print(j)
    st,end = motion_framelist[j]
    i = st
    flows_motion3=[]
    flow = None
    frame1 = cv2.imread(FRAME_PREFIX+str(i)+'.png');
    prv =  cv2.cvtColor( frame1, cv2.COLOR_RGB2GRAY )
    while i < end :
        i = i+1
        frame2 = cv2.imread(FRAME_PREFIX+str(i)+'.png');
        
        nxt = cv2.cvtColor( frame2, cv2.COLOR_RGB2GRAY )

        flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 12, 2, 2, 1.0, 1)
#        flow = hof(flow,method=1)   #uncomment this line to get HOF features
#        
        
        
        flows_motion3.append(flow)
        
#        prv = nxt
    motion_flows.append(flows_motion3)
    

    
print("hof done")






    
#
def frame_diff(frame1,frame2):
    diff = np.subtract(frame1,frame2)
    diff = np.square(diff);
    dis = np.sum(diff)
    return np.sqrt(dis)
#    


p=0
n = len(motion_flows)
dtw_array = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
#        if abs(len(motion_flows[i])- len(motion_flows[j])) > 6 :
#            dtw_array[i][j] = -1.0
#        else :
        dtw_array[i][j], path1 = fastdtw(motion_flows[i], motion_flows[j], dist=frame_diff)

        print(p,dtw_array[i][j])
        p += 1
#        dtw_array[i][j]= np.exp((-0.1 * dtw_array[i][j] )**2)
        if dtw_array[i][j] != 0:
            dtw_array[i][j] = 1/dtw_array[i][j]
            dtw_array[j][i] = dtw_array[i][j]
#    print("\n")
        
    
df = pd.DataFrame(dtw_array)
temp_max = df.values.max()


        
#t = df==-1
#df[t] = temp_max+10000
#
for i in range(n):
    df[i][i]=temp_max
    
    

for i in range(n):
    for j in range(i,n):
        df[i][j]= int((df[i][j]*100)/temp_max)
        df[j][i]=int(df[i][j])

sns.heatmap(df, annot=True,annot_kws={"size": 7},
            xticklabels=df.columns.values,
            yticklabels=df.columns.values,ax=l1_ax,fmt='g')

Sc = cluster.SpectralClustering(n_clusters=NUM_CLUSTERS,affinity='precomputed')
out = Sc.fit_predict(dtw_array)
print(out)
print(actual_clusters)
confusion_matrix = np.zeros((NUM_CLUSTERS,NUM_CLUSTERS))
#
df2 = pd.DataFrame({'Actual': actual_clusters, 'Clusters': out})
ct = pd.crosstab(df2['Actual'],df2['Clusters'])
length1_fig, l2_ax = plt.subplots()
sns.heatmap(ct, annot=True,annot_kws={"size": 7},ax=l2_ax,fmt='g')
#
#
#
print("Total no of motions : ",n );
print("rand_index score : ",metrics.adjusted_rand_score(actual_clusters, out) )
#






