# Clustering of dance motions
## Objective
Clustering of similar motions  in the given “Bharatanatyam” dance video.
### Input 
. The video was given in RGB format

. Annotation file with motion frame details.
### Output
. Cluster number for each motion

## Dataset Description

<table>
  <tr>
   <td>Adavu
   </td>
   <td>Performance1
<p>
 total frames
   </td>
   <td>Performance 2
<p>
 total frames
   </td>
   <td>Performance 3
<p>
 total frames
   </td>
   <td>Total motions
   </td>
   <td>no of unique motions
   </td>
  </tr>
  <tr>
   <td>Natta1
   </td>
   <td>1546
   </td>
   <td>1590
   </td>
   <td>1532
   </td>
   <td>32
   </td>
   <td>4
   </td>
  </tr>
  <tr>
   <td>Natta2
   </td>
   <td>1557
   </td>
   <td>1522
   </td>
   <td>1545
   </td>
   <td>32
   </td>
   <td>4
   </td>
  </tr>
  <tr>
   <td>Natta3
   </td>
   <td>2680
   </td>
   <td>2698
   </td>
   <td>2760
   </td>
   <td>64
   </td>
   <td>8
   </td>
  </tr>
  <tr>
   <td>Natta4
   </td>
   <td>5537
   </td>
   <td>5531
   </td>
   <td>5504
   </td>
   <td>128
   </td>
   <td>8
   </td>
  </tr>
  <tr>
   <td>Natta5
   </td>
   <td>2580
   </td>
   <td>2728
   </td>
   <td>2748
   </td>
   <td>64
   </td>
   <td>10
   </td>
  </tr>
  <tr>
   <td>Natta6
   </td>
   <td>2781
   </td>
   <td>2764
   </td>
   <td>2729
   </td>
   <td>64
   </td>
   <td>12
   </td>
  </tr>
  <tr>
   <td>Natta7
   </td>
   <td>2828
   </td>
   <td>3022
   </td>
   <td>2706
   </td>
   <td>64
   </td>
   <td>14
   </td>
  </tr>
  <tr>
   <td>Natta8
   </td>
   <td>2710
   </td>
   <td>2811
   </td>
   <td>2752
   </td>
   <td>48
   </td>
   <td>11
   </td>
  </tr>
</table>

## Challenges

. All the motions  didn’t  have  same no of frames even the similar ones.

. Also the frames of similar motions may be different.

. To compare similarity of two motions,need to find a good measure.

. The similarity measure should be more for similar motions and less for different motions

. It should be able to compare motions with different no of frames

## Overview of approaches

### Unsupervised Learning

. Used Dense optical flow and obtained the feature vector of each motion.

  . Tried various variations using HOF 
  
. Applied DTW  as similarity measure for motions and obtained similarity matrix.

. Used Spectral Clustering to cluster the data using the above similarity matrix.

### Supervised Learning

. Used Dense optical flow to get the features for each motion.

. Made all vectors equal size by appending with small value(1e-5)

. Used SVM for the classification


