"""
Get the mean values of phalanxes from SynthHand data.
"""

import numpy as np
import pickle
import os

links_dict = \
{  "0":  [1, 5, 9, 13, 17],   # wrist: [T0, I0, M0, R0, L0]
   "1":  [0, 2, 5],           # T0:    [W,  T1]
   "2":  [1, 3],              # T1:    [T0, T2]
   "3":  [4, 2],              # T2:    [T1, T3]
   "4":  [0, 3],              # T3:    [W,  T2]
   "5":  [0, 1, 6, 9],   
   "6":  [5, 7], 
   "7":  [6, 8],
   "8":  [0, 7], 
   "9":  [0, 5, 10, 13],
   "10": [9, 11],
   "11": [10, 12],
   "12": [0, 11],
   "13": [0, 9, 14, 17], 
   "14": [13, 15], 
   "15": [14, 16], 
   "16": [0, 15],
   "17": [0, 13, 18],
   "18": [17, 19],
   "19": [18, 20],
   "20": [0, 19]
}

mean_dict = \
{  "0":  [0, 0, 0, 0, 0],   
   "1":  [0, 0, 0],
   "2":  [0, 0],              
   "3":  [0, 0],              
   "4":  [0, 0],              
   "5":  [0, 0, 0, 0],   
   "6":  [0, 0], 
   "7":  [0, 0],
   "8":  [0, 0], 
   "9":  [0, 0, 0, 0],
   "10": [0, 0],
   "11": [0, 0],
   "12": [0, 0],
   "13": [0, 0, 0, 0], 
   "14": [0, 0], 
   "15": [0, 0], 
   "16": [0, 0],
   "17": [0, 0, 0],
   "18": [0, 0],
   "19": [0, 0],
   "20": [0, 0]
}

src_dir = r"F:\DataSets\Transformed_SynthHands"
counter = 0

for root, dirs, files in os.walk(src_dir):
    if files != []:
        for f in files:
            
            if ".txt" in f:
                counter += 1
                pos_arr = np.loadtxt(os.path.join(root, f))
                
                for key, values in links_dict.items():
                    for i, val in enumerate(values):
                        mean_dict[key][i] += np.sqrt(np.sum((pos_arr[int(key)] - pos_arr[val])**2))
                        

for key, values in mean_dict.items():
    txt = "{}: [".format(key)
    for i, val in enumerate(values):
        mean_dict[key][i] /= counter
        txt += "{:.3f}, ".format(mean_dict[key][i])
    
    txt += "],"
    print(txt)    
                    
