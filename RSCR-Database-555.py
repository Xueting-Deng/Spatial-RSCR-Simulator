#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.optimize import root
import math
import random
import time
from tqdm.notebook import tqdm


# In[2]:


get_ipython().run_line_magic('run', '"C:\\SBU-3\\Jupyter-Research\\RSCR\\Functions.ipynb"')


# [RSPR.PNG](attachment:RSPR.PNG)

# In[3]:


# Fixed parameters, the order is 0,1,2,u1,u2
Fixed_para = np.array([
    [-1,0,0], #0 - fixed
    [0,0,0],  #1 - fixed
    [0,0,3],  #2
    [0,0,1],  #u1 - fixed
    [0,0,0]   #u2
], dtype=np.float32);


# In[4]:


angle = np.array([30,90,150])*np.pi/180


# In[5]:


rotational_axis = []
for ang1 in angle:
    uy = np.cos(ang1)
    for ang2 in angle:
        ux = np.sin(ang1)*np.cos(ang2)
        uz = np.sin(ang1)*np.sin(ang2)
        rotational_axis.append(np.array([ux,uy,uz]))
rotational_axis = np.array(rotational_axis, dtype=np.float32)


# In[3]:


Joint2 = np.load('Joint2_5.npy')
Joint3 = np.load('Joint3_5.npy')
Joint4 = np.load('Joint4_5.npy')


# In[4]:


len(Joint2)


# In[8]:


# Size of the dataset
mec_num = 823*9*9
temp_count = 0

step = 360; # this step decides how to devide 2pi
mec_data = np.zeros((mec_num*27,9,3), dtype=np.float64);
path_data = np.zeros((mec_num*27,step,3), dtype=np.float64);


# In[9]:


########## run start#############
temp_count = 0
start_time = time.time()
fullyrotated_num = 0

for i in tqdm(range(1648,2471)):
    Fixed_para[2] = Joint2[i]
    Initial_pos_3 = Joint3[i]
    Initial_pos_4 = Joint4[i]

    for u2 in rotational_axis:
        Fixed_para[4] = u2
        for u5 in rotational_axis:
            temp_count += 1
            Initial_u5 = u5
            config_para = ComputeConfig(Fixed_para, Initial_pos_3, Initial_pos_4, Initial_u5);

            ######################################## Step 2: Calculate for whole rotation #############################################
            try:
                Initial_guess = Initial_pos_4, Initial_pos_4, Initial_u5;
                Initial_para = Initial_guess;
                Initial_para, condition = solve_equations(Fixed_para, config_para, Initial_guess, Initial_pos_3);
                if condition == False:
                    continue

                phi = 0;
                step_pos_3 = np.zeros((step,3), dtype=np.float64); # store every position of point_3
                step_para = np.zeros((step,9), dtype=np.float64); # store every solution
                #storage_data = np.zeros((step,27,3), dtype=np.float64); #store rigidbody data we need

                #store the initial data
                step_pos_3[0,:] = Initial_pos_3;
                step_para[0,:] = np.reshape(Initial_para,(1,9));

                u_13 = np.append(0, Initial_pos_3) - np.append(0,Fixed_para[1]);

                for i in range(step-1):  
                    phi = phi + 360/step;

                    #below calculate the next position of 3 by quaternion
                    theta = np.pi*phi/360; #thera is half of phi
                    rotation_quater = np.array([np.cos(theta), np.sin(theta)*Fixed_para[3,0], np.sin(theta)*Fixed_para[3,1], np.sin(theta)*Fixed_para[3,2]]);#the quaternion of rotation
                    new_u_13 = QuaterTimes(QuaterTimes(rotation_quater,u_13),QuaterConj(rotation_quater));
                    new_pos_3 = new_u_13[1:4] + Fixed_para[1,:];
                    step_pos_3[i+1,:] = new_pos_3;

                    temp_solution, condition = solve_equations(Fixed_para, config_para, step_para[i,:], new_pos_3);

                    if condition == True:
                        step_para[i+1,:] = np.reshape(temp_solution,(1,9));
                    else:
                        #print("no full rotation");
                        break

                if condition == True:
                    for i in range(step):
                        #storage_data[i,:,:] = RigidbodyMesh (step_para[i], step_pos_3[i]);
                        step_path = RigidbodyMesh (step_para[i], step_pos_3[i]);
                        for k in range(27):
                            path_data[fullyrotated_num*27+k,i,:] = step_path[k,:];

                    for j in range(27):
                        mec_data[fullyrotated_num*27+j,0:5,:] = Fixed_para;
                        mec_data[fullyrotated_num*27+j,5,:] = Initial_pos_3;
                        mec_data[fullyrotated_num*27+j,6,:] = Initial_pos_4;
                        mec_data[fullyrotated_num*27+j,7,:] = Initial_u5;
                        mec_data[fullyrotated_num*27+j,8,:] = path_data[fullyrotated_num*27+j,0,:];

                    fullyrotated_num += 1;
                    if fullyrotated_num %1000 == 0 or fullyrotated_num==1:
                        print("Fully rotated machine number: %d" % (fullyrotated_num))
                        print("Number of attemps: %d" % (temp_count))
                        print("Running Time:", round((time.time() - start_time)/60,2),"mins")

                else:
                    continue
            except:
                pass


########## run end #############
print("--------------The End---------------------")
print("Running Time:", round((time.time() - start_time)/60,2),"mins")
print("Fully rotated machine number: %d" % (fullyrotated_num))
print("Number of attemps: %d" % (temp_count))


# In[33]:


mec_data = mec_data[0:122526]
path_data = path_data[0:122526]


# In[34]:


np.save("C:\SBU-3\Jupyter-Research\RSCR\Saved Data\\3_555_Different_Ratio\Mec3.npy",mec_data)
np.save("C:\SBU-3\Jupyter-Research\RSCR\Saved Data\\3_555_Different_Ratio\Path3.npy",path_data)


# In[ ]:




