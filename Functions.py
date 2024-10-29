#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def IfRemove(path, th=1):
    condition = False
    dist = np.sum(np.power(np.subtract(path[0,:],path[-1,:]),2))
    if dist > th:
        condition = True
    return condition

def RemoveOpen (path_set, mec_set):
    num = len(path_set)
    num2 = len(mec_set)
    
    if num != num2:
        print('Datasets have different length')
    
    inx = []
    for i in range(num):
        con = IfRemove(path_set[i])
        if con == True:
            inx.append(i)
            
    path_set = np.delete(path_set,inx, 0)
    mec_set = np.delete(mec_set,inx, 0)
    
    return path_set, mec_set


# In[ ]:


def Display_num(arr,prec):
    np.set_printoptions(suppress=True,precision=prec)
    print(arr)


# In[1]:


# Plot path
def plotPath(Pts, limit=5, color = 'gray',linestyle = 'line1', dot_size = 1):

        
    if linestyle == 'line1':
        xline=Pts[:,0]
        yline=Pts[:,1]
        zline=Pts[:,2]
        xline = np.append(xline,xline[0])
        yline = np.append(yline,yline[0])
        zline = np.append(zline,zline[0])
        ax.plot3D(xline, yline, zline, color)
    elif linestyle == 'point1':
        ax.scatter3D(Pts[:,0], Pts[:,1], Pts[:,2], c=color, s=dot_size)
        #ax.scatter3D(Pts[0,0], Pts[0,1], Pts[0,2], c='Green', s=dot_size*5)
        #ax.scatter3D(Pts[-1,0], Pts[-1,1], Pts[-1,2], c='Blue', s=dot_size*5)
    elif linestyle == 'point2':
        ax.scatter3D(Pts[0,:], Pts[1,:], Pts[2,:], c=color, s=dot_size)
        #ax.scatter3D(Pts[0,0], Pts[1,0], Pts[2,0], c='Green', s=dot_size*5)
        #ax.scatter3D(Pts[0,-1], Pts[1,-1], Pts[2,-1], c='Blue', s=dot_size*5)
    elif linestyle == 'line2':
        xline=Pts[0,:]
        yline=Pts[1,:]
        zline=Pts[2,:]
        xline = np.append(xline,xline[0])
        yline = np.append(yline,yline[0])
        zline = np.append(zline,zline[0])
        ax.plot3D(xline, yline, zline, color)
    else:
        print('wrong linesyle')
    
    #ax.auto_scale_xyz([-limit, limit], [-limit, limit], [-limit, limit])
    #plt.tight_layout()


# In[2]:


def plotLine3D(P1,P2, color='gray',linewidth=2,alpha=1):
    ax.plot3D([P1[0],P2[0]], [P1[1],P2[1]], [P1[2],P2[2]],linewidth=linewidth, color=color, alpha=alpha)


# In[1]:


# Plot RSCR
def plotMec_RSCR(mec, tpe='1', transparency=1):
    if tpe == '1':
        Fix_0 = mec[0]; Fix_1 = mec[1]; Fix_2 = mec[2]; u_1 = mec[3]; u_2 = mec[4]
        Mov_3 = mec[5]; Mov_4 = mec[6]; u_5 = mec[7]; Coupler_6 = mec[8]
        ax.scatter3D(mec[1:3,0],mec[1:3,1],mec[1:3,2],marker='x',linewidths=1, color='black', alpha = transparency)
    elif tpe == '2':
        Fix_1 = mec[0]; Fix_2 = mec[1]; u_1 = mec[2]; u_2 = mec[3]
        Mov_3 = mec[4]; Mov_4 = mec[5]; u_5 = mec[6]; Coupler_6 = mec[7]
        ax.scatter3D(mec[0:2,0],mec[0:2,1],mec[0:2,2],marker='x',linewidths=1, color='black', alpha = transparency)

    
    plotLine3D(Fix_1, Mov_3, color='red', alpha = transparency) #13
    plotLine3D(Mov_3, Mov_4, color='green', alpha = transparency) #34
    plotLine3D(Fix_2, Mov_4, color='blue', alpha = transparency) #25(=24)
    plotLine3D(Mov_3, Coupler_6, color='green', alpha = transparency)#36
    plotLine3D(Coupler_6, Mov_4, color='green', alpha = transparency)#46
    
    #45
    ax.plot3D([Mov_4[0], Mov_4[0]+u_5[0]/5], [Mov_4[1], Mov_4[1]+u_5[1]/5], [Mov_4[2], Mov_4[2]+u_5[2]/5],linewidth=2, color='green', alpha = transparency) 
    ax.plot3D([Mov_4[0], Mov_4[0]-u_5[0]/5], [Mov_4[1], Mov_4[1]-u_5[1]/5], [Mov_4[2], Mov_4[2]-u_5[2]/5],linewidth=2, color='green', alpha = transparency) 

    
    #joint_45
    ax.plot3D([Mov_4[0], Mov_4[0]+u_5[0]/40], [Mov_4[1], Mov_4[1]+u_5[1]/40], [Mov_4[2], Mov_4[2]+u_5[2]/40],linewidth=6, color='blue', alpha = transparency) 
    ax.plot3D([Mov_4[0], Mov_4[0]-u_5[0]/40], [Mov_4[1], Mov_4[1]-u_5[1]/40], [Mov_4[2], Mov_4[2]-u_5[2]/40],linewidth=6, color='blue', alpha = transparency) 
    
    #arrow u1 and u2
    ax.quiver(Fix_1[0],Fix_1[1],Fix_1[2],-u_1[0]*1.2,-u_1[1]*1.2,-u_1[2]*1.2,colors='red', length=0.1, alpha = transparency) #u1
    ax.quiver(Fix_2[0],Fix_2[1],Fix_2[2],-u_2[0]*1.2,-u_2[1]*1.2,-u_2[2]*1.2,colors='blue', length=0.1, alpha = transparency) #u2
    
    #joint_1
    ax.plot3D([Fix_1[0], Fix_1[0]+u_1[0]/40], [Fix_1[1], Fix_1[1]+u_1[1]/40], [Fix_1[2], Fix_1[2]+u_1[2]/40],linewidth=6, color='gray', alpha = transparency) 
    ax.plot3D([Fix_1[0], Fix_1[0]-u_1[0]/40], [Fix_1[1], Fix_1[1]-u_1[1]/40], [Fix_1[2], Fix_1[2]-u_1[2]/40],linewidth=6, color='gray', alpha = transparency)
    
    #joint_2
    ax.plot3D([Fix_2[0], Fix_2[0]+u_2[0]/40], [Fix_2[1], Fix_2[1]+u_2[1]/40], [Fix_2[2], Fix_2[2]+u_2[2]/40],linewidth=6, color='gray', alpha = transparency) 
    ax.plot3D([Fix_2[0], Fix_2[0]-u_2[0]/40], [Fix_2[1], Fix_2[1]-u_2[1]/40], [Fix_2[2], Fix_2[2]-u_2[2]/40],linewidth=6, color='gray', alpha = transparency) 
    
    ax.scatter3D(Mov_3[0],Mov_3[1],Mov_3[2],marker='o',linewidths=5, color='red', alpha = transparency)
    
    ax.scatter3D(Coupler_6[0],Coupler_6[1],Coupler_6[2],marker='o',linewidths=3, color='orange', alpha = transparency)
    


# In[ ]:


# Plot RSCR
def plotMec_RSCR_with5(mec, tpe='1', transparency=1):
    if tpe == '1':
        Fix_0 = mec[0]; Fix_1 = mec[1]; Fix_2 = mec[2]; u_1 = mec[3]; u_2 = mec[4]
        Mov_3 = mec[5]; Mov_4 = mec[6]; Mov_5 = mec[7];u_5 = mec[8]; Coupler_6 = mec[9]
        ax.scatter3D(mec[1:3,0],mec[1:3,1],mec[1:3,2],marker='x',linewidths=1, color='black', alpha = transparency)
    elif tpe == '2':
        Fix_1 = mec[0]; Fix_2 = mec[1]; u_1 = mec[2]; u_2 = mec[3]
        Mov_3 = mec[4]; Mov_4 = mec[5]; Mov_5 = mec[6];u_5 = mec[7]; Coupler_6 = mec[8]
        ax.scatter3D(mec[1:3,0],mec[1:3,1],mec[1:3,2],marker='x',linewidths=1, color='black', alpha = transparency)
    
    plotLine3D(Fix_1, Mov_3, color='red', alpha = transparency) #13
    plotLine3D(Mov_3, Mov_4, color='green', alpha = transparency) #34
    plotLine3D(Fix_2, Mov_5, color='blue', alpha = transparency) #25
    plotLine3D(Mov_3, Coupler_6, color='green', alpha = transparency)#36
    plotLine3D(Coupler_6, Mov_4, color='green', alpha = transparency)#46
    
    #45
    ax.plot3D([Mov_4[0], Mov_4[0]+5*u_5[0]], [Mov_4[1], Mov_4[1]+5*u_5[1]], [Mov_4[2], Mov_4[2]+5*u_5[2]],linewidth=2, color='green', alpha = transparency) 
    ax.plot3D([Mov_4[0], Mov_4[0]-5*u_5[0]], [Mov_4[1], Mov_4[1]-5*u_5[1]], [Mov_4[2], Mov_4[2]-5*u_5[2]],linewidth=2, color='green', alpha = transparency) 

    
    #joint_5
    ax.plot3D([Mov_5[0], Mov_5[0]+u_5[0]/10], [Mov_5[1], Mov_5[1]+u_5[1]/10], [Mov_5[2], Mov_5[2]+u_5[2]/10],linewidth=6, color='blue', alpha = transparency) 
    ax.plot3D([Mov_5[0], Mov_5[0]-u_5[0]/10], [Mov_5[1], Mov_5[1]-u_5[1]/10], [Mov_5[2], Mov_5[2]-u_5[2]/10],linewidth=6, color='blue', alpha = transparency) 
    
    #arrow u1 and u2
    ax.quiver(Fix_1[0],Fix_1[1],Fix_1[2],u_1[0],u_1[1],u_1[2],colors='red', length=0.1, alpha = transparency) #u1
    ax.quiver(Fix_2[0],Fix_2[1],Fix_2[2],u_2[0],u_2[1],u_2[2],colors='blue', length=0.1, alpha = transparency) #u2
    
    #joint_1
    ax.plot3D([Fix_1[0], Fix_1[0]+u_1[0]/10], [Fix_1[1], Fix_1[1]+u_1[1]/10], [Fix_1[2], Fix_1[2]+u_1[2]/10],linewidth=6, color='gray', alpha = transparency) 
    ax.plot3D([Fix_1[0], Fix_1[0]-u_1[0]/10], [Fix_1[1], Fix_1[1]-u_1[1]/10], [Fix_1[2], Fix_1[2]-u_1[2]/10],linewidth=6, color='gray', alpha = transparency)
    
    #joint_2
    ax.plot3D([Fix_2[0], Fix_2[0]+u_2[0]/10], [Fix_2[1], Fix_2[1]+u_2[1]/10], [Fix_2[2], Fix_2[2]+u_2[2]/10],linewidth=6, color='gray', alpha = transparency) 
    ax.plot3D([Fix_2[0], Fix_2[0]-u_2[0]/10], [Fix_2[1], Fix_2[1]-u_2[1]/10], [Fix_2[2], Fix_2[2]-u_2[2]/10],linewidth=6, color='gray', alpha = transparency) 
    
    ax.scatter3D(Mov_3[0],Mov_3[1],Mov_3[2],marker='o',linewidths=5, color='red', alpha = transparency)
    
    ax.scatter3D(Coupler_6[0],Coupler_6[1],Coupler_6[2],marker='o',linewidths=3, color='orange', alpha = transparency)
    


# In[ ]:


# Plot RSCR
def plotMec_RSCR_onecolor(mec, tpe='1', transparency=1,color='gray'):
    if tpe == '1':
        Fix_0 = mec[0]; Fix_1 = mec[1]; Fix_2 = mec[2]; u_1 = mec[3]; u_2 = mec[4]
        Mov_3 = mec[5]; Mov_4 = mec[6]; u_5 = mec[7]; Coupler_6 = mec[8]
        ax.scatter3D(mec[1:3,0],mec[1:3,1],mec[1:3,2],marker='x',linewidths=1, color=color, alpha = transparency)
    elif tpe == '2':
        Fix_1 = mec[0]; Fix_2 = mec[1]; u_1 = mec[2]; u_2 = mec[3]
        Mov_3 = mec[4]; Mov_4 = mec[5]; u_5 = mec[6]; Coupler_6 = mec[7]
        ax.scatter3D(mec[0:2,0],mec[0:2,1],mec[0:2,2],marker='x',linewidths=1, color=color, alpha = transparency)

    
    plotLine3D(Fix_1, Mov_3, color=color, alpha = transparency) #13
    plotLine3D(Mov_3, Mov_4, color=color, alpha = transparency) #34
    plotLine3D(Fix_2, Mov_4, color=color, alpha = transparency) #25(=24)
    plotLine3D(Mov_3, Coupler_6, color=color, alpha = transparency)#36
    plotLine3D(Coupler_6, Mov_4, color=color, alpha = transparency)#46
    
    #45
    ax.plot3D([Mov_4[0], Mov_4[0]+u_5[0]], [Mov_4[1], Mov_4[1]+u_5[1]], [Mov_4[2], Mov_4[2]+u_5[2]],linewidth=2, color=color, alpha = transparency) 
    ax.plot3D([Mov_4[0], Mov_4[0]-u_5[0]], [Mov_4[1], Mov_4[1]-u_5[1]], [Mov_4[2], Mov_4[2]-u_5[2]],linewidth=2, color=color, alpha = transparency) 

    
    #joint_45
    ax.plot3D([Mov_4[0], Mov_4[0]+u_5[0]/10], [Mov_4[1], Mov_4[1]+u_5[1]/10], [Mov_4[2], Mov_4[2]+u_5[2]/10],linewidth=6, color=color, alpha = transparency) 
    ax.plot3D([Mov_4[0], Mov_4[0]-u_5[0]/10], [Mov_4[1], Mov_4[1]-u_5[1]/10], [Mov_4[2], Mov_4[2]-u_5[2]/10],linewidth=6, color=color, alpha = transparency) 
    
    #arrow u1 and u2
    ax.quiver(Fix_1[0],Fix_1[1],Fix_1[2],u_1[0],u_1[1],u_1[2],colors=color, length=0.4, alpha = transparency) #u1
    ax.quiver(Fix_2[0],Fix_2[1],Fix_2[2],u_2[0],u_2[1],u_2[2],colors=color, length=0.4, alpha = transparency) #u2
    
    #joint_1
    ax.plot3D([Fix_1[0], Fix_1[0]+u_1[0]/10], [Fix_1[1], Fix_1[1]+u_1[1]/10], [Fix_1[2], Fix_1[2]+u_1[2]/10],linewidth=6, color=color, alpha = transparency) 
    ax.plot3D([Fix_1[0], Fix_1[0]-u_1[0]/10], [Fix_1[1], Fix_1[1]-u_1[1]/10], [Fix_1[2], Fix_1[2]-u_1[2]/10],linewidth=6, color=color, alpha = transparency)
    
    #joint_2
    ax.plot3D([Fix_2[0], Fix_2[0]+u_2[0]/10], [Fix_2[1], Fix_2[1]+u_2[1]/10], [Fix_2[2], Fix_2[2]+u_2[2]/10],linewidth=6, color=color, alpha = transparency) 
    ax.plot3D([Fix_2[0], Fix_2[0]-u_2[0]/10], [Fix_2[1], Fix_2[1]-u_2[1]/10], [Fix_2[2], Fix_2[2]-u_2[2]/10],linewidth=6, color=color, alpha = transparency) 
    
    ax.scatter3D(Mov_3[0],Mov_3[1],Mov_3[2],marker='o',linewidths=5, color=color, alpha = transparency)
    
    ax.scatter3D(Coupler_6[0],Coupler_6[1],Coupler_6[2],marker='o',linewidths=3, color=color, alpha = transparency)
    


# In[ ]:


#Evaluate the smooth factor of bspline
def Coordinate_meansquared(y1,y2):
    MSE = 0;
    sub = np.subtract(y1,y2);
    l = len(sub[0]);
    for i in range(l-1):
        diff = math.sqrt(sub[0,i]**2 + sub[1,i]**2 + sub[2,i]**2)
        MSE += diff
    MSE = MSE/l
    return MSE

def Path_Interpolate(path, num_pts=100, smooth=0.01):
    px = path[:,0]
    py = path[:,1]
    pz = path[:,2]
    
    px = np.append(px,px[0]);
    py = np.append(py,py[0]);
    pz = np.append(pz,pz[0]);
    
    tck, u = splprep([px,py,pz],s=smooth)

    u_fine = np.linspace(0,1,num_pts)
    new_points = splev(u_fine,tck)
    #nx, ny, nz = splev(u,tck)
    
    return new_points


# In[ ]:


def random_exclusive(low, high, exclude):
    inx = random.randint(0,1)
    if inx == 0:
        x = random.randint(low,exclude-1);
    else:
        x = random.randint(exclude+1, high)
    return x


# In[1]:


# Function of quaternion
def QuaterTimes(q1, q2):
    a, b, c, d = q1;
    e, f, g, h = q2;
    w = a*e - b*f - c*g - d*h;
    x = b*e + a*f - d*g + c*h;
    y = c*e + d*f + a*g - b*h;
    z = d*e - c*f + b*g + a*h;
    
    return np.array([w, x, y, z])

def QuaterConj(q):
    w, x, y, z = q;
    return np.array([w, -x, -y, -z])

def distance(a, b):
    
    dis = math.sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]) + (a[2] - b[2])*(a[2] - b[2]));
    
    return dis

def RigidbodyMesh (upara, new_pos_3):
    
    mesh_data = np.zeros((27,3), dtype=np.float64);
    
    center = 0.5*(upara[0:3]-new_pos_3) + new_pos_3;
    #x axis
    x_rigid = np.cross((new_pos_3 - upara[0:3]),upara[6:9]);
    x_rigid = x_rigid/np.linalg.norm(x_rigid);
    #y axis
    y_rigid = new_pos_3 - upara[0:3];
    y_rigid = y_rigid/np.linalg.norm(y_rigid);
    #z axis
    z_rigid = np.cross(x_rigid,y_rigid);
    z_rigid = z_rigid/np.linalg.norm(z_rigid);

    #mesh, unit = 2
    mesh_data[0,:] = center; #[0,0,0]

    mesh_data[1,:] = center + 2*z_rigid; #[0,0,1]
    mesh_data[2,:] = center - 2*z_rigid; #[0,0,-1]
    
    mesh_data[3,:] = center + 2*x_rigid; #[1,0,0]
    mesh_data[4,:] = center - 2*x_rigid; #[-1,0,0]
    
    mesh_data[5,:] = center + 2*y_rigid; #[0,1,0]
    mesh_data[6,:] = center - 2*y_rigid; #[0,-1,0]
    
    
    mesh_data[7,:] = center + 2*x_rigid + 2*y_rigid; #[1,1,0]
    mesh_data[8,:] = center + 2*x_rigid - 2*y_rigid; #[1,-1,0]
    mesh_data[9,:] = center - 2*x_rigid + 2*y_rigid; #[-1,1,0]
    mesh_data[10,:] = center - 2*x_rigid - 2*y_rigid; #[-1,-1,0]
    
    mesh_data[11,:] = center + 2*x_rigid + 2*z_rigid; #[1,0,1]
    mesh_data[12,:] = center + 2*x_rigid - 2*z_rigid; #[1,0,-1]
    mesh_data[13,:] = center - 2*x_rigid + 2*z_rigid; #[-1,0,1]
    mesh_data[14,:] = center - 2*x_rigid - 2*z_rigid; #[-1,0,-1]
    
    mesh_data[15,:] = center + 2*y_rigid + 2*z_rigid; #[0,1,1]
    mesh_data[16,:] = center + 2*y_rigid - 2*z_rigid; #[0,1,-1]
    mesh_data[17,:] = center - 2*y_rigid + 2*z_rigid; #[0,-1,1]
    mesh_data[18,:] = center - 2*y_rigid - 2*z_rigid; #[0,-1,-1]
    
    mesh_data[19,:] = center + 2*x_rigid + 2*y_rigid + 2*z_rigid; #[1,1,1]
    mesh_data[20,:] = center + 2*x_rigid + 2*y_rigid - 2*z_rigid; #[1,1,-1]
    mesh_data[21,:] = center + 2*x_rigid - 2*y_rigid + 2*z_rigid; #[1,-1,1]
    mesh_data[22,:] = center - 2*x_rigid + 2*y_rigid + 2*z_rigid; #[-1,1,1]
    mesh_data[23,:] = center + 2*x_rigid - 2*y_rigid - 2*z_rigid; #[1,-1,-1]
    mesh_data[24,:] = center - 2*x_rigid - 2*y_rigid + 2*z_rigid; #[-1,-1,1]
    mesh_data[25,:] = center - 2*x_rigid + 2*y_rigid - 2*z_rigid; #[-1,1,-1]
    mesh_data[26,:] = center - 2*x_rigid - 2*y_rigid - 2*z_rigid; #[-1,-1,-1]
    
    return mesh_data

def solve_equations(fpara, config, guess, new):
    
    def ConstrainFun (upara):
        upara = np.reshape(np.array(upara),(3,3));
        f1 = distance(new, upara[0,:]) - config[1]; #l_34
        f2 = distance(fpara[2,:], upara[1,:]) - config[2]; #l_25
        f3 = np.dot((new-upara[0,:]),upara[2,:]) - config[1]*np.cos(config[6]); #ang_43_u5
        f4 = np.dot(upara[2,:], fpara[4,:]) - np.cos(config[5]); #ang_u2_u5
        f5 = np.dot(upara[2,:], (fpara[2,:]-upara[1,:])) - config[2]*np.cos(config[4]); #ang_u5_u52
        f6 = np.dot(fpara[4,:], (upara[1,:]-fpara[2,:])) - config[2]*np.cos(config[3]); #ang_u2_u25
        f7 = math.sqrt((upara[2,0] - 0)*(upara[2,0] - 0) + (upara[2,1] - 0)*(upara[2,1] - 0) + (upara[2,2] - 0)*(upara[2,2] - 0)) - 1; #u5 is a unit vector
        #make sure u_45 and u_5 are parallel, f7-f9_2 has 3 valid constraint
        u45 = upara[1,:]-upara[0,:];
        cross_45_u5 = np.cross((upara[1,:]-upara[0,:]),upara[2,:]); #the cross product of u_45 and u_5
        if abs(upara[2,2]) <= 10e-8:
            f8 = u45[2];
            f9 = cross_45_u5[2];

        else:
            f8 = cross_45_u5[0];
            f9 = cross_45_u5[1];
        #print(np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9]))
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9])

    result = root(ConstrainFun, guess)# method='krylov')#, options={'xtol':1e-8})krylov lm
    
    if result.success:
        solution = result.x
        return solution, True
    else:
        mes = result.message;
        #print(mes)
        
        return None, False

def ComputeConfig (fpara, pos3, pos4, u5):
    
    #configuration of the mechanism, the order is l13, l34, l25, 
    #ang_25_u2, ang_52_u5
    #ang_u2_u5, ang_43_u5
    config = np.array([0,0,0,0,0,0,0], dtype=np.float64);
    
    config[0] = np.linalg.norm(fpara[1,:]-pos3); #l13
    config[1] = np.linalg.norm(pos4-pos3); #l34
    config[2] = np.linalg.norm(fpara[2,:]-pos4); #l25
    
    
    #ang_u2_u25
    u2_u25 = np.dot(fpara[4,:], -fpara[2,:]+pos4)/np.linalg.norm(-fpara[2,:]+pos4);
    u2_u25 = u2_u25 if abs(u2_u25-1) > 1e-6 else 1
    u2_u25 = (u2_u25 if u2_u25 > -np.pi else u2_u25+np.pi) if u2_u25 < np.pi else u2_u25-np.pi;#make sure the ang is in ((0,pi)
    config[3] = np.arccos(u2_u25);
    
    #ang_u5_u52
    u5_u52 = np.dot(u5, fpara[2,:]-pos4)/np.linalg.norm(fpara[2,:]-pos4);
    u5_u52 = u5_u52 if abs(u5_u52-1) > 1e-6 else 1
    u5_u52 = (u5_u52 if u5_u52 > -np.pi else u5_u52+np.pi) if u5_u52 < np.pi else u5_u52-np.pi;#make sure the ang is in ((0,pi)
    config[4] = np.arccos(u5_u52);
    
    
    #ang_u2_u5
    u2_u5 = np.dot(fpara[4,:], u5)/np.linalg.norm(fpara[4,:]);
    u2_u5 = u2_u5 if abs(u2_u5-1) > 1e-6 else 1
    u2_u5 = (u2_u5 if u2_u5 > -np.pi else u2_u5+np.pi) if u2_u5 < np.pi else u2_u5-np.pi;#make sure the ang is in ((0,pi)
    config[5] = np.arccos(u2_u5);

    #ang_u43_u5
    u34_u5 = np.dot(pos3-pos4, u5)/np.linalg.norm(pos3-pos4);
    u34_u5 = u34_u5 if abs(u34_u5-1) > 1e-6 else 1
    u34_u5 = (u34_u5 if u34_u5 > -np.pi else u34_u5+np.pi) if u34_u5 < np.pi else u34_u5-np.pi;#make sure the ang is in ((0,pi)
    config[6] = np.arccos(u34_u5);
    
    return config


# In[1]:


"Only when coupler is on 34"
def ComputeConfig_with_coupler (fpara, pos3, pos4, u5, pos6):
    
    #configuration of the mechanism, the order is l13, l34, l25, 
    #ang_25_u2, ang_25_u5
    #ang_u2_u5, ang_43_u5
    config = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64);
    
    config[0] = np.linalg.norm(fpara[1,:]-pos3); #l13
    config[1] = np.linalg.norm(pos4-pos3); #l34
    config[2] = np.linalg.norm(fpara[2,:]-pos4); #l25
    
    
    #ang_u2_u25
    u2_u25 = np.dot(fpara[4,:], -fpara[2,:]+pos4)/np.linalg.norm(-fpara[2,:]+pos4);
    #u2_u25 = u2_u25 if abs(u2_u25-1) > 1e-6 else 1
    u2_u25 = (u2_u25 if u2_u25 > -np.pi else u2_u25+np.pi) if u2_u25 < np.pi else u2_u25-np.pi;#make sure the ang is in ((0,pi)
    config[3] = np.arccos(u2_u25);
    
    #ang_u5_u52
    u5_u52 = np.dot(u5, fpara[2,:]-pos4)/np.linalg.norm(fpara[2,:]-pos4);
    #u5_u52 = u5_u52 if abs(u5_u52-1) > 1e-6 else 1
    u5_u52 = (u5_u52 if u5_u52 > -np.pi else u5_u52+np.pi) if u5_u52 < np.pi else u5_u52-np.pi;#make sure the ang is in ((0,pi)
    config[4] = np.arccos(u5_u52);
    
    
    #ang_u2_u5
    u2_u5 = np.dot(fpara[4,:], u5)/np.linalg.norm(fpara[4,:]);
    u2_u5 = u2_u5 if abs(u2_u5)-1 > 1e-6 else 1
    u2_u5 = (u2_u5 if u2_u5 > -np.pi else u2_u5+np.pi) if u2_u5 < np.pi else u2_u5-np.pi;#make sure the ang is in ((0,pi)
    config[5] = np.arccos(u2_u5);

    #ang_u34_u5
    u34_u5 = np.dot(pos3-pos4, u5)/np.linalg.norm(pos3-pos4);
    #u34_u5 = u34_u5 if abs(u34_u5-1) > 1e-6 else 1
    u34_u5 = (u34_u5 if u34_u5 > -np.pi else u34_u5+np.pi) if u34_u5 < np.pi else u34_u5-np.pi;#make sure the ang is in ((0,pi)
    config[6] = np.arccos(u34_u5);
    
    
    config[7] = np.linalg.norm(pos3-pos6); #l36
    
    return config


def solve_equations_with_coupler(fpara, config, guess, new):
    
    def ConstrainFun (upara):
        upara = np.reshape(np.array(upara),(4,3));
        f1 = distance(new, upara[0,:]) - config[1]; #l_34
        f2 = distance(fpara[2,:], upara[1,:]) - config[2]; #l_25
        f3 = np.dot((new-upara[0,:]),upara[2,:]) - config[1]*np.cos(config[6]); #ang_43_u5
        f4 = np.dot(upara[2,:], fpara[4,:]) - np.cos(config[5]); #ang_u2_u5
        f5 = np.dot(upara[2,:], (fpara[2,:]-upara[1,:])) - config[2]*np.cos(config[4]); #ang_u5_u52
        f6 = np.dot(fpara[4,:], (upara[1,:]-fpara[2,:])) - config[2]*np.cos(config[3]); #ang_u2_u25
        f7 = math.sqrt((upara[2,0] - 0)*(upara[2,0] - 0) + (upara[2,1] - 0)*(upara[2,1] - 0) + (upara[2,2] - 0)*(upara[2,2] - 0)) - 1; #u5 is a unit vector
        #make sure u_45 and u_5 are parallel, f7-f9_2 has 3 valid constraint
        u45 = upara[1,:]-upara[0,:];
        cross_45_u5 = np.cross((upara[1,:]-upara[0,:]),upara[2,:]); #the cross product of u_45 and u_5
        if abs(upara[2,2]) <= 10e-8:
            f8 = u45[2];
            f9 = cross_45_u5[2];

        else:
            f8 = cross_45_u5[0];
            f9 = cross_45_u5[1];
            
        
        f10 = distance(new, upara[3,:]) - config[7]; #l_36
        #make sure u_36 and u_34 are parallel
        u36 = upara[3,:] - new;
        u34 = upara[0,:] - new;
        cross_36_34 = np.cross(u36,u34); #the cross product of u_36 and u_34
        if abs(u34[2]) <= 10e-8:
            f11 = u36[2];
            f12 = cross_36_34[2];

        else:
            f11 = cross_36_34[0];
            f12 = cross_36_34[1];
        
        
        #print(np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]))

        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    

    result = root(ConstrainFun, guess,method='krylov')
    
    if result.success:
        solution = result.x
        return solution, True
    else:
        mes = result.message;
        print(mes)
        return None, False


# In[ ]:


"Calcutate mec's configuration for ml when u2 parallel to u5"
def ComputeConfig_u2u5same_unnormolized(mec):
    l13 = np.linalg.norm(mec[1,:] - mec[5,:])
    l25 = np.linalg.norm(mec[2,:] - mec[6,:])
    l34 = np.linalg.norm(mec[5,:] - mec[6,:])
    l36 = np.linalg.norm(mec[5,:] - mec[8,:])
    return(np.array([l13,l25,l34,l36]))

def ComputeConfig_u2u5same_normolized(mec):
    x1 = mec[1,0]; y1 = mec[1,1]; z1 = mec[1,2];
    x2 = mec[2,0]; y2 = mec[2,1]; z2 = mec[2,2];
    ux = mec[3,0]; uy = mec[3,1]; uz = mec[3,2];
    
    l13 = np.linalg.norm(mec[1,:] - mec[5,:])
    l25 = np.linalg.norm(mec[2,:] - mec[6,:])
    l34 = np.linalg.norm(mec[5,:] - mec[6,:])
    l36 = np.linalg.norm(mec[5,:] - mec[8,:])
    return(np.array([x1,y1,z1,x2,y2,z2,ux,uy,uz,l13,l25,l34,l36]))


# In[ ]:


def ComputeConfig_RSSR (fpara, pos3, pos4):
    
    #The order is l_13, l_34, l_42, ang_u1_13, ang_u2_24
    config = np.array([0.0,0.0,0.0,90,90]);
    
    config[0] = np.linalg.norm(fpara[1,:]-pos3); #l13
    config[1] = np.linalg.norm(pos4-pos3); #l34
    config[2] = np.linalg.norm(fpara[2,:]-pos4); #l42
    
    
    config[3] = 0.5*np.pi #ang_u1_13 = 90°
    config[4] = 0.5*np.pi #ang_u2_24 = 90°
    
    return config

def solve_equations_RSSR(fpara, config, guess, new):
    
    def ConstrainFun_RSSR (upara):
        #define geometry constraint functions
        #upara = np.reshape(np.array(upara),(3,3));
        f1 = np.linalg.norm(new-upara[0,:])-config[1]; #l_34
        print(f1)
        f2 = np.linalg.norm(upara[0,:]-fpara[1,:])-config[2]; #l_42
        f3 = np.dot((upara[0,:]-fpara[1,:]),fpara[3,:]) - np.cos(config[4])*config[2]; #ang_u2_24
        print(np.array([[f1],[f2],[f3]]))
        
        return np.array([[f1],[f2],[f3]])


    result = root(ConstrainFun_RSSR, guess)#, options={'xtol':1e-8})krylov lm method='lm'
    
    if result.success:
        solution = result.x
        return solution, True
    else:
        mes = result.message;
        print(mes)
        
        return None, False

def RigidbodyMesh_RSSR(new_pos_3, new_pos_4, fpara):
    
    mesh_data = np.zeros((27,3), dtype=np.float64);
    
    center = 0.5*(new_pos_4 - new_pos_3) + new_pos_3;
    #x axis
    x_rigid = new_pos_3 - new_pos_4;#34
    x_rigid = x_rigid/np.linalg.norm(x_rigid);
    
    #z axis # 31 x 34 np.cross((new_pos_3 - upara[0:3]),upara[6:9])
    z_rigid = np.cross((new_pos_3 - fpara[1,:]),(new_pos_3 - new_pos_4));
    z_rigid = z_rigid/np.linalg.norm(z_rigid);
    
    #y axis
    y_rigid = np.cross(z_rigid,x_rigid);
    y_rigid = y_rigid/np.linalg.norm(y_rigid);
    

    #mesh, unit = 2
    mesh_data[0,:] = center; #[0,0,0]

    mesh_data[1,:] = center + 2*z_rigid; #[0,0,1]
    mesh_data[2,:] = center - 2*z_rigid; #[0,0,-1]
    
    mesh_data[3,:] = center + 2*x_rigid; #[1,0,0]
    mesh_data[4,:] = center - 2*x_rigid; #[-1,0,0]
    
    mesh_data[5,:] = center + 2*y_rigid; #[0,1,0]
    mesh_data[6,:] = center - 2*y_rigid; #[0,-1,0]
    
    
    mesh_data[7,:] = center + 2*x_rigid + 2*y_rigid; #[1,1,0]
    mesh_data[8,:] = center + 2*x_rigid - 2*y_rigid; #[1,-1,0]
    mesh_data[9,:] = center - 2*x_rigid + 2*y_rigid; #[-1,1,0]
    mesh_data[10,:] = center - 2*x_rigid - 2*y_rigid; #[-1,-1,0]
    
    mesh_data[11,:] = center + 2*x_rigid + 2*z_rigid; #[1,0,1]
    mesh_data[12,:] = center + 2*x_rigid - 2*z_rigid; #[1,0,-1]
    mesh_data[13,:] = center - 2*x_rigid + 2*z_rigid; #[-1,0,1]
    mesh_data[14,:] = center - 2*x_rigid - 2*z_rigid; #[-1,0,-1]
    
    mesh_data[15,:] = center + 2*y_rigid + 2*z_rigid; #[0,1,1]
    mesh_data[16,:] = center + 2*y_rigid - 2*z_rigid; #[0,1,-1]
    mesh_data[17,:] = center - 2*y_rigid + 2*z_rigid; #[0,-1,1]
    mesh_data[18,:] = center - 2*y_rigid - 2*z_rigid; #[0,-1,-1]
    
    mesh_data[19,:] = center + 2*x_rigid + 2*y_rigid + 2*z_rigid; #[1,1,1]
    mesh_data[20,:] = center + 2*x_rigid + 2*y_rigid - 2*z_rigid; #[1,1,-1]
    mesh_data[21,:] = center + 2*x_rigid - 2*y_rigid + 2*z_rigid; #[1,-1,1]
    mesh_data[22,:] = center - 2*x_rigid + 2*y_rigid + 2*z_rigid; #[-1,1,1]
    mesh_data[23,:] = center + 2*x_rigid - 2*y_rigid - 2*z_rigid; #[1,-1,-1]
    mesh_data[24,:] = center - 2*x_rigid - 2*y_rigid + 2*z_rigid; #[-1,-1,1]
    mesh_data[25,:] = center - 2*x_rigid + 2*y_rigid - 2*z_rigid; #[-1,1,-1]
    mesh_data[26,:] = center - 2*x_rigid - 2*y_rigid - 2*z_rigid; #[-1,-1,-1]
    
    return mesh_data

