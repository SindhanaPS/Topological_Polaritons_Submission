import numpy as np
from plot_func import *
from matplotlib import rc
from matplotlib import rcParams

Nk = 101
flag1 = 2

#####################################################
#                Importing data                     #
#####################################################

# 1D Eigenenergies
if flag1 == 1:
    data1 = np.loadtxt('cav_ex_bands_1D.txt')

    kind1 = data1[:,0]
    k1D = data1[:,1]
    E01D = data1[:,2]
    E11D = data1[:,3]
    E21D = data1[:,4]
    E31D = data1[:,5]
    Pol01D = data1[:,6]
    Pol11D = data1[:,7]
    Pol21D = data1[:,8]
    Pol31D = data1[:,9]

# 2D Berry curvature and eigenenergies
if flag1 == 2:
    data7 = np.loadtxt('cav_ex_bands.txt')

    data3 = np.loadtxt('cav_ex_berry_0.txt')
    data4 = np.loadtxt('cav_ex_berry_1.txt')
    data5 = np.loadtxt('cav_ex_berry_2.txt')
    data6 = np.loadtxt('cav_ex_berry_3.txt')

    # Eigenenergies
    kind3 = data7[:,0]
    kind4 = data7[:,1]

    kx = np.zeros((Nk,Nk),dtype=float)
    ky = np.zeros((Nk,Nk),dtype=float)
    E0=np.zeros((Nk,Nk),dtype=float)
    E1=np.zeros((Nk,Nk),dtype=float)
    E2=np.zeros((Nk,Nk),dtype=float)
    E3=np.zeros((Nk,Nk),dtype=float)

    for k in range(len(kind3)):
        i = int(kind3[k])
        j = int(kind4[k])
        kx[i,j] = data7[k,2]
        ky[i,j] = data7[k,3]
        E0[i,j] = data7[k,4]
        E1[i,j] = data7[k,5]
        E2[i,j] = data7[k,6]
        E3[i,j] = data7[k,7]
 
    # Berry curvature
    kind1 = data3[:,0]
    kind2 = data3[:,1]

    kx2=np.zeros((Nk-1,Nk-1),dtype=float)
    ky2=np.zeros((Nk-1,Nk-1),dtype=float)
    F0=np.zeros((Nk-1,Nk-1),dtype=float)
    F1=np.zeros((Nk-1,Nk-1),dtype=float)
    F2=np.zeros((Nk-1,Nk-1),dtype=float)
    F3=np.zeros((Nk-1,Nk-1),dtype=float)
    Pol0=np.zeros((Nk-1,Nk-1),dtype=float)
    Pol1=np.zeros((Nk-1,Nk-1),dtype=float)
    Pol2=np.zeros((Nk-1,Nk-1),dtype=float)
    Pol3=np.zeros((Nk-1,Nk-1),dtype=float)

    for k in range(len(kind1)):
        i = int(kind1[k])
        j = int(kind2[k])
        kx2[i,j] = data3[k,2]
        ky2[i,j] = data3[k,3]
        F0[i,j] = data3[k,4]
        F1[i,j] = data4[k,4]
        F2[i,j] = data5[k,4]
        F3[i,j] = data6[k,4]
        Pol0[i,j] = data3[k,5]
        Pol1[i,j] = data4[k,5]
        Pol2[i,j] = data5[k,5]
        Pol3[i,j] = data6[k,5]

# Stokes parameters
    data7 = np.loadtxt('cav_stokes_0.txt')
    data8 = np.loadtxt('cav_stokes_1.txt')

    kind1 = data7[:,0]
    kind2 = data7[:,1]

    kx3 = np.zeros((Nk,Nk),dtype=float)
    ky3 = np.zeros((Nk,Nk),dtype=float)
    S10 = np.zeros((Nk,Nk),dtype=float)
    S20 = np.zeros((Nk,Nk),dtype=float)
    S30 = np.zeros((Nk,Nk),dtype=float)
    S11 = np.zeros((Nk,Nk),dtype=float)
    S21 = np.zeros((Nk,Nk),dtype=float)
    S31 = np.zeros((Nk,Nk),dtype=float)

    for k in range(len(kind1)):
        i = int(kind1[k])
        j = int(kind2[k])
        kx3[i,j] = data7[k,2]
        ky3[i,j] = data7[k,3]
        S10[i,j] = data7[k,4]
        S20[i,j] = data7[k,5]
        S30[i,j] = data7[k,6]
        S11[i,j] = data8[k,4]
        S21[i,j] = data8[k,5]
        S31[i,j] = data8[k,6]


######################################################
#                 Formatting                         #
######################################################

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 24}

rc('font', **font)

rcParams['text.latex.preamble'] = [
       r'\usepackage{physics}',
       r'\usepackage{amsmath}',
]

rcParams['axes.linewidth'] = 1

######################################################
#                Plot band structure 1D              #
######################################################

if flag1 == 1:
    fname1 = 'cav_ex_bands_1D.pdf'
    CavExBands1Dbetter(k1D, E01D, E11D, E21D, E31D, Pol01D, Pol11D, Pol21D, Pol31D, fname1)

#####################################################
#        Plot Berry curvature 2D                    #
#####################################################

if flag1 == 2:
    fname7 = 'cav_ex_berry_0_2D.pdf'
    CavBerry2D(kx2, ky2, F0, fname7)

    fname8 = 'cav_ex_berry_1_2D.pdf'
    CavBerry2D(kx2, ky2, F1, fname8)

    fname9 = 'cav_ex_berry_2_2D.pdf'
    CavBerry2D(kx2, ky2, F2, fname9)

    fname10 = 'cav_ex_berry_3_2D.pdf'
    CavBerry2D(kx2, ky2, F3, fname10)

#####################################################
#        Plot Berry "polarization" 2D               #
#####################################################

if flag1 == 2:
    fname7 = 'Pol_0_2D.pdf'
    CavPol2D(kx2, ky2, Pol0, fname7)

    fname8 = 'Pol_1_2D.pdf'
    CavPol2D(kx2, ky2, Pol1, fname8)

    fname9 = 'Pol_2_2D.pdf'
    CavPol2D(kx2, ky2, Pol2, fname9)

    fname10 = 'Pol_3_2D.pdf'
    CavPol2D(kx2, ky2, Pol3, fname10)

######################################################
