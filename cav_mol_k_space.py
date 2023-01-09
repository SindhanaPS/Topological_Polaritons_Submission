import numpy as np
from cmath import pi
from cav_func import *
from cav_mol_func import *
from topo2 import *

Nk = 101               # has to be odd
fp = 0.3                 # dimensionless  fdown (for Ce:YAG)
fm = 0                # dimensionless    fup (for Ce:YAG)

##### pset = 1 we will use parameter set porphyrin
##### pset = 2 we will use parameter set Ce:YAG
##### pset = 3 we will use parameter set monolayer MoS2
pset = 1

##############################################
#            Parameter set porphyrin         #
##############################################
if pset == 1:
    E0 = 3.8                   # in eV
    beta0 = 0.1                # in eV
    beta = 9*10**(-4)           # in eV.(\mu m)^2
    zeta = 2.5*10**(-3)         # in eV.(\mu m)
    m = 125                     # in hbar^2*eV^(-1)*(\mu m)^(-2)
    rho = 3.55*10**(5)         # in (\mu m)^(-2)
    L = 0.745                  # in \mu m
    eps = 1.5                    # dimensionless
    mu0 = 2.84           # in D
    we = 3.8056            # in eV
    g = 0                  # in eV
    Nz = 100               # number of molecular layers along the z axis
    n = 11                  # cavity mode number

    # Transition dipole moments in units of D
    mup = np.sqrt(1-fm-2*fp)*mu0*np.array([1,1j,0])/np.sqrt(2)
    mum = np.sqrt(1-fp-2*fm)*mu0*np.array([1,-1j,0])/np.sqrt(2)

    fp = 0 
    fm = 0

##############################################

##############################################
#            Parameter set Ce:YAG            #
##############################################
if pset == 2:
    E0 = 2.5                   # in eV
    beta0 = 0.1                # in eV
    beta = 9*10**(-4)           # in eV.(\mu m)^2
    zeta = 2.5*10**(-3)         # in eV.(\mu m)
    m = 125                     # in hbar^2*eV^(-1)*(\mu m)^(-2)
    rho = 1.25*10**(7)         # in (\mu m)^(-2)
    L = 0.745                  # in \mu m
    eps = 12                    # dimensionless
    mu0 = 5.46           # in D
    we = 2.53            # in eV
    g = 0                  # in eV
    Nz = 1               # number of molecular layers along the z axis (since I consider a 0.1\mu m thick slab, I have already included it as an effective aerial density)
    n = 9                  # cavity mode number

    # Transition dipole moments in units of D
    mup = np.sqrt(fp)*mu0*np.array([1,1j,0])/np.sqrt(2)
    mum = np.sqrt(fm)*mu0*np.array([1,-1j,0])/np.sqrt(2)

    fp = 0 
    fm = 0

##############################################

##############################################
#            Parameter set MoS2              #
##############################################
if pset == 3:
    E0 = 1.8                   # in eV
    beta0 = 0.1                # in eV
    beta = 9*10**(-4)           # in eV.(\mu m)^2
    zeta = 2.5*10**(-3)         # in eV.(\mu m)
    m = 125                     # in hbar^2*eV^(-1)*(\mu m)^(-2)

    L = 0.745                  # in \mu m
    # rho, eps, mu0 are chosen such that mu0sqrt(rho*we/2L*eps*eps0)=19.5meV
    rho = 10**(6)         # in (\mu m)^(-2)
    eps = 1                    # dimensionless
    mu0 = 6.24           # in D

    we = 1.855            # in eV
    g = 0                  # in eV
    Nz = 1               # number of molecular layers along the z axis
    n = 5                  # cavity mode number

    # Transition dipole moments in units of D
    mup = np.sqrt(1-2*fp)*mu0*np.array([1,1j,0])/np.sqrt(2)
    mum = np.sqrt(1-2*fm)*mu0*np.array([1,-1j,0])/np.sqrt(2)

    fp = 0 
    fm = 0

##############################################

# in \mu m^{-1} i.e. (micro meter)^{-1}
maxk = 13

# Grid in k-space
Kx = np.linspace(-maxk,maxk,num=Nk)
Ky = np.linspace(-maxk,maxk,num=Nk)
kx,ky = np.meshgrid(Kx,Ky,indexing='ij')

# Hamiltonian diagonalization
####### flag1=1 1D band structure
####### flag1=2 2D band structure
flag1 = 2
phi = 0


if flag1 == 1:
    fname5 = 'cav_ex_bands_1D.txt'
elif flag1 == 2:
    fname5 = 'cav_ex_bands.txt'

fname6 = 'cav_ex_bands_vector_0.txt'
fname7 = 'cav_ex_bands_vector_1.txt'
fname8 = 'cav_ex_bands_vector_2.txt'
fname9 = 'cav_ex_bands_vector_3.txt'

kx,ky,H,E,v = CavExHamil(E0, m, zeta, beta0, beta, Nk, maxk, rho, L, eps, we, g, mup, mum, fp, fm, fname5, fname6, fname7, fname8, fname9, flag1, phi, Nz, n)

# Chern number calculation
if flag1 == 2:
    # Regular Berry curvature computation using the full 4-band model
    fname10 = 'cav_ex_berry_0.txt'
    fname11 = 'cav_ex_berry_1.txt'
    fname12 = 'cav_ex_berry_2.txt'
    fname13 = 'cav_ex_berry_3.txt'    

    C0,F0 = chern(Nk,kx,ky,v,0,fname10)
    C1,F1 = chern(Nk,kx,ky,v,1,fname11)
    C2,F2 = chern(Nk,kx,ky,v,2,fname12)
    C3,F3 = chern(Nk,kx,ky,v,3,fname13)

    print(C0,C1,C2,C3)
    print('Chern number=', int(round(C0.real)), int(round(C1.real)), int(round(C2.real)), int(round(C3.real)))

    # Berry curvature using a 2-band approximate Stokes vector
    fname0S = 'cav_stokes_0.txt'
    fname1S = 'cav_stokes_1.txt'
    fname14 = 'cav_berry_stokes_0.txt'
    fname15 = 'cav_berry_stokes_1.txt'

    S01,S02,S03 = Stokes(Nk,kx,ky,v,0,fname0S)
    S11,S12,S13 = Stokes(Nk,kx,ky,v,1,fname1S)

    bp0,bm0 = VectorFromStokes(Nk,kx,ky,S01,S02,S03)
    bp1,bm1 = VectorFromStokes(Nk,kx,ky,S11,S12,S13)

    vnew = np.zeros((Nk,Nk,2,2),dtype=np.complex_)

    for k1 in range(Nk):
       for k2 in range(Nk):
          for j in range(2):
             vnew[k1,k2,0,0] = bp0[k1,k2]
             vnew[k1,k2,1,0] = bm0[k1,k2]

             vnew[k1,k2,0,1] = bp1[k1,k2]
             vnew[k1,k2,1,1] = bm1[k1,k2]

    CS0,FS0 = chern(Nk,kx,ky,vnew,0,fname14)
    CS1,FS1 = chern(Nk,kx,ky,vnew,1,fname15)

    print(CS0,CS1)
    print('Chern number=', int(round(CS0.real)), int(round(CS1.real)))
