import numpy as np
import cmath as cm
from cmath import pi

def linkphase(v1,v2):
    return np.vdot(v1,v2)/abs(np.vdot(v1,v2))

def chern(Nk,kx,ky,v,idx,fname):
    file1 = open(fname, 'w')
    U=np.zeros((Nk,Nk,2),dtype=np.complex_)                                 # link variable
    F=np.zeros((Nk-1,Nk-1),dtype=np.complex_)                                   # lattice field strength
    Pol=np.zeros((Nk-1,Nk-1),dtype=np.float_)                                   # "Polarization"

    # Polarization of band
    for k1 in range(Nk-1):
        for k2 in range(Nk-1):
            bplus2 = (v[k1,k2,0,idx]*np.conj(v[k1,k2,0,idx])).real
            bminus2 = (v[k1,k2,1,idx]*np.conj(v[k1,k2,1,idx])).real
            Pol[k1,k2] = (bplus2-bminus2)/(bplus2+bminus2)

    for k1 in range(Nk-1):
        for k2 in range(Nk-1):
            U[k1,k2,0]=linkphase(v[k1,k2,:,idx],v[k1+1,k2,:,idx])
            U[k1,k2,1]=linkphase(v[k1,k2,:,idx],v[k1,k2+1,:,idx])

    # Edge k points
    k1=Nk-1
    for k2 in range(Nk-1):
        U[k1,k2,1]=linkphase(v[k1,k2,:,idx],v[k1,k2+1,:,idx])
    k2=Nk-1
    for k1 in range(Nk-1):
        U[k1,k2,0]=linkphase(v[k1,k2,:,idx],v[k1+1,k2,:,idx])
 
    C=0                                                                     # Chern number
    v=0
    # Berry curvature
    for k1 in range(Nk-1):
        for k2 in range(Nk-1):
            F[k1,k2]=cm.log(U[k1,k2,0]*U[k1+1,k2,1]*np.conj(U[k1,k2+1,0])*np.conj(U[k1,k2,1]))
            print(k1, "  ", k2, "  ", kx[k1,k2], "  ", ky[k1,k2], "  ", F[k1,k2].imag, "  ", Pol[k1,k2], file=file1)
            C+=F[k1,k2]
    
    # Berry curvature: boundary
    Urest=1
    # bottom edge: right to left
    k2=0
    for k1 in range(Nk-1):
        Urest*=np.conj(U[k1,k2,0])
    # left edge: bottom to top
    k1=0
    for k2 in range(Nk-1):
        Urest*=U[k1,k2,1]
    # top edge: left to right
    k2=Nk-1
    for k1 in range(Nk-1):
        Urest*=U[k1,k2,0]
    # right edge: top to bottom
    k1=Nk-1
    for k2 in range(Nk-1):
        Urest*=np.conj(U[k1,k2,1])

    Frest=cm.log(Urest)
    C+=Frest

    C=C/(2*pi*1j)
    file1.close()
    return(C,F)
