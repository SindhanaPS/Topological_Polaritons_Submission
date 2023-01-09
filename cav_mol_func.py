import numpy as np
import scipy.linalg as la
from cmath import pi
from cav_func import *

def JTECoupl(E0, m, beta0, beta, rho, L, eps, mu, kx, ky, Nz, n):
    ####### This function computes the light matter coupling strength between a transition with dipole mu and
    ####### the TE mode of the cavity
    # rho is the areal density of molecules in our system in units of (\mu m)^(-2)
    # mu is the dipole moment in units of D, it is a 3*1 vector.
    # L is the cavity thickness in units of \mu m
    # eps is the dimensionless dielectric constant
    # kx, ky are the wavevectors in units of (\mu m)^(-1)
    # m is the effective mass in units of hbar^2*eV^(-1)*(mu m)^(-2)
    # n is the mode number of the cavity which takes on a non-zero integer value n=0,1,2,..
    # Nz is the number of molecular layers along the z direction

    # Units conversion
    # 1D = 0.020819*10^(-3) e. \mu m
    # permittivity of free space eps0 = 55.263 e^2/(eV. \mu m)
    Dconv = 0.020819*10**(-3)
    eps0 = 55.263
    
    # Energy of the TE and TM modes in units eV
    TM,TE = CavTETM(E0, m, beta0, beta, kx, ky)

    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)
    k = np.sqrt(k2)

    cosphi = np.divide(kx,k)
    sinphi = np.divide(ky,k)

    # This will have units (e.\mu m)^(-1) * \sqrt(eV)
    Coeff = -np.sqrt(Nz)*np.sqrt(rho)*np.sqrt(1/(L*eps*eps0))*np.sin(n*pi/2)

    # These will all have units of \sqrt(eV)
    sqrtTE = np.sqrt(TE)
    Xcomp = -np.multiply(sqrtTE,sinphi)
    Ycomp = np.multiply(sqrtTE,cosphi)
    Zcomp = 0

    # This will have units (e. \mu m)* \sqrt(eV)
    DotProd = np.multiply(Dconv*mu[0],Xcomp) + np.multiply(Dconv*mu[1],Ycomp) + np.multiply(Dconv*mu[2],Zcomp)

    # This will have units of eV
    JTEMu = np.multiply(Coeff,DotProd)

    return JTEMu

def JTMCoupl(E0, m, beta0, beta, rho, L, eps, mu, kx, ky, Nz, n):
    ####### This function computes the light matter coupling strength between a transition with dipole mu and
    ####### the TM mode of the cavity
    # rho is the areal density of molecules in our system in units of (\mu m)^(-2)
    # mu is the dipole moment in units of D, it is a 3*1 vector.
    # L is the cavity thickness in units of \mu m
    # eps is the dimensionless dielectric constant
    # kx, ky are the wavevectors in units of (\mu m)^(-1)
    # m is the effective mass in units of hbar^2*eV^(-1)*(mu m)^(-2)
    # n is the mode number of the cavity which takes on a non-zero integer value n=0,1,2,..
    # Nz is the number of molecular layers along the z direction

    # Units conversion
    # 1D = 0.020819*10^(-3) e. \mu m
    # permittivity of free space eps0 = 55.263 e^2/(eV. \mu m)
    Dconv = 0.020819*10**(-3)
    eps0 = 55.263
    
    # Energy of the TE and TM modes in units eV
    TM,TE = CavTETM(E0, m, beta0, beta, kx, ky)

    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)
    k = np.sqrt(k2)

    cosphi = np.divide(kx,k)
    sinphi = np.divide(ky,k)

    # This will have units (e)^(-1) * \sqrt(eV)
    Coeff = -np.sqrt(Nz)*np.sqrt(rho)*np.sqrt(1/(k2+(n*pi/L)**2))*np.sqrt(1/(L*eps*eps0))

    sqrtTM = np.sqrt(TM)
    # These will all have units of \sqrt(eV)*(\mu m)^(-1)
    Xcomp = np.sin(n*pi/2)*np.multiply(sqrtTM,cosphi)*n*pi/L
    Ycomp = np.sin(n*pi/2)*np.multiply(sqrtTM,sinphi)*n*pi/L
    Zcomp = -1j*np.cos(n*pi/2)*np.multiply(sqrtTM,k)

    # This will have units (e)* \sqrt(eV)
    DotProd = np.multiply(Dconv*mu[0],Xcomp) + np.multiply(Dconv*mu[1],Ycomp) + np.multiply(Dconv*mu[2],Zcomp)

    # This will have units of eV
    JTMMu = np.multiply(Coeff,DotProd)

    return JTMMu

def JplusCoupl(E0, m, beta0, beta, rho, L, eps, mu, kx, ky, Nz, n):
    ####### This function computes the light matter coupling strength between a transition with dipole mu and
    ####### the + mode of the cavity
    # rho is the areal density of molecules in our system in units of (\mu m)^(-2)
    # mu is the dipole moment in units of D
    # L is the cavity thickness in units of \mu m
    # eps is the dimensionless dielectric constant
    # kx, ky are the wavevectors in units of (\mu m)^(-1)
    # m is the effective mass in units of hbar^2*eV^(-1)*(mu m)^(-2)
    # n is the mode number of the cavity which takes on a non-zero integer value n=0,1,2,..
    # Nz is the number of molecular layers along the z direction

    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)
    k = np.sqrt(k2)
     
    if kx.ndim == 2: 
        N1 = np.shape(kx)[0]
        N2 = np.shape(kx)[1]

        JplusMu = np.zeros((N1,N2),dtype=np.complex_)

        # Correct the k = 0 point
        for k1 in range(N1):
            for k2 in range(N2):
                if k[k1,k2]==0:
                    JHMuk0, JVMuk0 = JHVMuk0(E0, m, beta0, beta, rho, L, eps, mu, Nz, n)
                    JplusMu[k1,k2] = (JHMuk0 + 1j*JVMuk0)/np.sqrt(2)
                else:
                    cosphi = np.divide(kx[k1,k2],k[k1,k2])
                    sinphi = np.divide(ky[k1,k2],k[k1,k2])
                    eiphi = cosphi + 1j*sinphi
   
                    # These will have units of eV
                    JTEMu = JTECoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1,k2], ky[k1,k2], Nz, n)
                    JTMMu = JTMCoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1,k2], ky[k1,k2], Nz, n)

                    # This will have units of eV
                    JplusMu[k1,k2] = eiphi*(JTMMu+1j*JTEMu)/np.sqrt(2)
    elif kx.ndim == 1:
        N1 = np.shape(kx)[0]
        
        JplusMu = np.zeros(N1,dtype=np.complex_)

        # Correct the k = 0 point
        for k1 in range(N1):
            if k[k1] == 0:
                JHMuk0, JVMuk0 = JHVMuk0(E0, m, beta0, beta, rho, L, eps, mu, Nz, n)
                JplusMu[k1] = (JHMuk0 + 1j*JVMuk0)/np.sqrt(2)
            else:
                cosphi = np.divide(kx[k1],k[k1])
                sinphi = np.divide(ky[k1],k[k1])
                eiphi = cosphi + 1j*sinphi
   
                # These will have units of eV
                JTEMu = JTECoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1], ky[k1], Nz, n)
                JTMMu = JTMCoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1], ky[k1], Nz, n)

                # This will have units of eV
                JplusMu[k1] = eiphi*(JTMMu+1j*JTEMu)/np.sqrt(2)
    
    return JplusMu

def JminusCoupl(E0, m, beta0, beta, rho, L, eps, mu, kx, ky, Nz, n):
    ####### This function computes the light matter coupling strength between a transition with dipole mu and
    ####### the - mode of the cavity
    # rho is the areal density of molecules in our system in units of (\mu m)^(-2)
    # mu is the dipole moment in units of D
    # L is the cavity thickness in units of \mu m
    # eps is the dimensionless dielectric constant
    # kx, ky are the wavevectors in units of (\mu m)^(-1)
    # m is the effective mass in units of hbar^2*eV^(-1)*(mu m)^(-2)
    # n is the mode number of the cavity which takes on a non-zero integer value n=0,1,2,..
    # Nz is the number of molecular layers along the z direction

    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)
    k = np.sqrt(k2)

    if kx.ndim == 2: 
        N1 = np.shape(kx)[0]
        N2 = np.shape(kx)[1]

        JminusMu = np.zeros((N1,N2),dtype=np.complex_)

        # Correct the k = 0 point
        for k1 in range(N1):
            for k2 in range(N2):
                if k[k1,k2]==0:
                    JHMuk0, JVMuk0 = JHVMuk0(E0, m, beta0, beta, rho, L, eps, mu, Nz, n)
                    JminusMu[k1,k2] = (JHMuk0 - 1j*JVMuk0)/np.sqrt(2)
                else:
                    cosphi = np.divide(kx[k1,k2],k[k1,k2])
                    sinphi = np.divide(ky[k1,k2],k[k1,k2])
                    emiphi = cosphi - 1j*sinphi
  
                    # These will have units of eV
                    JTEMu = JTECoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1,k2], ky[k1,k2], Nz, n)
                    JTMMu = JTMCoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1,k2], ky[k1,k2], Nz, n)

                    # This will have units of eV
                    JminusMu[k1,k2] = emiphi*(JTMMu - 1j*JTEMu)/np.sqrt(2) 
    elif kx.ndim == 1:
        N1 = np.shape(kx)[0]
        
        JminusMu = np.zeros(N1,dtype=np.complex_)

        # Correct the k = 0 point
        for k1 in range(N1):
            if k[k1] == 0:
                JHMuk0, JVMuk0 = JHVMuk0(E0, m, beta0, beta, rho, L, eps, mu, Nz, n)
                JminusMu[k1] = (JHMuk0 - 1j*JVMuk0)/np.sqrt(2)
            else:
                cosphi = np.divide(kx[k1],k[k1])
                sinphi = np.divide(ky[k1],k[k1])
                emiphi = cosphi - 1j*sinphi
   
                # These will have units of eV
                JTEMu = JTECoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1], ky[k1], Nz, n)
                JTMMu = JTMCoupl(E0, m, beta0, beta, rho, L, eps, mu, kx[k1], ky[k1], Nz, n)

                # This will have units of eV
                JminusMu[k1] = emiphi*(JTMMu - 1j*JTEMu)/np.sqrt(2)

    return JminusMu

def JHVMuk0(E0, m, beta0, beta, rho, L, eps, mu, Nz, n):
    ####### This function computes the light matter coupling strength between a transition with dipole mu and
    ####### the H and V modes of the cavity at k=0
    # rho is the areal density of molecules in our system in units of (\mu m)^(-2)
    # mu is the dipole moment in units of D, it is a 3*1 vector.
    # L is the cavity thickness in units of \mu m
    # eps is the dimensionless dielectric constant
    # m is the effective mass in units of hbar^2*eV^(-1)*(mu m)^(-2)
    # n is the mode number of the cavity which takes on a non-zero integer value n=0,1,2,..
    # Nz is the number of molecular layers along the z direction

    # Units conversion
    # 1D = 0.020819*10^(-3) e. \mu m
    # permittivity of free space eps0 = 55.263 e^2/(eV. \mu m)
    Dconv = 0.020819*10**(-3)
    eps0 = 55.263
    
    # Energy of the H and V modes at k=0 in units eV
    H,V = CavHV(E0, m, beta0, beta, 0, 0)

    # This will have units (e.\mu m)^(-1) * \sqrt(eV)
    Coeff = -np.sqrt(Nz)*np.sqrt(rho)*np.sqrt(1/(L*eps*eps0))*np.sin(n*pi/2)

    # These will all have units of \sqrt(eV)
    XcompH = np.sqrt(H)
    YcompH = 0
    ZcompH = 0

    XcompV = 0
    YcompV = np.sqrt(V)
    ZcompV = 0

    # This will have units (e. \mu m)* \sqrt(eV)
    DotProdH = np.multiply(Dconv*mu[0],XcompH) + np.multiply(Dconv*mu[1],YcompH) + np.multiply(Dconv*mu[2],ZcompH)
    DotProdV = np.multiply(Dconv*mu[0],XcompV) + np.multiply(Dconv*mu[1],YcompV) + np.multiply(Dconv*mu[2],ZcompV)

    # This will have units of eV
    JHMuk0 = np.multiply(Coeff,DotProdH)
    JVMuk0 = np.multiply(Coeff,DotProdV)

    return (JHMuk0,JVMuk0)

def CavExHamil(E0, m, zeta, beta0, beta, Nk, maxk, rho, L, eps, we, g, mup, mum, fp, fm, fname, fname2, fname3, fname4, fname5, flag1, phi, Nz, n):
    # File with kx, ky and energy of bands within a square centered at k=0 in k-space
    file1 = open(fname,'w')
    file2 = open(fname2, 'w')
    file3 = open(fname3, 'w')
    file4 = open(fname4, 'w')
    file5 = open(fname5, 'w')

    # Initialize variables
    if flag1 == 2:
        H = np.zeros((Nk,Nk,4,4),dtype=np.complex_)
        E = np.zeros((Nk,Nk,4),dtype=float)
        v = np.zeros((Nk,Nk,4,4),dtype=np.complex_)

        Kx = np.linspace(-maxk,maxk,num=Nk)
        Ky = np.linspace(-maxk,maxk,num=Nk)
        kx,ky = np.meshgrid(Kx,Ky,indexing='ij')
    elif flag1 == 1:
        H = np.zeros((Nk,4,4),dtype=np.complex_)
        E = np.zeros((Nk,4),dtype=float)
        v = np.zeros((Nk,4,4),dtype=np.complex_)
        Pol = np.zeros((Nk,4),dtype=float)

        kx = np.linspace(-maxk*np.cos(phi),maxk*np.cos(phi),num=Nk)
        ky = np.linspace(-maxk*np.sin(phi),maxk*np.sin(phi),num=Nk)
        k = np.linspace(-maxk,maxk,num=Nk)
        

    Eplus = CavPlus(E0, m, zeta, kx, ky)
    Eminus = CavMinus(E0, m, zeta, kx, ky)
    Coupl = CavCoupl(beta0, beta, kx, ky)
    JpMup = JplusCoupl(E0, m, beta0, beta, rho, L, eps, mup, kx, ky, Nz, n)
    JpMum = JplusCoupl(E0, m, beta0, beta, rho, L, eps, mum, kx, ky, Nz, n)
    JmMup = JminusCoupl(E0, m, beta0, beta, rho, L, eps, mup, kx, ky, Nz, n)
    JmMum = JminusCoupl(E0, m, beta0, beta, rho, L, eps, mum, kx, ky, Nz, n)

    # Hamiltonian definition and diagonalization
    if flag1 == 2:
        for k1 in range(Nk):
            for k2 in range(Nk):
                # Cavity plus mode
                H[k1,k2,0,0] = Eplus[k1,k2]
                # Cavity minus mode 
                H[k1,k2,1,1] = Eminus[k1,k2]
                # Cavity coupling
                H[k1,k2,0,1] = Coupl[k1,k2]
                H[k1,k2,1,0] = np.conj(H[k1,k2,0,1])
                # Molecule plus state
                H[k1,k2,2,2] = we + g
                # Molecule minus state
                H[k1,k2,3,3] = we - g
                # Cavity plus and molecule plus coupling
                H[k1,k2,2,0] = JpMup[k1,k2]*np.sqrt(1-fm-2*fp)
                H[k1,k2,0,2] = np.conj(H[k1,k2,2,0])
                # Cavity plus and molecule minus coupling
                H[k1,k2,3,0] = JpMum[k1,k2]*np.sqrt(1-fp-2*fm)
                H[k1,k2,0,3] = np.conj(H[k1,k2,3,0])
                # Cavity minus and molecule plus coupling
                H[k1,k2,2,1] = JmMup[k1,k2]*np.sqrt(1-fm-2*fp)
                H[k1,k2,1,2] = np.conj(H[k1,k2,2,1])
                # Cavity minus and molecule minus coupling
                H[k1,k2,3,1] = JmMum[k1,k2]*np.sqrt(1-fp-2*fm)
                H[k1,k2,1,3] = np.conj(H[k1,k2,3,1])

                # Diagonalize Hamiltonian
                E[k1,k2],v[k1,k2]=la.eigh(H[k1,k2,:,:])
                print(k1, "  ", k2, "  ", kx[k1,k2], "  ", ky[k1,k2], "  ", E[k1,k2,0], "  ", E[k1,k2,1], "  ", E[k1,k2,2], "  ", E[k1,k2,3], file=file1)
                print(kx[k1,k2], "  ", ky[k1,k2], "  ", v[k1,k2,0,0], "  ", v[k1,k2,1,0], "  ", v[k1,k2,2,0], "  ", v[k1,k2,3,0], file=file2)
                print(kx[k1,k2], "  ", ky[k1,k2], "  ", v[k1,k2,0,1], "  ", v[k1,k2,1,1], "  ", v[k1,k2,2,1], "  ", v[k1,k2,3,1], file=file3)
                print(kx[k1,k2], "  ", ky[k1,k2], "  ", v[k1,k2,0,2], "  ", v[k1,k2,1,2], "  ", v[k1,k2,2,2], "  ", v[k1,k2,3,2], file=file4)
                print(kx[k1,k2], "  ", ky[k1,k2], "  ", v[k1,k2,0,3], "  ", v[k1,k2,1,3], "  ", v[k1,k2,2,3], "  ", v[k1,k2,3,3], file=file5)
    elif flag1 == 1:
        for k1 in range(Nk):
                # Cavity plus mode
                H[k1,0,0] = Eplus[k1]
                # Cavity minus mode 
                H[k1,1,1] = Eminus[k1]
                # Cavity coupling
                H[k1,0,1] = Coupl[k1]
                H[k1,1,0] = np.conj(H[k1,0,1])
                # Molecule plus state
                H[k1,2,2] = we + g
                # Molecule minus state
                H[k1,3,3] = we - g
                # Cavity plus and molecule plus coupling
                H[k1,2,0] = JpMup[k1]*np.sqrt(1-fm-2*fp)
                H[k1,0,2] = np.conj(H[k1,2,0])
                # Cavity plus and molecule minus coupling
                H[k1,3,0] = JpMum[k1]*np.sqrt(1-fp-2*fm)
                H[k1,0,3] = np.conj(H[k1,3,0])
                # Cavity minus and molecule plus coupling
                H[k1,2,1] = JmMup[k1]*np.sqrt(1-fm-2*fp)
                H[k1,1,2] = np.conj(H[k1,2,1])
                # Cavity minus and molecule minus coupling
                H[k1,3,1] = JmMum[k1]*np.sqrt(1-fp-2*fm)
                H[k1,1,3] = np.conj(H[k1,3,1])
                
                # Diagonalize Hamiltonian
                E[k1],v[k1]=la.eigh(H[k1,:,:])
                
                # Polarization of bands (cavity component only)
                for idx in range(4):
                    bplus2 = (v[k1,0,idx]*np.conj(v[k1,0,idx])).real
                    bminus2 = (v[k1,1,idx]*np.conj(v[k1,1,idx])).real
                    Pol[k1,idx] = (bplus2-bminus2)/(bplus2+bminus2)

                print(k1, "  ", k[k1], "  ", E[k1,0], "  ", E[k1,1],  "  ", E[k1,2], "  ", E[k1,3], "  ", Pol[k1,0], "  ", Pol[k1,1], "  ", Pol[k1,2], "  ", Pol[k1,3], file=file1)
                print(k[k1], "  ", v[k1,0,0], "  ", v[k1,1,0], "  ", v[k1,2,0], "  ", v[k1,3,0], file=file2)
                print(k[k1], "  ", v[k1,0,1], "  ", v[k1,1,1], "  ", v[k1,2,1], "  ", v[k1,3,1], file=file3)
                print(k[k1], "  ", v[k1,0,2], "  ", v[k1,1,2], "  ", v[k1,2,2], "  ", v[k1,3,2], file=file4)
                print(k[k1], "  ", v[k1,0,3], "  ", v[k1,1,3], "  ", v[k1,2,3], "  ", v[k1,3,3], file=file5)

    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    return (kx,ky,H,E,v)

