import numpy as np

def CavPlus(E0, m, zeta, kx, ky):
    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)
    
    # kcosphi = kx 
    Eplus = E0 + k2/(2*m) + zeta*kx
    return Eplus

def CavMinus(E0, m, zeta, kx, ky):
    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)
    
    # kcosphi = kx 
    Eminus = E0 + k2/(2*m) - zeta*kx
    return Eminus

def CavCoupl(beta0, beta, kx, ky):
    # k^2=kx^2+ky^2
    # Coupling of the + and - cavity modes

    k2 = np.square(kx)+np.square(ky)
    k = np.sqrt(k2)

    Coupl = -beta0 + beta*(np.square(kx) - np.square(ky) - 1j*2*np.multiply(kx,ky))
    return Coupl
    
def CavTETM(E0, m, beta0, beta, kx, ky):
    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)
    k = np.sqrt(k2)

    cos2phi = np.square(np.divide(kx,k))-np.square(np.divide(ky,k))

    TM = E0 + k2/(2*m) + beta*k2 - beta0*cos2phi
    TE = E0 + k2/(2*m) - beta*k2 + beta0*cos2phi

    return (TM,TE)

def CavHV(E0, m, beta0, beta, kx, ky):
    # k^2=kx^2+ky^2
    k2 = np.square(kx)+np.square(ky)

    H = E0 + k2/(2*m) + beta*(np.square(kx)-np.square(ky)) - beta0
    V = E0 + k2/(2*m) - beta*(np.square(kx)-np.square(ky)) + beta0

    return (H,V)

def Stokes(Nk,kx,ky,v,idx,fnameS):
   # Computes the Stokes parameters S1, S2 and S3. v is in the +,- basis

   file1 = open(fnameS, 'w')

   S1 = np.zeros((Nk,Nk),dtype=np.float_)
   S2 = np.zeros((Nk,Nk),dtype=np.float_)
   S3 = np.zeros((Nk,Nk),dtype=np.float_)
   for k1 in range(Nk):
      for k2 in range(Nk):
         # basis change from +,- to H,V
         H = v[k1,k2,0,idx]/np.sqrt(2) + v[k1,k2,1,idx]/np.sqrt(2)
         V = 1j*v[k1,k2,0,idx]/np.sqrt(2) -1j*v[k1,k2,1,idx]/np.sqrt(2)

         # basis change from H,V to D1,D2 (linear polarization rotated by 45degrees from H,V basis)
         D1 = (H + V)/np.sqrt(2)
         D2 = (H - V)/np.sqrt(2)

         # Stokes parameters
         S1[k1,k2] = H*np.conj(H) - V*np.conj(V)
         S2[k1,k2] = D1*np.conj(D1) - D2*np.conj(D2)
         S3[k1,k2] = v[k1,k2,0,idx]*np.conj(v[k1,k2,0,idx]) - v[k1,k2,1,idx]*np.conj(v[k1,k2,1,idx])

         # Saving Stokes parameters
         print(k1, "  ", k2, "  ", kx[k1,k2], "  ", ky[k1,k2], "  ", S1[k1,k2], "  ", S2[k1,k2], "  ", S3[k1,k2], file=file1)

   file1.close()
   return(S1,S2,S3)

