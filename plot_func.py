import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as tck
import matplotlib as mpl
import math as m
from cmath import pi
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patheffects as pe

def CavBerry2D(kx, ky, F, fname):
    # Berry curvature in k-space 2D
    minF = np.min(F)
    maxF = np.max(F)
    maxF2 = np.maximum(-minF,maxF)
    divnorm = TwoSlopeNorm(vmin=-maxF2, vcenter=0., vmax=maxF2)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel(r'$k_{x}$ ($\mu$m$^{-1}$)')
    plt.ylabel(r'$k_{y}$ ($\mu$m$^{-1}$)',labelpad=-10)
    ax.tick_params(width=1)
    plt.ylim(-10,10)
    im1 = plt.pcolormesh(kx, ky, F, cmap=cm.jet, norm=divnorm, rasterized=True)
    cbar = plt.colorbar(im1,fraction=0.0246,aspect=30,pad=0.02)
    cbar.ax.tick_params(labelsize=20,width=1)
    cbar.formatter.set_useMathText(True)
    cbar.formatter.set_powerlimits((0, 0))
    tick_locator = tck.MaxNLocator(nbins=2)
    cbar.locator = tick_locator
    cbar.ax.yaxis.get_offset_text().set(size=20)
    cbar.update_ticks()
    plt.savefig(fname,bbox_inches='tight',dpi=100)
    plt.show()

def CavPol2D(kx, ky, Pol, fname):
    # Polarization in k-space 2D
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel(r'$k_{x}$ ($\mu$m$^{-1}$)')
    plt.ylabel(r'$k_{y}$ ($\mu$m$^{-1}$)',labelpad=-10)
    ax.tick_params(width=1)
    plt.ylim(-10,10)
    im1 = plt.pcolormesh(kx, ky, Pol, cmap=cm.bwr, norm=divnorm, rasterized=True)
    cbar = plt.colorbar(im1,fraction=0.0246,aspect=30,pad=0.02)
    cbar.ax.tick_params(labelsize=20,width=1)
    cbar.ax.locator_params(nbins=3)
    plt.savefig(fname,bbox_inches='tight',dpi=100)
    plt.show()

def CavExBands1Dbetter(k, E0, E1, E2, E3, Pol0, Pol1, Pol2, Pol3, fname):
    fig = plt.figure()
    Nk = len(k)
    E = np.zeros((Nk,4),dtype=float)
    Pol = np.zeros((Nk,4),dtype=float)

    E[:,0] = E0
    E[:,1] = E1
    E[:,2] = E2
    E[:,3] = E3

    Pol[:,0] = Pol0
    Pol[:,1] = Pol1
    Pol[:,2] = Pol2
    Pol[:,3] = Pol3
    
    ax = plt.gca()
    for idx in range(2):
        plt.plot(k, E[:,idx], color = 'white', lw=4.5)
        points = np.array([k,E[:,idx]]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]], axis=1)
        norm = plt.Normalize(-1,1)
        lc = LineCollection(segments, cmap='bwr', norm=norm)
        lc.set_array(Pol[:,idx])
        lc.set_linewidth(5)
        line = ax.add_collection(lc)

    plt.ylim(3.64,3.81)
    plt.xlabel(r'$k_x$ ($\mu$m$^{-1}$)')
    plt.ylabel(r'Energy (eV)')
    plt.savefig(fname, bbox_inches='tight', dpi=100)
    plt.show()

