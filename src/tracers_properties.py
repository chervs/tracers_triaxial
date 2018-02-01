import numpy as np
import matplotlib.pyplot as plt
import sys

from pygadgetreader import readsnap
import weights



def radial_bins(r, nbins):
    r_hist, r_edges = np.histogram(r, nbins)
    dr = r_edges[1]-r_edges[0]
    r_edges += dr/2
    return r_hist, r_edges[:-1]

def stellar_quantity(w, pids, wids, phys_quantitiy):
    """

    """

    s=np.argsort(wids)
    weights=w[s]

    in_ids = index_ids(pids, wids)

    phys_quantitiy_ids = phys_quantitiy[in_ids]
    phys_quantitiy_w = phys_quantitiy_ids[s]


    return phys_quantitiy_w, wids[s]


def den_tracers(w, pids, wids, r, mass, nbins):

    #r = r[1:]
    #mass = mass[1:]


    s=np.argsort(wids)
    weights=w[s]


    mass_w = mass[s]*weights
    r_ids = r[s]

    r_hist, r_edges = radial_bins(r_ids, nbins)

    mass_bins=np.ndarray(shape=np.size(r_hist), dtype=float)


    for i in range(0,np.size(r_hist)-1):
        wbin=np.where((r_ids>=r_edges[i]) & (r_ids<r_edges[i+1]))

        if(np.size(wbin)>0):
            mass_bins[i]=np.sum(mass_w[wbin]) #reverse indices in IDL is much faster than this junk
        else:
            print('Warning: no particles shown in bin range [{:.2f}-{:.2f}] Kpc'.format(i, i+1))
            mass_bins[i]=0.0

    shells = np.ndarray(shape=np.size(r_hist), dtype=float)

    for i in range(0, np.size(r_edges)-1):
        shells[i]=4/3.*np.pi*(r_edges[i+1]**3-r_edges[i]**3)

    density = mass_bins/shells

    return density
