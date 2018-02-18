import numpy as np
import sys

from pygadgetreader import readsnap
from .weights import *



def radial_bins(r, nbins):
    r_hist, r_edges = np.histogram(r, nbins)
    dr = r_edges[1]-r_edges[0]
    r_edges += dr/2
    return r_hist, r_edges[:-1]


def finding_indices(ids_w, ids):
    """
    Return the indicies in which the ids of the two snapshots are the same.
    """

    indices = np.zeros(len(ids_w))
    for i in range(len(indices)):
        indices[i] = (np.where(ids=ids_w[i])[0])

    return indices


def stellar_properties(w_ids, weights, ids, mw_pos, mw_vel, massarr):
    """
    Returns the stellar particles properties weighted.

    Parameters:
    ----------

    w_ids : numpy.array
            array with ids of the weights
    weights : numpy.array
            array with ids of the weights
    ids : numpy.array
            array with ids of the weights
    mw_pos : numpy.array
            array with ids of the weights
    mw_vel : numpy.array
            array with ids of the weights
    massarr : numpy.array
            array with ids of the weights
    """
    # finds the indicies where ids_init are in w_ids_h
    indices_ids_init = finding_indices(w_ids_h, ids_init)
    ids_init_w = ids_init[indices_ids_init]
    pos_init_w = mw_pos[indices_ids_init]
    vel_init_w = mw_vel[indices_ids_init]*weights
    massarr_init_w = massarr_init[indices_ids_init]*weights

    return ids_init_w, pos_init_w, vel_init_w, massarr_init_w

def stellar_quantity(weights, pids, wids, pq):
    """
    Assign the stellar particle weights to a given physical
    quantity (pq).

    Parameters:
    ----------

    Returns:
    --------


    """

    assert len(pids)==len(wids) , 'Length of ids arrays should be the same'
    assert len(weights)==len(wids) , 'Length of ids arrays should be the same'


    sort_index_w=np.argsort(wids)
    weights_sorted=weights[sort_index_w]

    sort_index_ids = np.argsort(pids)
    pq_sort = pq[sort_index_ids]

    phys_quantitiy_w = pq_sort * weights_sorted


    return phys_quantitiy_w, wids[sort_index_w]




def future_quantity(pos, pids,  wids, pq):
    """
    Assign the stellar particle weights to a given physical
    quantity (pq).

    Parameters:
    ----------

    Returns:
    --------


    """

    assert len(pids)==len(wids) , 'Length of ids arrays should be the same'
    assert len(weights)==len(wids) , 'Length of ids arrays should be the same'


    sort_index_fut=np.argsort(pids)
    pos_sorted_fut=pos[sort_index_fut]

    sort_index_wids = np.argsort(wids)
    pq_sort_wids = pq[sort_index_wids]

    return pos_sorted_fut, pq_sort_wids


def den_tracers(w, wids, r, mass, nbins, rcut):
    """

    Paramters:
    -----------

    w : ndarray
        Weights of the DM particles
    wids: ndarray
        Array with the ids of the weights particles, the firs id is missing (not yet clear why, but is in weight_triaxial function)
    r : ndarray
        Array with the distances to the particles

    mass : ndarray
    nbins : int
    rcut : float




    To-do:
    1. Implement the mass!
    2. Check the mass.
    3. organize this function.
    4. Implement assert conditions.
    """



    r = r[1:] # remove the first index to even len(ids)=len(wids)
    #mass = mass[1:]


    s=np.argsort(wids)
    weights=w[s]
    r_ids = r[s]

    w=np.where(ids>1) ## account for the fact there are a bounch of weights=zeros
    r_ids = r_ids[s]


    #mass_w = mass[s]*weights

    r_edges = np.linspace(1, rcut, nbins)

    mass_bins=np.ndarray(shape=np.size(r_edges)-1, dtype=float)


    for i in range(0,len(r_edges)-1):
        wbin=np.where((r_ids>=r_edges[i]) & (r_ids<r_edges[i+1]))

        if(np.size(wbin)>0):
            mass_bins[i]=np.sum(weights[wbin]) # sums the weights at each radial bin
        else:
            print('Warning: no particles shown in bin range [{:.2f}-{:.2f}] Kpc'.format(i, i+1))
            mass_bins[i]=0.0

    shells = np.ndarray(shape=np.size(r_edges), dtype=float)
    #rint(mass_bins)
    for i in range(0, np.size(r_edges)-1):
        shells[i]=4/3.*np.pi*(r_edges[i+1]**3-r_edges[i]**3)

    density = (mass_bins)/shells[:-1]

    return density



if __name__ == "__main__":

    weights_hern, w_ids_hern = weight_triaxial(rr, Ekk, Epp, ids, partmass, bins_w, nbins, 1, 'Plummer', [a])

    density_hern = den_tracers(weights_hern, w_ids_hern, rr, massarr, plot_bins, rcut)
    density_hern_fut = den_tracers(weights_hern, w_ids_hern, rr_fut, massarr_fut, plot_bins, rcut)



