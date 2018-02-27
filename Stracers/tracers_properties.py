import numpy as np
import sys
sys.path.append('../../MW_anisotropy/code/')

#import reading_snapshots
from pygadgetreader import readsnap
from .weights import *

# TEMPORAL !!!!!!!!!!!!!!
def all_host_particles(xyz, vxyz, pids, pot, mass, N_host_particles):
    """
    Function that return the host and the sat particles
    positions and velocities.

    Parameters:
    -----------
    xyz: snapshot coordinates with shape (n,3)
    vxys: snapshot velocities with shape (n,3)
    pids: particles ids
    Nhost_particles: Number of host particles in the snapshot
    Returns:
    --------
    xyz_mw, vxyz_mw, xyzlmc, vxyz_lmc: coordinates and velocities of
    the host and the sat.

    """
    sort_indexes = np.sort(pids)
    N_cut = sort_indexes[N_host_particles]
    host_ids = np.where(pids<N_cut)[0]
    return xyz[host_ids], vxyz[host_ids], pids[host_ids], pot[host_ids], mass[host_ids]

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


def Stellar_properties(ids_weights, weights, ids, mass, pos, vel):
    """
    Paramters:
    ----------

    ids_weights :

    weights :

    ids :

    mass :

    pos :

    vel :

    """
    assert len(ids_weights)<= len(ids), 'Error: length of weights ids is larger than lenght of ids!'

    common_ids = np.isin(ids, ids_weights)
    # selecting elements that are common in weight_ids and ids:
    ids_n = ids[common_ids]
    mass_n = mass[common_ids]
    pos_n = pos[common_ids]
    vel_n = vel[common_ids]

    assert len(ids_n) == len(ids_weights), 'Same number of particles'
    # Sorting ids:
    ids_w_arg_sort = np.argsort(ids_weights)
    ids_arg_sort = np.argsort(ids_n)


    comp_items = ids_weights[ids_w_arg_sort]==ids_n[ids_arg_sort]
    false_ind = np.where(comp_items==False)[0]
    assert (len(false_ind)==0), 'Error: Hey!'

    # Sorting ids and properties
    weights_n = weights[ids_w_arg_sort]
    ids_weights_n = ids_weights[ids_w_arg_sort]

    # Sorting future properties
    ids_n = ids_n[ids_arg_sort]
    mass_n = mass_n[ids_arg_sort]*weights_n
    posx = pos_n[ids_arg_sort,0]
    posy = pos_n[ids_arg_sort,1]
    posz = pos_n[ids_arg_sort,2]

    velx = vel_n[ids_arg_sort,0]*weights_n
    vely = vel_n[ids_arg_sort,1]*weights_n
    velz = vel_n[ids_arg_sort,2]*weights_n

    all_pos = np.array([posx, posy, posz]).T
    all_vel = np.array([velx, vely, velz]).T

    return [ids_weights_n, weights_n, ids_n], mass_n, all_pos, all_vel

def energies(snap, N_host_particles, rcut=0, lmc=0):
    """
    Paramters:
    ----------

    snap : string
        path and name of the snapshot

    rcut : int
        truncation radii (no trunction by default rcut=0)


    Returns:
    --------
    Distances (rr)
    Kinetic energy (Ekk)
    Potential (MW_pot)
    Ids (MW_ids)
    Mass (MW_mass)
    Pos (MW_pos)
    Vel (MW_vel)

    """

    pp= readsnap(snap, 'pos', 'dm')
    vv= readsnap(snap, 'vel', 'dm')
    massarr= readsnap(snap, 'mass', 'dm')
    Epp = readsnap(snap, 'pot', 'dm')
    ids = readsnap(snap, 'pid', 'dm')


    # Selecting MW particles
    #N_host_particles = 100000000
    if lmc ==1 :
        MW_pos, MW_vel, MW_ids, MW_pot, MW_mass = all_host_particles(pp, vv, ids, Epp, massarr, N_host_particles)
    elif lmc==0:
        MW_pos, MW_vel, MW_ids, MW_pot, MW_mass = pp, vv, ids, Epp, massarr

    assert len(MW_ids)==N_host_particles, 'Error: something went wrong selecting the host particles'

    rr=np.sqrt(MW_pos[:,0]**2+MW_pos[:,1]**2+MW_pos[:,2]**2)


    if rcut>0:
        r_cut = np.where((rr<rcut))[0]

        rr = rr[r_cut]
        MW_pos = MW_pos[r_cut]
        MW_vel = MW_vel[r_cut]
        MW_mass = MW_mass[r_cut]
        MW_pot = MW_pot[r_cut]
        MW_ids = MW_ids[r_cut]

    v2=MW_vel[:,0]**2+MW_vel[:,1]**2+MW_vel[:,2]**2
    Ekk=0.5*v2

    return rr, Ekk, MW_pot, MW_ids, MW_mass, MW_pos, MW_vel


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
