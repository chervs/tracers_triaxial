import numpy as np
import sys
sys.path.append('../../MW_anisotropy/code/')

#import reading_snapshots
from pygadgetreader import readsnap
from .weights import *




def den_profile(r, mass, rbins, rcut):
    """

    Computes the density profiles of a halo.

    Paramters:
    -----------

    r : numpy.array
        Array with the distances to the particles

    mass : numpy.array
        Masses of the particles.
    nbins : int
        Number of radial bins to use.
    rcut : Float
        Radius at which truncate the halo.




    To-do:
    ------

    """



    r_bins = np.linspace(0, 100, rbins)
    dr = (r_bins[1] - r_bins[0])/2.0
    rho_bins = np.zeros(rbins-1)
    for i in range(1, len(r_bins)):
       index = np.where((r<r_bins[i]) & (r>=r_bins[i-1]))[0]
       V = 4/3. * np.pi * (r_bins[i]**3-r_bins[i-1]**3)
       rho_bins[i-1] = np.sum(m[index]) / V

    return r_bins+dr, density


def vel_cartesian_to_spherical(pos, vel):
    """
    Computes velocities in spherical coordinates from cartesian.

    Parameters:
    -----------

    pos : numpy.array
        3-D Cartesian positions of the particles.
    vel : numpy.array
        3-D Cartesian velocities of the particles.

    Returns:
    --------

    vr :
    vthehta :
    vphi :

    """
    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    theta = np.arccos(pos[:,2]/r)
    phi = np.arctan2(pos[:,1], pos[:,0])

    vr = np.sin(theta)*np.cos(phi)*vel[:,0] \
         +  np.sin(theta)*np.sin(phi)*vel[:,1] \
         +  np.cos(theta)*vel[:,2]

    v_theta = np.cos(theta)*np.cos(phi)*vel[:,0]\
              + np.cos(theta)*np.sin(phi)*vel[:,1]\
              - np.sin(theta)*vel[:,2]

    v_phi = -np.sin(phi)*vel[:,0] + np.cos(phi)*vel[:,1]

    return vr, v_theta, v_phi

def velocity_dispersion(pos, vel):
    """
    Computes the velocity dispersions in spherical coordinates.


    Parameters:
    ----------
    pos : 3d numpy array
        Array with the cartesian coordinates of the particles
    vel : 3d numpy array
        Array with the cartesian velocities of the particles

    Returns:
    --------
    sigma_r : float
        The value of sigma_r.
    sigma_theta : float
        The value of sigma_theta
    sigma_phi : float
        The value of sigma_phi

    """

    vr, v_theta, v_phi = vel_cartesian_to_spherical(pos, vel)
    sigma_r = np.std(vr)
    sigma_theta = np.std(v_theta)
    sigma_phi = np.std(v_phi)

    return sigma_r, sigma_theta, sigma_phi

def velocity_dispersion_weights(pos, vel, weights):
    """
    Computes the velocity dispersions for stellar particles using
    the weights fro DM particles.

    Uses Eq: (3) in Laporte 13a to compute the velocity dispersion this is:

    \sigma_* =  \dfrac{\sum_{i}^N \omega_i (v_i - \bar{v_i})}{\sum_i^N \omega_i}

    N = number of particles, \omega_i the weights.


    Parameters:
    ----------
    pos : 3d numpy array
        Array with the cartesian coordinates of the particles
    vel : 3d numpy array
        Array with the cartesian velocities of the particles

    Returns:
    --------
    sigma_r : float
        The value of sigma_r.
    sigma_theta : float
        The value of sigma_theta
    sigma_phi : float
        The value of sigma_phi

    """

    vr, v_theta, v_phi = vel_cartesian_to_spherical(pos, vel)

    vr1 = np.zeros(len(vr))
    vtheta1 = np.zeros(len(vr))
    vphi1 = np.zeros(len(vr))

    for i in range(len(vr)):
        vr1[i] = weights[i]*(vr[i]-np.mean(vr))**2
        vtheta1[i] = weights[i]*(v_theta[i]-np.mean(v_theta))**2
        vphi1[i] = weights[i]*(v_phi[i]-np.mean(v_phi))**2

    sigma_r = np.sqrt(np.abs(np.sum(vr1))/np.sum(weights))
    sigma_theta = np.sqrt(np.sum(vtheta1)/np.sum(weights))
    sigma_phi = np.sqrt(np.sum(vphi1)/np.sum(weights))

    return sigma_r, sigma_theta, sigma_phi


def velocity_dispersions_r(pos, vel, n_bins, rmax, weights, weighted=0):
    """
    Compute the velocity dispersions in radial bins.i

    Parameters:
    ----------
    pos : 3d numpy array
        Array with the cartesian coordinates of the particles
    vel : 3d numpy array
        Array with the cartesian velocities of the particles
    n_bins : int
        Number of radial bins to compute the velocity dispersions.
    rmax : int
        Maximum radius to compute the velocity dispersions.

    Returns:
    --------
    sigma_r : numpy array
    sigma_theta : numpy array
    sigma_phi : numpy array

    """
    dr = np.linspace(0, rmax, n_bins)
    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    #r = pos
    vr_disp_r = np.zeros(len(dr)-1)
    vtheta_disp_r = np.zeros(len(dr)-1)
    vphi_disp_r = np.zeros(len(dr)-1)

    if weighted==1:
        print('Computing the velocity dispersion profile for the stellar halo!')
        for i in range(len(dr)-1):
            index = np.where((r<dr[i+1]) & (r>dr[i]))
            vr_disp_r[i], vtheta_disp_r[i], vphi_disp_r[i] = velocity_dispersion_weights(pos[index], vel[index]\
                                                                                        , weights[index])

    else:
        for i in range(len(dr)-1):
            index = np.where((r<dr[i+1]) & (r>dr[i]))
            vr_disp_r[i], vtheta_disp_r[i], vphi_disp_r[i] = velocity_dispersion(pos[index], vel[index])

    return vr_disp_r, vtheta_disp_r, vphi_disp_r

if __name__ == "__main__":

    weights_hern, w_ids_hern = weight_triaxial(rr, Ekk, Epp, ids, partmass, bins_w, nbins, 1, 'Plummer', [a])

    density_hern = den_tracers(weights_hern, w_ids_hern, rr, massarr, plot_bins, rcut)
    density_hern_fut = den_tracers(weights_hern, w_ids_hern, rr_fut, massarr_fut, plot_bins, rcut)
