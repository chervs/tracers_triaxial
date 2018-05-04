import numpy as np
#import reading_snapshots

from sklearn.neighbors import NearestNeighbors
from astropy.coordinates import SkyCoord
from astropy import units as u


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
       rho_bins[i-1] = np.sum(mass[index]) / V

    return r_bins+dr, rho_bins

def pos_cartesian_to_galactic(pos, vel):
    """
    Make mock observations in galactic coordinates.
    uses Astropy SkyCoord module.
    l and b coordinates are computed from cartesian coordinates.
    l : [-180, 180]
    b : [-90, 90]

    Parameters:
    -----------
    pos : 3d-numpy array.
    vel : 3d-numpy array.
    lmin : float.
           Minimum latitute of the observation in degrees.
    lmax : float.
           Maximum latitute of the observation in degrees.
    bmin : float.
           Minimum longitude of the observation in degrees.
    bmax : float.
           Maximum longitude of the observation in degrees.
    rmax : float.
           Maximum radius to compute the anisotropy and dispersion
           profiles.
    n_bins_r : int
           Number of bins to do the radial measurements.

    """
    ## transforming to galactic coordinates.

    c_gal = SkyCoord(pos, representation='cartesian',frame='galactic')
    c_gal.representation = 'spherical'

    ## to degrees and range of l.

    l_degrees = c_gal.l.wrap_at(180 * u.deg).radian
    b_degrees = c_gal.b.radian

    ## Selecting the region of observation.


    return l_degrees, b_degrees
    
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
    #print('Computing velocity dispersion inside a radial bin!')

    vr, v_theta, v_phi = vel_cartesian_to_spherical(pos, vel)

    vr1 = np.zeros(len(vr))
    vtheta1 = np.zeros(len(vr))
    vphi1 = np.zeros(len(vr))

    #print('The number of particles is', len(vr1))

    vr_mean = np.mean(vr)
    vtheta_mean = np.mean(v_theta)
    vphi_mean = np.mean(v_phi)

    #or i in range(len(vr)):
    vr1 = weights*(vr-vr_mean)**2
    vtheta1 = weights*(v_theta-vtheta_mean)**2
    vphi1 = weights*(v_phi-vphi_mean)**2
    W = np.sum(weights)
    #print('Sum of weights', W)
    N = len(weights)
    sigma_r = np.sqrt(np.sum(vr1)/(W))
    sigma_theta = np.sqrt(np.sum(vtheta1)/(W))
    sigma_phi = np.sqrt(np.sum(vphi1)/(W))

    return sigma_r, sigma_theta, sigma_phi


def velocity_dispersions_r(pos, vel, n_bins, rmax, weights, weighted):
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
    print('rmax', rmax)
    dr = np.linspace(0, rmax, n_bins)
    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    #r = pos
    vr_disp_r = np.zeros(len(dr)-1)
    vtheta_disp_r = np.zeros(len(dr)-1)
    vphi_disp_r = np.zeros(len(dr)-1)
    #w_rbins = np.zeros(len(dr)-1)
    #v_mean_r = np.zeros(len(dr)-1)
    #v_mean_w = np.zeros(len(dr)-1)

    if weighted==1:
        print('Computing the velocity dispersion profile for the stellar halo!')
        for i in range(len(dr)-1):
            index = np.where((r<dr[i+1]) & (r>dr[i]))[0]
            #v_mean_r[i] = np.mean(np.sqrt(vel[index,0]**2 + vel[index,1]**2 + vel[index,2]**2))
            #v_mean_w[i] = sum(weights[index]*(np.sqrt(vel[index,0]**2 + vel[index,1]**2 + vel[index,2]**2)))/sum(weights[index])
            vr_disp_r[i], vtheta_disp_r[i], vphi_disp_r[i]\
             = velocity_dispersion_weights(pos[index], vel[index]\
                                           ,weights[index])
    elif weighted==0:
        for i in range(len(dr)-1):
            index = np.where((r<dr[i+1]) & (r>dr[i]))
            vr_disp_r[i], vtheta_disp_r[i], vphi_disp_r[i] = velocity_dispersion(pos[index], vel[index])

    return dr, vr_disp_r, vtheta_disp_r, vphi_disp_r

def velocity_dispersions_octants(pos, vel, nbins, rmax, weights, weighted):

    """
    Computes the velocity dispersion in eight octants in the sky,
    defined in galactic coordinates as follows:

    octant 1 :
    octant 2 :
    octant 3 :
    octant 4 :
    octant 5 :
    octant 6 :
    octant 7 :
    octant 8 :

    Parameters:
    -----------

    Output:
    -------



    """

    ## Making the octants cuts:

    d_b_rads = np.linspace(-np.pi/2., np.pi/2., 5)
    d_l_rads = np.linspace(-np.pi, np.pi, 3)
    #r_bins = np.linspace(0, 300, 31)

    ## Arrays to store the velocity dispersion profiles
    vr_octants = np.zeros((nbins-1, 8))
    v_theta_octants = np.zeros((nbins-1, 8))
    v_phi_octants = np.zeros((nbins-1, 8))

    ## Octants counter, k=0 is for the radial bins!
    k = 0

    l, b = pos_cartesian_to_galactic(pos, vel)

    for i in range(len(d_l_rads)-1):
        for j in range(len(d_b_rads)-1):
            index = np.where((l<d_l_rads[i+1]) & (l>d_l_rads[i]) &\
                             (b>d_b_rads[j]) & (b<d_b_rads[j+1]))

            if weighted==0:
                dr, vr_octants[:,k], v_theta_octants[:,k], v_phi_octants[:,k] \
                = velocity_dispersions_r(pos[index], vel[index], nbins, \
                                         rmax, weights[index], 0)
            elif weighted==1:
                print('HERE in OCTANTS weighted')
                dr, vr_octants[:,k], v_theta_octants[:,k], v_phi_octants[:,k] \
                = velocity_dispersions_r(pos[index], vel[index], nbins, rmax,\
                                         weights[index], 1)

            k+=1

    return dr, vr_octants, v_theta_octants, v_phi_octants




def sigma2d_NN(pos, vel, lbins, bbins, n_n, d_slice, weights, relative=False):
    """
    Returns a 2d histogram of the anisotropy parameter in galactic coordinates.

    Parameters:
    ----------
    pos : numpy ndarray
        3d array with the cartesian positions of the particles.
    vel : numpy.ndarray
        3d array with the cartesian velocoties of the particles.
    lbins : int
        Numer of bins to do the grid in latitude.
    bbins : int
        Number of bins to do the grid in logitude.
    n_n : int
        Number of neighbors.
    d_slice : float
        galactocentric distance to make the slice cut.
    weights : numpy.array
        Array with the weights for the particles
    relative :  If True, the velocity dispersion is computed relative to the
                mean. (default = False)

    Returns:
    --------

    sigma_r_grid : numpy ndarray
        2d array with the radial velocity dispersions.
    sigma_t_grid : numoy ndarray
        2d array with the tangential velocity dispersions.
    """

    ## Defining the
    d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins)
    d_l_rads = np.linspace(-np.pi, np.pi, lbins)
    ## Defining the 2d arrays for the velocity dispersions.

    sigma_r_grid = np.zeros((lbins-1, bbins-1))
    sigma_t_grid = np.zeros((lbins-1, bbins-1))


    r = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5

    # Finding the NN.
    k = 0
    neigh = NearestNeighbors(n_neighbors=n_n, radius=1, algorithm='ball_tree')
    ngbrs = neigh.fit(pos)
    # Computing mean velocity dispersions
    if relative==True:
        index_cut =  np.where((r<(d_slice+5)) & (r>(d_slice-5)))
        sigma_r_mean, sigma_theta_mean, sigma_phi_mean =  velocity_dispersion_weights(pos[index_cut], vel[index_cut], weights[indix_cut])


    for i in range(len(d_l_rads)-1):
        for j in range(len(d_b_rads)-1):
            #print(i, j)
            gc = SkyCoord(l=d_l_rads[i]*u.radian, b=d_b_rads[j]*u.radian, frame='galactic', distance=d_slice*u.kpc)
            pos_grid = gc.cartesian.xyz.value
            # Finding the nearest neighbors.
            distances, indices = neigh.kneighbors([pos_grid])
            sigma_r, sigma_theta, sigma_phi = velocity_dispersion_weights(pos[indices[0,:]], vel[indices[0,:]], weights[indices[0,:]])
            if relative==True:
                sigma_t_grid[i][j] = ((sigma_theta**2 + sigma_phi**2))**0.5 - (sigma_phi_mean**2 + sigma_theta_mean**2)**0.5
                sigma_r_grid[i][j] = sigma_r - sigma_r_mean
            else:
                sigma_t_grid[i][j] = ((sigma_theta**2 + sigma_phi**2))**0.5
                sigma_r_grid[i][j] = sigma_r
            k+=1
    return sigma_r_grid, sigma_t_grid



"""
if __name__ == "__main__":

    weights_hern, w_ids_hern = weight_triaxial(rr, Ekk, Epp, ids, partmass, bins_w, nbins, 1, 'Plummer', [a])

    density_hern = den_tracers(weights_hern, w_ids_hern, rr, massarr, plot_bins, rcut)
    density_hern_fut = den_tracers(weights_hern, w_ids_hern, rr_fut, massarr_fut, plot_bins, rcut)
"""
