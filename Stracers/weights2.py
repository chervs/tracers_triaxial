import numpy as np
import sys
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from pygadgetreader import readsnap
from tracers_dens import *
from smooth import savitzky_golay

"""
To-do:

1. Organize weight_triaxial function
2. How to properly bin the energy?, how many bins?
3.
"""

def rho_tracers(r, M, profile, profile_params):
    """
    Density profiles for the
    to-do:

    1. pass as an argument the density function instead of the if statements.
    2. profile paramas as *profile_params
    """
    if profile == 'Plummer':
        rho = dens_plummer(r, M, profile_params[0])
    elif profile == 'Hernquist':
        rho = dens_hernquist(r, M, profile_params[0])
    elif profile == 'NFW':
        rho = dens_NFWnRvir(r, M, profile_params[0], profile_params[1])
    elif profile == 'Einasto':
        rho = dens_Einasto(r, M, profile_params[0], profile_params[1])

    return rho

def densities_derivatives(rbins, psi_bins, interp_bins=100, profile='Hernquist'):
    """
    Computes the derivatives of 
    
    rbins : number of radial bins.
    
    psi_bins : psi binned.
    
    inter_bins : int
        Values of the interpolated 
    
    
    """
    spl1 = InterpolatedUnivariateSpline(rbins, psi_bins)

    # interpolating in the radial bins.  
    rbins_hr = np.linspace(min(rbins), max(rbins), interp_bins)
    #nu_tracer_hr = spl1(rbins_hr)
    nu_tracer=rho_tracers(rbins_hr, 1, profile, [10])
    psi_hr = spl1(rbins_hr)


    # First derivative.
    dnu_dpsi = np.gradient(nu_tracer, psi_hr)
    #spl3 = interp1d(rbins, dnu_dpsi, kind='cubic')
    #dnu_dpsi_hr = spl3(rbins_hr)

    # second derivative
    #dnu2_dpsi2 = np.gradient(dnu_dpsi, psi2)
    
    # smoothing first derivative
    #dnu_dpsi_smooth = savitzky_golay(dnu_dpsi, 5, 3)
    dnu2_dpsi2 = np.gradient(dnu_dpsi, psi_hr)
    # smoothing second derivative
    #dnu2_dpsi2_smooth = savitzky_golay(dnu2_dpsi2, 5, 3)

    return rbins_hr, nu_tracer, psi_hr, dnu_dpsi, dnu2_dpsi2


#def interpolate_energies()

def energies(r_part, Ep, v, n_rbins):
    r"""
    
    Parameters:
    -----------
    
    r_part: 'numpy.array`
        Galactocentric distances of the DM particles in [$kpc$].
    pmass: float
        Mass of a DM particle in units of solar masses.
    Ep : 'numpy.array`
        potential energy in units of
    v : 'numpy.array`
        velocities in km/s
    n_rbins : int
        number of bins, this will be used to bin the potential energy.
    
    Returns:
    --------
    
    $\Psi$ : $\Psi = -\Phi + \Phi_0$ Relative potential. 
    $\epsilon$ : $\epsilon = -E + \phi_0$ relative total energy.
    
    
    """
    
    G = 4.30071e-6 # Gadget units
    nbins_coarsed = 20
    E_k = 0.5*v**2 # Kinetic energy / mass.
    E = E_k + Ep
    epsilon = (-1.0)*E

    # Binning the data in logarithmic bins in radius
    # Should I need to worried about centering the bin?
    # yes if you are going to interpolate.
    
    rbins = np.logspace(np.min(np.log10(r_part)), np.max(np.log10(r_part)), nbins_coarsed)
    
    # spaced between rbins
    dr = np.zeros(len(rbins)-1)
    for i in range(len(rbins)-1):
        dr[i] = rbins[i+1] - rbins[i]
    
    pot = np.zeros(nbins_coarsed-1)
    
    for i in range(len(rbins)-1):
        index_bins = np.where((r_part<rbins[i+1])
                              & (r_part>=rbins[i]))[0]
        
        if len(index_bins) == 0:
            pot[i] = 0
            print('Warning : No particles found at r={:0>2f} kpc'.format(rbins[i]))
        else:
            pot[i] = np.mean(Ep[index_bins])
    
                   
    f_interp_pot = interp1d(rbins[:-1]+dr, pot, kind='cubic')
    r_interp = np.linspace(rbins[0]+dr[0], rbins[-2]+dr[-2], n_rbins)
    pot_interp = f_interp_pot(r_interp)
    psi = (-1.0)*pot_interp



    #Binning Energy for g(E) and f(E) (f(epsilon)) calculations
    Histo_E, Edges = np.histogram(E, bins=N_Eb)
    Histo_epsilon, epsdges = np.histogram(epsilon, bins=N_Eb)

    # Are these always positive?
    dE = Edges[1]-Edges[0]
    depsilon = epsedges[1]-epsedges[0]    

    Edges = Edges + dE/2.
    epsedges = epsedges + dE/2.


    return psi, epsilon, Histo_E, dE

def NE(E, dE):
    """
    Computes the differential energy distribution.

    to-do:
    check the units of the energy!

    Input:
    -----

    E : numpy.array
        histrogram of the energies.
    dE : 
        energy difference between states.

    """

    N_E = E/dE

    return N_E

def weight_triaxial(r, Ek, Ep, partID, m, bsize, N_Eb, stellar_mass, profile, profile_params):
    """
    N_Eb : number of bins for the Energy



    """


    #Fetching derivatives from the data necessary for the Eddington formula evalution
    # D

    
    nu_tracer_hr, psi2_hr, dnu_dpsi, dnu2_dpsi2 = densities_derivatives(rbins, psi2, interp_bins=1000, profile='Hernquist')
    
    #return rbins, psi2_hr, nu_tracer_hr, dnu_dpsi, dnu2_dpsi2

    

    #Total N(E) differential energy distribution
    #Histo_M=Histo_E*m/np.sqrt((Ebins[2]-Ebins[1])**2)

    # EDDINGTON FORMULA --------------
    dpsi=np.ndarray(shape=np.size(psi2), dtype=float)
    for i in range (1, np.size(dpsi)):
        dpsi[i]=psi2[i]-psi2[i-1]


    distribution_function=np.ndarray(shape=np.size(epsilon_bins), dtype=float)
    for i in range(0,np.size(epsilon_bins)):
        eps=epsilon_bins[i]
        w=np.where(psi2<eps)[0]

        if (np.size(w)!=0):
            #w=np.array(w)
            tot1=dpsi[w]
            tot2=dnu2_dpsi2[w]
            tot3=np.sqrt(2.0*(eps-psi2[w]))
            tot=tot1*tot2/tot3
            val=(1.0)/(np.sqrt(8.0)*np.pi**2)*np.sum(tot) #Arthur's eval as Sum (in sims no divergence due to res)
            #print val, i, "val, i"
            distribution_function[i]=val
        else:
            distribution_function[i]=0

    #return dnu2_dpsi2, dpsi, psi2, epsilon_bins, distribution_function

    #DENSITY OF STATES--------------
    wrme=np.ndarray(shape=np.size(Ebins), dtype=int)
    rme=np.ndarray(shape=np.size(Ebins), dtype=float)

    # Nico: commented this to avoid some values of pot2>Ebins since
    # taking the max(wpot_equals_E) don't garantee to avoid them.
    #for i in range(0, np.size(Ebins)):
    #    wpot_equals_E=np.where(pot2<=Ebins[i])[0]
    #    if (len(wpot_equals_E)!=0):
    #        wrme[i]=np.max(wpot_equals_E)
    #    else:
    #        wrme[i]=0
    #
    density_of_states=np.ndarray(shape=np.size(Ebins), dtype=float) # density of states integral (evaluated as sum)
    for i in range(0,np.size(Ebins)):
        wpot_equals_E=np.where(pot2<=Ebins[i])[0]
        if (len(wpot_equals_E)==0):
            g1=0.0
        else:
            g1=rbins[wpot_equals_E]**2
            #g2=np.sqrt(2.0*(Ebins[i]-pot2[0:wrme[i]]))
            g2=np.sqrt(2.0*(Ebins[i]-pot2[wpot_equals_E]))
            #density_of_states[i]=(4.0*np.pi)**2*np.sum(binsize_r[0:wrme[i]]*g1*g2)
            density_of_states[i]=(4.0*np.pi)**2*np.sum(binsize_r[wpot_equals_E]*g1*g2)


    indsort=np.argsort(distribution_function) #sorted indices
    indsort=indsort[::-1] #reverse
    # weights= D.F(tracers)/ (D.F.(self-consistent)) - self-consistent D.F. f(E) generates the potential Phi
    # N(E)=f(E)*g(E)
    Weights=distribution_function[indsort[::-1]]/((Histo_M)/density_of_states)

    # cast the weights to every particle
    Weights_array=np.ndarray(shape=np.size(r), dtype=float)

    for j in range(0, np.size(Edges)-1):
        wbin=np.where((E>=Edges[j]) & (E<Edges[j+1]))[0]
        if(np.size(wbin)!=0):
            Weights_array[wbin]=Weights[j]
    #Ensure that the sum of the weights = mass of the tracers - this is not strictly needed

    X=stellar_mass/(np.sum(Weights_array)*m)
    Weights_array=Weights_array*X

    #print(np.size(Weights_array))
    #return the IDS from which the weights are associated to the particles
    #needed for tracking where the tracers end up in subsequent snapshots
    # Each particle gets a weight.
    assert len(Weights_array) == len(r), 'Error: number of weights different to the number of particles'
    return Weights_array, partID
    
    
if __name__ == "__main__":
    snapshot = sys.argv[1]


    pp = readsnap(snapshot, 'pos', 'dm')
    vv = readsnap(snapshot, 'vel', 'dm')
    mass = readsnap(snapshot, 'mass', 'dm')
    Epp = readsnap(snapshot, 'pot', 'dm')
    ids = readsnap(snapshot, 'pid', 'dm')


    rr = np.sqrt(pp[:,0]**2+pp[:,1]**2+pp[:,2]**2)

    # truncating the halo

    r_cut = index = np.where(rr<100)[0]

    pp = pp[r_cut]
    rr = rr[r_cut]
    vv = vv[r_cut]
    mass = mass[r_cut]
    Epp = Epp[r_cut]
    ids = ids[r_cut]

    # Energies

    partmass=mass[3]*1e10 #generated the halo particles as "bulge"-type in Gadget file
    v2=vv[:,0]**2+vv[:,1]**2+vv[:,2]**2
    Ekk=0.5*v2
    #weight, p_ids = weight_triaxial(rr,Ekk,Epp,ids,partmass,0.01,100,1, profile)

    rbins, nu_tracer, psi2, dnu_dpsi, dnu2_dpsi2, rbins_hr, nu_tracer_hr, psi2_hr, dnu_dpsi_hr, dnu2_dpsi2_hr = weight_triaxial(rr, Ekk, Epp, ids, partmass, 0.01, 100, 1, 'Hernquist', [40.82])
    print(len(rbins))
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 15))
    plt.subplot(3,1,1)
    plt.plot(np.log(rbins_hr), np.log(nu_tracer_hr), lw=1, c='C9')
    plt.scatter(np.log(rbins), np.log(nu_tracer), s=1, c='k')
    plt.subplot(3,1,2)
    plt.plot(np.log(rbins_hr), np.log(dnu_dpsi_hr), lw=1, c='C9')
    plt.scatter(np.log(rbins), np.log(dnu_dpsi), s=1, c='k')
    plt.subplot(3,1,3)
    plt.plot(np.log(rbins_hr), np.log(dnu2_dpsi2_hr), lw=1, c='C9')
    plt.scatter(np.log(rbins), np.log(dnu2_dpsi2), s=1, c='k')
    plt.savefig('nu_tracers.png', bbox_inches='tight', dpi=150)
