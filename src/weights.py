#!bin/

import numpy as np
import sys

from pygadgetreader import readsnap
from tracers_dens import *

"""
To-do:

1. Organize weight_triaixal function
2. Why weights are negative ?
3. How to proplery bin the energy?
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

def weight_triaxial(r, Ek, Ep, partID, m, bsize, N_Eb, stellar_mass, profile, profile_params):
    """
    N_Eb : number of bins for the Energy



    """
    G = 4.30071e-6
    stellar_mass=stellar_mass*1e10
    Ep=Ep-G*m*np.size(r)/np.max(r) #correction term
    E=Ek+Ep

    shift_energy = -np.min(Ep)
    E += shift_energy
    Ep += shift_energy

    #I chose 300 because this code was initially used for cosmological halos which were way too messed up beyond 300 kpc
    w=np.where((r<300) & (r!=r[0]))
    r=r[w]
    Ep=Ep[w]
    Ek=Ek[w]
    E=E[w]
    partID=partID[w]

    # Histogram of radius to smooth the potential Phi(r)
    #  - spherical averaging for triaxial halos for Laporte 2013
    #
    MIN,MAX = np.min(np.log10(r)), np.max(np.log10(r))
    Nbins= (MAX-MIN)/bsize
    histo_rad,redges=np.histogram(np.log10(r), bins = np.linspace(MIN, MAX, Nbins))
    rbins=np.ndarray(shape=np.size(redges)-1, dtype=float)

    for i in range(1,np.size(redges)):
        rbins[i-1]=redges[i-1]-(redges[i]-redges[i-1])/2.

    rbins=10**rbins

    nn=np.size(rbins)
    binsize_r=np.ndarray(shape=nn, dtype=float)

    #     binsize_r is evaluated here for g(E) calculation
    for j in range(0,nn):
        binsize_r[j]=10**redges[j+1]-10**redges[j]

    #TRACER PARAMETRISATION
    #bb=0.5 #scale radius
    nu_tracer=rho_tracers(rbins, 1, profile, profile_params)
    #bb=5
    #nu_tracer=(3.0/(4.0*np.pi*bb**3))*(1.0+(rbins/bb)**2)**(-2.5)
    #nu_tracer=stellar_density(stellar_mass, params)

    #Need to do the reverse indices here -
    pot2=np.ndarray(shape=np.size(histo_rad), dtype=float)

    for j in range(0, np.size(redges)-1):
        wbin=np.where((np.log10(r)>=redges[j]) & (np.log10(r)<redges[j+1]))
        if(np.size(wbin)>0):
            pot2[j]=np.mean(Ep[wbin]) #reverse indices in IDL is much faster than this junk

    # forgot why I wanted more than 20 particles in the bins, maybe sth to do with gradient not working with missing data
    # this can be improved using an interpolating scheme.

    w=np.where(histo_rad>20.)
    rbins=rbins[w]
    binsize_r=binsize_r[w]
    nu_tracer=nu_tracer[w]
    pot2=pot2[w]
    pot2-=shift_energy
    psi2=(-1.0)*pot2
    E-=shift_energy
    epsilon=(-1.0)*E

    #Fetching derivatives from the data necessary for the Eddington formula evalution

    dnu_dpsi=np.gradient(nu_tracer, psi2)
    dnu2_dpsi2=np.gradient(dnu_dpsi, psi2)

    #Binning Energy for g(E) and f(E) (f(epsilon)) calculations
    Histo_E, Edges = np.histogram(E, bins=N_Eb)
    Ebins=np.ndarray(shape=np.size(Histo_E), dtype=float)
    for i in range(1,np.size(Edges)):
        Ebins[i-1]=Edges[i-1]-(Edges[i]-Edges[i-1])/2.

    Histo_epsilon, epsdges = np.histogram(epsilon, bins=N_Eb)
    epsilon_bins=np.ndarray(shape=np.size(Histo_epsilon), dtype=float)
    for i in range(1,np.size(epsdges)):
        epsilon_bins[i-1]=epsdges[i-1]-(epsdges[i]-epsdges[i-1])/2.


    #Total N(E) differential energy distribution
    Histo_M=Histo_E*m/np.sqrt((Ebins[2]-Ebins[1])**2)

    # EDDINGTON FORMULA --------------
    dpsi=np.ndarray(shape=np.size(psi2), dtype=float)
    for i in range (1, np.size(dpsi)):
        dpsi[i]=psi2[i]-psi2[i-1]


    distribution_function=np.ndarray(shape=np.size(epsilon_bins), dtype=float)
    for i in range(0,np.size(epsilon_bins)):
        w=np.where(psi2<epsilon_bins[i])
        #x=np.min(w) #i don't think I use this anywhere
        eps=epsilon_bins[i]
        if (np.size(w[0])!=0):
            w=np.array(w)
            tot1=dpsi[w[0,0]::]
            tot2=dnu2_dpsi2[w[0,0]::]
            tot3=np.sqrt(2.0*(eps-psi2[w[0,0]::]))
            tot=tot1*tot2/tot3
            val=(1.0)/(np.sqrt(8.0)*np.pi**2)*np.sum(tot) #Arthur's eval as Sum (in sims no divergence due to res)
            #print val, i, "val, i"
            distribution_function[i]=val
        else:
            distribution_function[i]=0
    #DENSITY OF STATES--------------
    wrme=np.ndarray(shape=np.size(Ebins), dtype=int)
    rme=np.ndarray(shape=np.size(Ebins), dtype=float)
    for i in range(0, np.size(Ebins)):
        wpot_equals_E=np.where(pot2<=Ebins[i])
        if (np.size(wpot_equals_E)!=0):
            wrme[i]=np.max(np.array(wpot_equals_E))
        else:
            wrme[i]=0

    density_of_states=np.ndarray(shape=np.size(Ebins), dtype=float) # density of states integral (evaluated as sum)
    for i in range(0,np.size(Ebins)):
        if (np.size(wrme[i])==0):
            g1=0.0
        else:
            g1=rbins[0:wrme[i]]**2
            g2=np.sqrt(2.0*(Ebins[i]-pot2[0:wrme[i]]))
            density_of_states[i]=(4.0*np.pi)**2*np.sum(binsize_r[0:wrme[i]]*g1*g2)

    indsort=np.argsort(distribution_function) #sorted indices
    indsort=indsort[::-1] #reverse
    # weights= D.F(tracers)/ (D.F.(self-consistent)) - self-consistent D.F. f(E) generates the potential Phi
    # N(E)=f(E)*g(E)
    Weights=distribution_function[indsort[::-1]]/((Histo_M)/density_of_states)
    # cast the weights to every particle
    Weights_array=np.ndarray(shape=np.size(r), dtype=float)
    for j in range(0, np.size(Edges)-1):
        wbin=np.where((E>=Edges[j]) & (E<Edges[j+1]))
        if(np.size(wbin[0])!=0):
            Weights_array[wbin]=Weights[j]

    #Ensure that the sum of the weights = mass of the tracers - this is not strictly needed
    X=stellar_mass/(np.sum(Weights_array)*m)
    Weights_array=Weights_array*X

    #print(np.size(Weights_array))
    #return the IDS from which the weights are associated to the particles
    #needed for tracking where the tracers end up in subsequent snapshots
    # Each particle gets a weight.
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
    weight, p_ids = weight_triaxial(rr,Ekk,Epp,ids,partmass,0.01,100,1, profile)
