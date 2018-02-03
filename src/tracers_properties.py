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

    phys_quantitiy_w = phys_quantitiy_ids[s]


    return phys_quantitiy_w, wids[s]


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
    import soda
    import sys

    rcut = float(sys.argv[1])
    bins_w = float(sys.argv[2])
    nbins = int(sys.argv[3])
    plot_bins = int(sys.argv[4])


    pp= readsnap('../halos/LMC3_2M_vir_000', 'pos', 'dm')
    vv= readsnap('../halos/LMC3_2M_vir_000', 'vel', 'dm')
    massarr= readsnap('../halos/LMC3_2M_vir_000', 'mass', 'dm')
    Epp = readsnap('../halos/LMC3_2M_vir_000', 'pot', 'dm')
    ids = readsnap('../halos/LMC3_2M_vir_000', 'pid', 'dm')

    rr=np.sqrt(pp[:,0]**2+pp[:,1]**2+pp[:,2]**2)

    r_cut = index = np.where((rr<rcut))[0]

    pp = pp[r_cut]
    rr = rr[r_cut]
    vv = vv[r_cut]

    massarr = massarr[r_cut]
    Epp = Epp[r_cut]
    ids = ids[r_cut]

    pp_fut= readsnap('../halos/LMC3_2M_vir_009', 'pos', 'dm')
    massarr_fut= readsnap('../halos/LMC3_2M_vir_009', 'mass', 'dm')
    ids_fut = readsnap('../halos/LMC3_2M_vir_009', 'pid', 'dm')

    rr_fut=np.sqrt(pp_fut[:,0]**2+pp_fut[:,1]**2+pp_fut[:,2]**2)


    partmass=massarr[0]*1e10 #generated the halo particles as "bulge"-type in Gadget file
    v2=vv[:,0]**2+vv[:,1]**2+vv[:,2]**2
    Ekk=0.5*v2

    a=0.5
    r_profiles = np.linspace(1, rcut, plot_bins-1)
    teo_plummer = soda.profiles.dens_plummer(a, r_profiles, 1)
    #weights_plum, w_ids = weights.weight_triaxial(rr, Ekk, Epp, ids, partmass, bins_w, nbins, 1, 'Plummer', [a])
    weights_hern, w_ids_hern = weights.weight_triaxial(rr, Ekk, Epp, ids, partmass, bins_w, nbins, 1, 'Hernquist', [a])
    #print(ids, w_ids_hern)

    density_hern = den_tracers(weights_hern, w_ids_hern, rr, massarr, plot_bins, rcut)
    density_hern_fut = den_tracers(weights_hern, w_ids_hern, rr_fut, massarr_fut, plot_bins, rcut)



    plt.title('Stellar tracers density $N_b = {}, bins_w = {}$'.format(nbins, bins_w))
    plt.loglog(r_profiles, density_hern, label=r'$\rho_{*} pres$', c='k')
    plt.loglog(r_profiles, density_hern_fut, label=r'$\rho_{*} fut$', c='r')
    plt.xlabel('$r[Kpc]$')
    plt.ylabel(r'$\rho $')
    #plt.loglog(r_profiles, teo_plummer, label=r'$\rho_{theo}$', c='k', ls='--')
    plt.legend()
    plt.savefig('density_tracer_{}_{}.png'.format(bins_w, nbins))
    plt.close()
