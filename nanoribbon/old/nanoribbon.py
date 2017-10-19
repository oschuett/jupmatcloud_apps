import numpy as np
import matplotlib.pyplot as plt
from aiida.orm import load_node, DataFactory
from aiida.backends.utils import load_dbenv, is_dbenv_loaded
from aiida.orm.calculation.job.quantumespresso.pp import PpCalculation
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import cpk_colors
from ase.neighborlist import NeighborList

from aiida.orm.data.array.bands import BandsData
if not is_dbenv_loaded():
    load_dbenv()

#===============================================================================
def find_bandgap(bandsdata, number_electrons=None, fermi_energy=None):
    """
    Tries to guess whether the bandsdata represent an insulator.
    This method is meant to be used only for electronic bands (not phonons)
    By default, it will try to use the occupations to guess the number of
    electrons and find the Fermi Energy, otherwise, it can be provided
    explicitely.
    Also, there is an implicit assumption that the kpoints grid is
    "sufficiently" dense, so that the bandsdata are not missing the
    intersection between valence and conduction band if present.
    Use this function with care!

    :param (float) number_electrons: (optional) number of electrons in the unit cell
    :param (float) fermi_energy: (optional) value of the fermi energy.

    :note: By default, the algorithm uses the occupations array
      to guess the number of electrons and the occupied bands. This is to be
      used with care, because the occupations could be smeared so at a
      non-zero temperature, with the unwanted effect that the conduction bands
      might be occupied in an insulator.
      Prefer to pass the number_of_electrons explicitly

    :note: Only one between number_electrons and fermi_energy can be specified at the
      same time.

    :return: (is_insulator, gap), where is_insulator is a boolean, and gap a
             float. The gap is None in case of a metal, zero when the homo is
             equal to the lumo (e.g. in semi-metals).
    """

    def nint(num):
        """
        Stable rounding function
        """
        if (num > 0):
            return int(num + .5)
        else:
            return int(num - .5)

    if fermi_energy and number_electrons:
        raise ValueError("Specify either the number of electrons or the "
                         "Fermi energy, but not both")

    try:
        stored_bands = bandsdata.get_bands()
    except KeyError:
        raise KeyError("Cannot do much of a band analysis without bands")

    if len(stored_bands.shape) == 3:
        # I write the algorithm for the generic case of having both the
        # spin up and spin down array

        # put all spins on one band per kpoint
        bands = np.concatenate([_ for _ in stored_bands], axis=1)
    else:
        bands = stored_bands

    # analysis on occupations:
    if fermi_energy is None:

        num_kpoints = len(bands)

        if number_electrons is None:
            try:
                _, stored_occupations = bandsdata.get_bands(also_occupations=True)
            except KeyError:
                raise KeyError("Cannot determine metallicity if I don't have "
                               "either fermi energy, or occupations")

            # put the occupations in the same order of bands, also in case of multiple bands
            if len(stored_occupations.shape) == 3:
                # I write the algorithm for the generic case of having both the
                # spin up and spin down array

                # put all spins on one band per kpoint
                occupations = np.concatenate([_ for _ in stored_occupations], axis=1)
            else:
                occupations = stored_occupations

            # now sort the bands by energy
            # Note: I am sort of assuming that I have an electronic ground state

            # sort the bands by energy, and reorder the occupations accordingly
            # since after joining the two spins, I might have unsorted stuff
            bands, occupations = [np.array(y) for y in zip(*[zip(*j) for j in
                                                                [sorted(zip(i[0].tolist(), i[1].tolist()),
                                                                        key=lambda x: x[0])
                                                                 for i in zip(bands, occupations)]])]
            number_electrons = int(round(sum([sum(i) for i in occupations]) / num_kpoints))

            homo_indexes = [np.where(np.array([nint(_) for _ in x]) > 0)[0][-1] for x in occupations]
            if len(set(homo_indexes)) > 1:  # there must be intersections of valence and conduction bands
                return False, None
            else:
                homo = [_[0][_[1]] for _ in zip(bands, homo_indexes)]
                try:
                    lumo = [_[0][_[1] + 1] for _ in zip(bands, homo_indexes)]
                except IndexError:
                    raise ValueError("To understand if it is a metal or insulator, "
                                     "need more bands than n_band=number_electrons")

        else:
            bands = np.sort(bands)
            number_electrons = int(number_electrons)

            # find the zero-temperature occupation per band (1 for spin-polarized
            # calculation, 2 otherwise)
            number_electrons_per_band = 4 - len(stored_bands.shape)  # 1 or 2
            # gather the energies of the homo band, for every kpoint
            homo = [i[number_electrons / number_electrons_per_band - 1] for i in bands]  # take the nth level
            try:
                # gather the energies of the lumo band, for every kpoint
                lumo = [i[number_electrons / number_electrons_per_band] for i in bands]  # take the n+1th level
            except IndexError:
                raise ValueError("To understand if it is a metal or insulator, "
                                 "need more bands than n_band=number_electrons")

        if number_electrons % 2 == 1 and len(stored_bands.shape) == 2:
            # if #electrons is odd and we have a non spin polarized calculation
            # it must be a metal and I don't need further checks
            return False, None

        # if the nth band crosses the (n+1)th, it is an insulator
        gap = min(lumo) - max(homo)
        if gap == 0.:
            return False, 0.
        elif gap < 0.:
            return False, gap
        else:
            return True, gap, max(homo), min(lumo)

    # analysis on the fermi energy
    else:
        # reorganize the bands, rather than per kpoint, per energy level

        # I need the bands sorted by energy
        bands.sort()

        levels = bands.transpose()
        max_mins = [(max(i), min(i)) for i in levels]

        if fermi_energy > bands.max():
            raise ValueError("The Fermi energy is above all band energies, "
                             "don't know what to do")
        if fermi_energy < bands.min():
            raise ValueError("The Fermi energy is below all band energies, "
                             "don't know what to do.")

        # one band is crossed by the fermi energy
        if any(i[1] < fermi_energy and fermi_energy < i[0] for i in max_mins):
            return False, 0.

        # case of semimetals, fermi energy at the crossing of two bands
        # this will only work if the dirac point is computed!
        elif (any(i[0] == fermi_energy for i in max_mins) and
                  any(i[1] == fermi_energy for i in max_mins)):
            return False, 0.
        # insulating case
        else:
            # take the max of the band maxima below the fermi energy
            homo = max([i[0] for i in max_mins if i[0] < fermi_energy])
            # take the min of the band minima above the fermi energy
            lumo = min([i[1] for i in max_mins if i[1] > fermi_energy])

            gap = lumo - homo
            if gap <= 0.:
                raise Exception("Something wrong has been implemented. "
                                "Revise the code!")
            return True, gap, homo, lumo

#===============================================================================
def get_calc_obj(bands_object):
    bands_calc_obj = None
    scf_calc_obj = None
    hartree_calc_obj = None
    RemoteData = DataFactory('remote')
    StructureData = DataFactory('structure') 
    bands_calc_obj=bands_object.get_inputs()[0]

    for inp in bands_calc_obj.get_inputs():
        if type(inp) == RemoteData :
            tmp=inp
        if type(inp) == StructureData:
            tmp_str=inp
    scf_calc_obj=tmp.get_inputs()[0]
    for out in  tmp.get_outputs():
        if type(out) == PpCalculation:
            try:
                out.res['vacuum_level']
                hartree_calc_obj=out
            except:
                continue

    return bands_calc_obj, scf_calc_obj, hartree_calc_obj,tmp_str


#===============================================================================
def plot_b(hartree_run, bands_run,scf_fun,erang, allign,ax1,ax2=None):

#    bands_run=load_node(bands_n)
#    hartree_run=load_node(hartree_n)
    vac_lev = 0.0
    if hartree_run != None:
        vac_lev=hartree_run.res['vacuum_level']*27.211385/2
    fermi_energy = bands_run.res['fermi_energy'] - vac_lev
    bands=bands_run.out.output_band.get_bands() - vac_lev
    n_bands=bands.shape[1]
    x=np.linspace(0,0.5,n_bands)
    x_cross=[]
    y_cross=[]
    #print bands[0,:,0]
    #print bands.shape
    for i in range (0,bands.shape[-1]):
        if all(p < fermi_energy for p in bands[0,:,i]):
            ax1.plot(x, bands[0,:,i], color='gray')
        else:
            ax1.plot(x, bands[0,:,i], color='gray')

        if ax2 != None:
            ax2.plot(x, bands[1,:,i], color='blue')
        j=0
        while j < n_bands-1:
            if (bands[0,j,i]<fermi_energy and bands[0,j+1,i]>fermi_energy) or (bands[0,j,i]>fermi_energy and bands[0,j+1,i]<fermi_energy):
                x_cross.append(x[j])
                y_cross.append(fermi_energy)
            j+=1
        ax1.plot(x_cross,y_cross,'co')
#    print "fermi and allign",fermi_energy, allign
    ax1.set_ylim([allign+erang[0],allign+erang[1]])
#   plt.text(0,fermi_energy,'E_f={:4.3f} (eV)'.format(fermi_energy),
#   style='oblique', bbox={'facecolor':'blue', 'alpha':1, 'pad':0})

    ax1.axhline(y=fermi_energy, linewidth=2, color='red', ls='--') 
#    ax1.axes.get_xaxis().set_visible(False)


##===============================================================================
#def plot_bands(bands_object, ax1,ax2=None):
#    print "id:",bands_object.pk, 
#    bands_calc_obj=bands_object.get_inputs()[0]
#    RemoteData = DataFactory('remote')
#    StructureData = DataFactory('structure') 
#
#    for inp in bands_calc_obj.get_inputs():
#        if type(inp) == RemoteData :
#            tmp=inp
#        if type(inp) == StructureData:
#            tmp_str=inp
#    arr= tmp_str.get_composition()
#    print "formula:",arr,
#    scf_calc_obj=tmp.get_inputs()[0]
#
#    for out in  tmp.get_outputs():
#        if type(out) == PpCalculation:
#            hartree_cal_obj=out
#
#
#    fermi_energy= bands_calc_obj.res['fermi_energy']
#    noe=bands_calc_obj.res['number_of_electrons']
##    print "band gap:", find_bandgap(bands_object, fermi_energy=fermi_energy),
#    print "Magnetization:problem", # scf_calc_obj.res['fermi_energy'], #scf_calc_obj.res['absolute_magnetization_units'],
#    print "Total energy", scf_calc_obj.res['energy'],  scf_calc_obj.res['energy_units']
#
#    plot_b(hartree_cal_obj, bands_calc_obj,ax1,ax2)


#===============================================================================
def get_all_properties(node):
    props = {}

    props['pk'] = node.pk

    # gather more properties
    bands_calc_obj, scf_calc_obj, hartree_calc_obj, struct = get_calc_obj(node)
    props['bands_calc_obj'] = bands_calc_obj
    props['scf_calc_obj'] = scf_calc_obj
    props['hartree_calc_obj'] = hartree_calc_obj
    props['struct'] = struct

    if hasattr(scf_calc_obj.res, "absolute_magnetization"):
        props['amag'] = scf_calc_obj.res['absolute_magnetization']
    else:
        props['amag'] = 0.0

    if hasattr(scf_calc_obj.res, "total_magnetization"):
        props['tmag'] = scf_calc_obj.res['total_magnetization']
    else:
        props['tmag'] = 0.0

    props['fermi'] = scf_calc_obj.res['fermi_energy']

    # find vacuum level
    if hartree_calc_obj:
        vac_lev = hartree_calc_obj.res['vacuum_level']*27.211385/2.0
    else:
        vac_lev=0.0

    # find HOMO, LUMO, and Gap
    parts = find_bandgap(node, fermi_energy=props['fermi'])
    props['fermi'] -= vac_lev
    props['gap'] = parts[1]
    if parts[0]:
        props['homo'] = parts[2] - vac_lev
        props['lumo'] = parts[3] - vac_lev
    else:
        props['homo'] = props['fermi']
        props['lumo'] = props['fermi']

    props['formula'] = struct.get_ase().get_chemical_formula()

    return(props)


#===============================================================================
def plot_thumbnail(ax, ase_struct):
    s = ase_struct.repeat((2,1,1))

    cov_radii = [ covalent_radii[a.number] for a in s]
    nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
    nl.update(s)

    #plt.clf()
    #sizex = 5 #2.8
    #sizey = sizex*(s.cell[1][1]-10)/s.cell[0][0]
    #fig_thumb, ax_thumb = plt.subplots(1,1,figsize=(sizex,sizey))
    ax.set_aspect(1)
    ax.axes.set_xlim([0,s.cell[0][0]])
    ax.axes.set_ylim([5,s.cell[1][1]-5])
    ax.set_axis_bgcolor((0.423,0.690,0.933))
    ax.axes.get_yaxis().set_visible(False)

    name = ase_struct.get_chemical_formula() # get name before repeat
    ax.set_xlabel(name, fontsize=12)
    ax.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')

    ax.plot([5,5],[5,s.cell[1][1]-5],color='r',linewidth=4,linestyle='--')
    ax.plot([5+s.cell[0][0]/2,5+s.cell[0][0]/2],[5,s.cell[1][1]-5],color='r',linewidth=4,linestyle='--')

    for at in s:
        #circles
        x,y,z = at.position
        n = atomic_numbers[at.symbol]
        ax.add_artist(plt.Circle((x,y), covalent_radii[n]*0.5, color=cpk_colors[n], fill=True, clip_on=True))
        #bonds
        nlist = nl.get_neighbors(at.index)[0]
        for theneig in nlist:
            x,y,z = (s[theneig].position +  at.position)/2
            x0,y0,z0 = at.position
            if (x-x0)**2 + (y-y0)**2 < 2 :
                ax.plot([x0,x],[y0,y],color=cpk_colors[n],linewidth=2,linestyle='-')
#EOF