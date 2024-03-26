from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import c_light, e_charge, mec2
from pmd_beamphysics.species import mass_of

from scipy.constants import physical_constants


from gpt.tools import transform_to_centroid_coordinates
import numpy as np

from gpt.parsers import read_gdf_file
from gpt.parsers import read_particle_gdf_file

def identify_species(mass, charge):
    """
    Simple function to identify a species based on its mass in kg and charge in C.
    
    Finds species:
        'electron'
        'positron'
        'proton'
        'H2+'
    
    TODO: more species
    
    """

    qelem = physical_constants['elementary charge'][0]

    if np.isclose(mass, 3.347115e-27, atol=0) and np.isclose(charge, qelem, atol=0):
        return 'H2+'

    elif np.isclose(mass,  physical_constants['electron mass'][0], atol=0) and np.isclose(charge, -qelem, atol=0):
        return 'electron'

    elif np.isclose(mass,  physical_constants['electron mass'][0], atol=0) and np.isclose(charge, +qelem, atol=0):
        return 'positron'

    elif np.isclose(mass,  physical_constants['proton mass'][0], atol=0) and np.isclose(charge, +qelem, atol=0):
        return 'proton'
        
    else:
        raise ValueError(f'Cannot identify species with mass {mass} and charge {charge}')
   

def raw_data_to_particle_data(gpt_output_dict, verbose=False):

    """
    Convert a gpt_out (tout or screen) dict to a standard form
    
    """
    data = {}

    n_particle = len(gpt_output_dict['x'])
     
    masses = np.unique(gpt_output_dict['m'])
    charges = np.unique(gpt_output_dict['q'])
    assert len(masses) == 1, 'All masses must be the same.'
    assert len(charges) == 1, 'All charges must be the same'
    mass = masses[0]
    charge = charges[0]

    species = identify_species(mass, charge)
    
    data['species'] = species
    data['n_particle'] = n_particle

    data['x'] = gpt_output_dict['x']
    data['y'] = gpt_output_dict['y']
    data['z'] = gpt_output_dict['z']
    factor = c_light**2 /e_charge # kg -> eV

    #data['px'] = gpt_output_dict['GBx']*gpt_output_dict['m']*factor
    #data['py'] = gpt_output_dict['GBy']*gpt_output_dict['m']*factor
    #data['pz'] = gpt_output_dict['GBz']*gpt_output_dict['m']*factor

    mc = mass_of(species)  # Returns rest energy in eV, which corresponds to same numeric value of mc [eV/c] 

    data['px'] = gpt_output_dict['GBx']*mc
    data['py'] = gpt_output_dict['GBy']*mc
    data['pz'] = gpt_output_dict['GBz']*mc

    data['t'] = gpt_output_dict['t']
    data['status'] = np.full(n_particle, 1)
    data['id'] = gpt_output_dict['ID']

    #print(c_light, e_charge, gpt_output_dict['m'][0], m_e)

    data['weight'] = abs(gpt_output_dict['q']*gpt_output_dict['nmacro'])

    if( np.all(data['weight'] == 0.0) ):
        data['weight']= np.full(data['weight'].shape, 1/len(data['weight']))
    
    
    return data


def raw_data_to_particle_groups(touts, screens, verbose=False, ref_ccs=False):

    """
    Coverts a list of touts to a list of ParticleGroup objects
    """
    if(verbose):
        print('   Converting tout and screen data to ParticleGroup(s)')

    if(ref_ccs):

        pg_touts = [ ParticleGroup(data=raw_data_to_particle_data(datum))  for datum in touts ]
        pg_screens = [ ParticleGroup(data=raw_data_to_particle_data(datum))  for datum in screens ]
        new_touts = [transform_to_centroid_coordinates(tout) for tout in pg_touts]
        
        return new_touts + pg_screens     

    else:
        return [ ParticleGroup(data=raw_data_to_particle_data(datum))  for datum in touts+screens ] 


def gdf_to_particle_groups(gdffile, verbose=False, load_fields=False):

    """
    Read an output gdf file from GPT into a lists of tout and screen particle groups
    """
    print(load_fields)

    (tdata, pdata, fields) = read_gdf_file(gdffile, verbose=verbose, load_fields=load_fields)

    all_pgs = raw_data_to_particle_groups(tdata, pdata, verbose=verbose)

    touts = all_pgs[:len(tdata)]
    screens = all_pgs[len(tdata):]

    return (touts, screens, fields)

def initial_beam_to_particle_group(gdffile, verbose=0, extra_screen_keys=['q','nmacro','ID', 'm'], missing_data=None):

    screen  = read_particle_gdf_file(gdffile, verbose=verbose, extra_screen_keys=extra_screen_keys)

    if(missing_data is not None):

        for mdatum in missing_data:

            if(mdatum not in screen.keys() and len(missing_data[mdatum])==len(screen['x'])):
                screen[mdatum] = missing_data[mdatum]


    return ParticleGroup(data=raw_data_to_particle_data(screen))


def particle_stats(particle_groups, key):
    """
    Gets statistic of a list of particle groups
    
    
    key can be any key that ParticleGroup can calculate:
        mean_energy
        mean_z
        mean_t
        sigma_x
        norm_emit_x
        mean_kinetic_energy
        ...
        
    
    """
    return np.array([p[key] for p in particle_groups])

    
    
    
