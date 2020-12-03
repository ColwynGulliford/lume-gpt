from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import c_light, e_charge, m_e


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
        'H2+'
    
    TODO: more species
    
    """
    
    
    if mass == 3.347115e-27 and charge== 1.602e-19:
        return 'H2+'
    
    m = round(mass*1e32)/1e32
    q = round(charge*1e20)/1e20
    if m == 9.1e-31:
        if q == 1.6e-19:
            return 'positron'
        if q == -1.6e-19:
            return 'electron'
        
    raise Exception(f'Cannot identify species with mass {mass} and charge {charge}')
   

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

    mec = m_e #[mc] in eV/c

    data['px'] = gpt_output_dict['GBx']*mec
    data['py'] = gpt_output_dict['GBy']*mec
    data['pz'] = gpt_output_dict['GBz']*mec

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


def gdf_to_particle_groups(gdffile, verbose=False):

    """
    Read an output gdf file from GPT into a lists of tout and screen particle groups
    """

    (tdata, pdata) = read_gdf_file(gdffile, verbose=verbose)

    all_pgs = raw_data_to_particle_groups(tdata, pdata, verbose=verbose)

    touts = all_pgs[:len(tdata)]
    screens = all_pgs[len(tdata):]

    return (touts, screens)

def initial_beam_to_particle_group(gdfile, verbose=0):

    screen  = read_particle_gdf_file(gdffile, verbose=verbose)
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

    
    
    
