from .gpt import run_gpt
from .gpt_distgen import run_gpt_with_distgen
from .parsers import read_particle_gdf_file
from .tools import full_path
import numpy as np
import json
from inspect import getfullargspec
import os



           
def get_norm_emitt(x,p):

    x0 = x.mean()
    p0 = p.mean()
    stdx = x.std()
    stdp = p.std()
    xp = np.mean((x-x0)*(p-p0))

    return np.sqrt( stdx**2 * stdp**2 - xp**2 )
    


def default_gpt_merit(G):
    """
    default merit function to operate on an evaluated LUME-GPT object G.  
    
    Returns dict of scalar values containing all stat quantities a particle group can compute 
    """
    # Check for error
    if G.error:
        return {'error':True}
    else:
        m= {'error':False}

    if(G.initial_particles):
        start_n_particle = G.initial_particles['n_particle']

    elif(G.get_dist_file()):

        iparticles=read_particle_gdf_file(G.get_dist_file())
        start_n_particle = len(iparticles['x'])

    else:
        raise ValueError('evaluate.default_gpt_merit: could not find initial particles.')


    try:

        # Load final screen for calc
        if(len(G.screen)>0):

            screen = G.screen[-1]   # Get data on last screen

            cartesian_coordinates = ['x', 'y', 'z']
            cylindrical_coordinates = ['r', 'theta']
            all_coordinates = cartesian_coordinates + cylindrical_coordinates

            all_momentum = [f'p{var}' for var in all_coordinates]
            cartesian_velocity = [f'beta_{var}' for var in cartesian_coordinates]
            angles = ['xp', 'yp']
            energy = ['energy', 'kinetic_energy', 'p', 'gamma']

            all_variables = all_coordinates + all_momentum + cartesian_velocity + angles + energy

            keys =  ['n_particle', 'norm_emit_x', 'norm_emit_y', 'higher_order_energy_spread']

            stats = ['mean', 'sigma', 'min', 'max']
            for var in all_variables:
                for stat in stats:
                    keys.append(f'{stat}_{var}')

            for key in keys:
                m[f'end_{key}']=screen[key]

            # Extras
            m['end_z_screen']=screen['mean_z']
            m['end_n_particle_loss'] = start_n_particle - m['end_n_particle']
            m['end_total_charge'] = screen['charge']
            
    except Exception as ex:

        m['error']=True
    
    # Remove annoying strings
    if 'why_error' in m:
        m.pop('why_error')
        
    return m


def evaluate(settings, 
             simulation='gpt', 
             archive_path=None, 
             merit_f=None, 
             gpt_input_file=None,
             distgen_input_file=None,
             workdir=None, 
             use_tempdir=True,
             gpt_bin='$GPT_BIN',
             timeout=2500,
             auto_phase=False,
             verbose=False,
             gpt_verbose=False,
             asci2gdf_bin='$ASCI2GDF_BIN'):
    """
    Evaluate gpt using possible simulations:
        'gpt'
        'gpt_with_distgen'
    
    Returns a flat dict of outputs. 
    
    If merit_f is provided, this function will be used to form the outputs. 
    Otherwise a default funciton will be applied.
    
    Will raise an exception if there is an error. 
    
    """

    # Pick simulation to run 
    if simulation=='gpt':
        G = run_gpt(settings, **params)

    elif simulation == 'gpt_with_distgen':

        # Import here to limit dependency on distgen
        from .gpt_distgen import run_gpt_with_distgen
        G = run_gpt_with_distgen(settings=settings,
                                 gpt_input_file=gpt_input_file,
                                 distgen_input_file=distgen_input_file,
                                 workdir=workdir, 
                                 use_tempdir=use_tempdir,
                                 gpt_bin=gpt_bin,
                                 timeout=timeout,
                                 auto_phase=auto_phase,
                                 verbose=verbose,
                                 gpt_verbose=gpt_verbose,
                                 asci2gdf_bin=asci2gdf_bin)
    else:
        raise 
        
    if merit_f:
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)

    if output['error']:
        raise
    
    fingerprint = G.fingerprint()
    
    output['fingerprint'] = fingerprint
    
    if archive_path:
        path = full_path(archive_path)
        assert os.path.exists(path), f'archive path does not exist: {path}'
        archive_file = os.path.join(path, fingerprint+'.h5')
        G.archive(archive_file)
        output['archive'] = archive_file
        
    return output


def evaluate_gpt(settings, archive_path=None, merit_f=None, **params):
    """
    Convenience wrapper. See evaluate. 
    """
    return evaluate(settings, simulation='gpt', 
                    archive_path=archive_path, merit_f=merit_f, **params)


def evaluate_gpt_with_distgen(settings, archive_path=None, merit_f=None, **params):
    """
    Convenience wrapper. See evaluate. 
    """
    return evaluate(settings, simulation='gpt_with_distgen', 
                    archive_path=archive_path, merit_f=merit_f, **params)





