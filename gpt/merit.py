from .parsers import read_particle_gdf_file
import numpy as np
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

        iparticles=read_particle_gdf_file(os.path.join(G.path, G.get_dist_file()))
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

            all_variables = all_coordinates + all_momentum + cartesian_velocity + angles + energy + ['t']

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

            # Basic Custom paramters:
            m['end_max[sigma_x, sigma_y]'] = max([m['end_sigma_x'], m['end_sigma_y']])
            m['end_max[norm_emit_x, norm_emit_y]'] = max([m['end_norm_emit_x'], m['end_norm_emit_y']])
            
    except Exception:

        m['error']=True
    
    # Remove annoying strings
    if 'why_error' in m:
        m.pop('why_error')
        
    return m

