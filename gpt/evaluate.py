from .gpt import run_gpt
from .gpt_distgen import run_gpt_with_distgen
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
    merit function to operate on an evaluated LUME-GPT object G  
    
    Returns dict of scalar values
    """
    # Check for error
    if G.error:
        return {'error':True}
    else:
        m= {'error':False}
    
    # Load final screen for calc
    if(len(G.screen)>0):
        screen = G.screen[-1]    
        m['end_std_x'] = screen['x'].std()
        m['end_std_y'] = screen['y'].std()
        m['end_qbunch'] = np.abs(np.sum(screen['q']*screen['nmacro']))
        m['end_norm_emitt_x'] = get_norm_emitt(screen['x'],screen['GBx'])
        m['end_norm_emitt_y'] = get_norm_emitt(screen['y'],screen['GBy'])
        m['end_std_t']=screen['t'].std()
        m['end_n_particle']=len(screen['x'])
        m['end_z_screen']=screen['z'].mean()

    else:
        raise ValueError('No final screen in GPT data passed to gpt.evaluate.default_gpt_merit!')

    # Remove annoying strings
    if 'why_error' in m:
        m.pop('why_error')
        
    return m


def evaluate(settings, simulation='gpt', archive_path=None, merit_f=None, **params):
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
        G = run_gpt_with_distgen(settings, **params)
        print('blip')
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





