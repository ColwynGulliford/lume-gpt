from .gpt import run_gpt
from .gpt_distgen import run_gpt_with_distgen
from .tools import full_path
import numpy as np
import json
from inspect import getfullargspec
import os


def end_output_data(output):
    """
    Some outputs are lists. Get the last item. 
    """
    o = {}
    for k in output:
        val = output[k]
        if isinstance(val, str): # Encode strings
            o[k] = val.encode()
        elif np.isscalar(val):
            o[k]=val
        else:
            o['end_'+k]=val[-1]
           
    return o


def default_gpt_merit(G):
    """
    merit function to operate on an evaluated LUME-Astra object A. 
    
    Returns dict of scalar values
    """
    # Check for error
    if G.error:
        return {'error':True}
    else:
        m= {'error':False}
    
    # Gather output
    m.update(end_output_data(G.output))
    
    # Load final screen for calc
    screen = G.screen[-1]        
    
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





