from .gpt import run_gpt
from .gpt_distgen import run_gpt_with_distgen
from .parsers import read_particle_gdf_file
from .tools import full_path
import numpy as np
import json
from inspect import getfullargspec
import os





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
        raise ValueError(f'Unsupported simulation {simulation}')
        
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





