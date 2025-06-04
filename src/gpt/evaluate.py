from .gpt import run_gpt
from .tools import full_path
from .merit import default_gpt_merit

import os

def evaluate_gpt(settings=None, 
            initial_particles=None,
            gpt_input_file=None, 
            workdir=None, 
            gpt_bin='$GPT_BIN', 
            timeout=2500, 
            auto_phase=False,
            verbose=False,
            gpt_verbose=False,
            asci2gdf_bin='$ASCI2GDF_BIN',
            merit_f=None,
            archive_path=None):
    """
    Evaluate gpt. 
    
    If merit_f is provided, this function will be used to form the outputs. 
    Otherwise a default funciton will be applied.
    
    Will raise an exception if there is an error. 
    
    """

    G = run_gpt(settings=settings,
                 initial_particles=initial_particles,
                 gpt_input_file=gpt_input_file,
                 workdir=workdir, 
                 gpt_bin=gpt_bin,
                 timeout=timeout,
                 auto_phase=auto_phase,
                 verbose=verbose,
                 gpt_verbose=gpt_verbose,
                 asci2gdf_bin=asci2gdf_bin)  

        
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





