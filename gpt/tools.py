
from hashlib import blake2b
import numpy as np
import json

import subprocess
import os, errno
import time

def execute(cmd):
    """
    
    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    
    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")
        
    Useful in Jupyter notebook
    
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

# Alternative execute
def execute2(cmd, timeout=None):
    """
    Execute with time limit (timeout) in seconds, catching run errors. 
    """
    
    output = {'error':True, 'log':''}
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout = timeout)
        output['log'] = p.stdout
        output['error'] = False
        output['why_error'] =''
    except subprocess.TimeoutExpired as ex:
        output['log'] = ex.stdout+'\n'+str(ex)
        output['why_error'] = 'timeout'
    except:
        output['log'] = 'unknown run error'
        output['why_error'] = 'unknown'
    return output

def execute3(cmd, kill_msgs=[], verbose=False, timeout=1e6, dirname=None):

    tstart = time.time()
   
    exception = None
    run_time = 0
    all_good = True 

    kill_on_warning = len(kill_msgs)>1
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    log = []

    while(all_good):

        pout = (process.stderr.readline()).decode("utf-8")

        if(pout):
            log.append(pout)
            if(verbose):
                print(pout.strip()) 

        if(pout == '' and process.poll() is not None):
            break

        if(time.time()-tstart > timeout): 
            process.terminate()
            exception = "run timed out"
            break

        if(kill_on_warning):
            for warning in kill_msgs:
                if(warning in pout):
                    process.terminate()
                    exception = pout               
                    break

    rc = process.poll()
    
    tstop = time.time()
    if(verbose>0):
        print("done. Time ellapsed: "+"{0:.2f}".format(tstop-tstart) + " sec.")
  
    run_time=tstop-tstart

    return run_time, exception, log


def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))



class NpEncoder(json.JSONEncoder):
    """
    See: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data. 
    Used JSON dumps to form strings, and the blake2b algorithm to hash.
    
    """
    h = blake2b(digest_size=16)
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()  



