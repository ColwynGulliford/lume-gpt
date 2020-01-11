
from hashlib import blake2b
import numpy as np
import json

import subprocess
import os, errno



def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))



class NumpyEncoder(json.JSONEncoder):
    """
    See: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data. 
    Used JSON dumps to form strings, and the blake2b algorithm to hash.
    
    """
    h = blake2b(digest_size=16)
    for key in keyed_data:
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NumpyEncoder).encode()
        h.update(s)
    return h.hexdigest()  
