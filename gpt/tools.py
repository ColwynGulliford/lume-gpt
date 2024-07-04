
from hashlib import blake2b
import numpy as np
import json

import subprocess
import os
import datetime
import time

import importlib

from math import pi
import math

from gpt.watcher import Watcher

from pmd_beamphysics import ParticleGroup

from scipy.constants import c


DEFAULT_KILL_MSGS = ["gpt: Spacecharge3Dmesh:", 'Error:', 'gpt: No valid GPT license', 'malloc', 'Segmentation fault']

def execute(cmd, kill_msgs=[], verbose=False, timeout=1e6, workdir=''):

    """ Function for execution of GPT """
    w = Watcher(cmd=cmd, timeout=timeout, verbose=verbose, kill_msgs=kill_msgs, workdir=workdir)
    w.run()

    return w.run_time, w.exception, w.log


def executeOld(cmd):
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

def execute3(cmd, kill_msgs=[], verbose=False, timeout=1e6):

    tstart = time.time()
   
    exception = None
    run_time = 0 
    log = []

    kill_on_warning = len(kill_msgs)>1
    
    try:

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        while(process.poll() is None):

            pout = (process.stderr.readline())#.decode("utf-8")

            if(pout):
                log.append(pout)
                if(verbose):
                    print(pout.strip()) 

            elif pout is None:
                break

            if(pout == '' and process.poll() is not None):
                break

            if(time.time()-tstart > timeout): 
                process.kill()
                exception = "timeout"
                break

            if(kill_on_warning):
                for warning in kill_msgs:
                    if(warning in pout):
                        process.kill()
                        exception = pout               
                        break

        rc = process.poll()

    except Exception as ex:
        exectption=str(ex)

    tstop = time.time()
    if(verbose>0):
        print(f'done. Time ellapsed: {tstop-tstart} sec.')
  
    run_time=tstop-tstart

    return run_time, exception, log

def execute4(cmd, kill_msgs=[], verbose=False, timeout=1e6):
    pass








"""UTC to ISO 8601 with Local TimeZone information without microsecond"""
def isotime():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()    



def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))


def native_type(value):
    """
    Converts a numpy type to a native python type.
    See:
    https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998
    """
    return getattr(value, 'tolist', lambda: value)()    


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

def get_function(name):
    """
    Returns a function from a fully qualified name or global name.
    """
    
    # Check if already a function
    if callable(name):
        return name
    
    if not isinstance(name, str):
        raise ValueError(f'{name} must be callable or a string.')
    
    if name in globals(): 
        if callable(globals()[name]):
            f = globals()[name]
        else:
            raise ValueError(f'global {name} is not callable')
    else:
        # try to import
        m_name, f_name = name.rsplit('.', 1)
        module = importlib.import_module(m_name)
        f = getattr(module, f_name)
    
    return f 


def is_floatable(value):
    """Checks if an object can be cast to a float, returns True or False"""

    try:
        float(value)
        return True
    except:
        return False


# Geometry
def deg(theta_rad):
    return theta_rad*180/math.pi

def rad(theta_deg):
    return theta_deg*math.pi/180

def cvector(rvector):
    """ Converts a numpy array or list into a column vector """
 
    if(rvector is None):
        return rvector
    elif(isinstance(rvector,list)):
        return np.array([np.array(rvector)]).T
    elif(rvector.shape==(3,1)):
        return rvector
    elif(rvector.shape==(3,)):
        return np.array([np.array(rvector)]).T

    print(rvector.shape)
    print('wooooooops')

def rotation_matrix(theta=0, phi=0, psi=0, units='deg'):

    """Defines a general 3d rotation in terms of the orientation angles theta, phi, psi"""

    rpy = Rpy(theta,units)
    rmx = Rmx(phi, units)
    rpz = Rpz(psi, units)

    return np.matmul(rpy, np.matmul(rmx, rpz))

def Rpy(angle=0, units='deg'):

    """Defines a roation in around +y direction"""

    if(units=='deg'):
        angle = angle*pi/180

    C = np.cos(angle)
    S = np.sin(angle)

    M = np.identity(3)

    M[0,0] = +C
    M[0,2] = +S
    M[2,0] = -S
    M[2,2] = +C

    return M

def Rmx(angle=0, units='deg'):

    """Defines a roation around -x direction"""

    if(units=='deg'):
        angle = angle*pi/180

    C = np.cos(angle)
    S = np.sin(angle)

    M = np.identity(3)

    M[1,1] = +C
    M[1,2] = +S
    M[2,1] = -S
    M[2,2] = +C

    return M

def Rpz(angle=0, units='deg'):

    """Defines a roation arounx +z direction"""

    if(units=='deg'):
        angle = angle*pi/180

    C = np.cos(angle)
    S = np.sin(angle)

    M = np.identity(3)

    M[0,0] = +C
    M[0,1] = -S
    M[1,0] = +S
    M[1,1] = +C

    return M


def get_arc(R, p1, e1, theta, npts=100):

    thetas = np.linspace(0, theta, npts)
    arc = np.zeros( (3, npts) )

    for ii, theta in enumerate(thetas):

        M = rotation_matrix(theta)
        pii = p1 + np.sign(theta)*R*(e1 - np.matmul(M,e1))
        arc[:,ii] = np.squeeze(pii)

    return arc



def write_ecs(p, M):

    ecs_str = ''
    for ii in range(3):
        ecs_str = ecs_str + f'{p[ii,0]}, '

    for jj in range(2):
        for ii in range(3):
            ecs_str = ecs_str + f'{M[ii,jj]}, '

    return ecs_str

def in_ecs(p, ecs_origin, M_ecs):
    return np.matmul( np.linalg.inv(M_ecs), (p-ecs_origin))



def transform_to_centroid_coordinates(particles, e2=cvector([0,1,0])):

    names = ['x', 'y', 'z', 'px', 'py',   'pz', 't']

    centroid = {name:particles[f'mean_{name}'] for name in names}

    p0 = cvector([particles['mean_px'], particles['mean_py'], particles['mean_pz']]) 
    e3 = p0/np.linalg.norm(p0)
    e1 = cvector(np.cross(e2.T, e3.T).T )

    data = {'n_particle':particles['n_particle'],
            'species':particles['species'],
            'weight':particles['weight'],
            'status':particles['status'],
            'id':particles['id']
    }

    M = np.linalg.inv(np.concatenate( (e1, e2, e3), axis=1))

    r = np.zeros( (3, len(particles['x']) ) )
    r[0,:] = particles['x'] - centroid['x']
    r[1,:] = particles['y'] - centroid['y']
    r[2,:] = particles['z'] - centroid['z']

    p = np.zeros( (3, len(particles['x']) ) )
    p[0,:] = particles['px']
    p[1,:] = particles['py']
    p[2,:] = particles['pz']

    new_r = np.matmul(M, r)
    new_p = np.matmul(M, p)

    data['t']=particles['t']

    for ii, name in enumerate(['x','y','z']):
        data[name] = new_r[ii,:]
        data['p'+name] = new_p[ii,:]
        
    return ParticleGroup(data=data)

    

    def max_energy_gain(z, Ez, w):

        return np.abs( np.trapezoid( Ez*np.exp(1j*w*z/c), z) )



