import numpy as np
import math
import cmath
from scipy import optimize

c = 299792458

def load_map(filename, center=True, zscale=1):
    
    data = np.loadtxt(filename) 
    z = data[:,0]
    E = data[:,2]
    
    if(center):
        z = center_map(z, E, zscale=zscale)
    
    return (z,E)
    
def center_map(z, E, zscale=1):
    
    zc = z[E==np.max(E)]
    return (z-zc)*zscale

def integrateEz(z, E, phi, f, s=1):
    
    phi = phi*math.pi/180
    w = 2*math.pi*f
    phase = w*z/c + phi
    
    return np.trapz(z, E*np.exp(s*np.complex(0,1)*phase))

def integrateReEz(z, E, phi, f):
    return np.real(integrateEz(z, E, phi, f))

def minus_integrateReEz(phi, z, E, f):
    return -integrateReEz(z, E, phi, f)

def get_oncrest_phase(z, E, f):
    return (-cmath.phase(integrateEz(z, E, 0, f))*180/math.pi)%360

def save_map(z, E, filename):
    
    data = np.zeros( (len(z), 2) )
    data[:,0] = z
    data[:,1] = E
    
    np.savetxt(filename, data, header='z Ez', comments=' ')
