"""
Archiving functions
"""
from pmd_beamphysics import ParticleGroup

import numpy as np


def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)



def opmd_init(h5, basePath='/screen/%T/', particlesPath='/' ):
    """
    Root attribute initialization.
    
    h5 should be the root of the file.
    """
    d = {
        'basePath':basePath,
        'dataType':'openPMD',
        'openPMD':'2.0.0',
        'openPMDextension':'BeamPhysics;SpeciesType',
        'particlesPath':particlesPath    
    }
    for k,v in d.items():
        h5.attrs[k] = fstr(v)
        
        
#-------------------------------------
# input read/write
        
# Write input file to dataset
def write_input_h5(h5, gpt_input, name='input'):
    """
    gpt_input is a dict with:
       lines
       variables
    Writes thes to h5 in new group 'name'
    
    See: read_input_h5
    """
    g = h5.create_group(name)
    g.attrs['lines'] = gpt_input['lines']
    
    g2 = g.create_group('variables')
    for k, v in gpt_input['variables'].items():
        g2.attrs[k] = v

        # Write input file to dataset
        
def read_input_h5(h5):
    """
    h5 should have:
        .attrs['lines'] = list of lines
       group 'variables' with .attrs that are the variables

    See: write_input_h5
    """
    gpt_input = {}
    gpt_input['lines'] = list(h5.attrs['lines'])
    gpt_input['variables'] = dict(h5['variables'].attrs)

    return gpt_input



#-------------------------------------
# output read/write
    
def write_output_h5(h5, gpt_output, name='output'):
    """
    Writes all output to h5 in new group with name. 
    
    For now, only writes gpt_output['particles']
    """
    g = h5.create_group(name)

    write_particles_h5(g, gpt_output['particles'], name='particles')

    
def read_output_h5(h5):
    """
    Reads output and returns a dict. 
    
    See: write_output_h5
    
    """
    gpt_output = {}
    gpt_output['particles'] = read_particles_h5(h5['particles'])
    return gpt_output    
    
    
def write_particles_h5(h5, particles, name='screen'):
    """
    Write all screens to file, simply named by their index
    
    See: read_particles_h5
    """
    g = h5.create_group(name)
    
    # Set base attributes
    opmd_init(h5, basePath='/'+name+'/%T/', particlesPath='/' )
    
    # Loop over screens
    for i, particle_group in enumerate(particles):
        name = str(i)        
        particle_group.write(g, name=name)  
        
        
def read_particles_h5(h5):
    """
    Reads particles from h5
    
    See: write_particles_h5
    """
    # This should be a list of '0', '1', etc.
    # Cast to int, sort, reform to get the list order correct.
    ilist = sorted([int(x) for x in list(h5)])
    glist = [str(i) for i in ilist]
    
    return [ParticleGroup(h5=h5[g]) for g in glist]        