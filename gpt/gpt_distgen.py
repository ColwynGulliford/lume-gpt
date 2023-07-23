from gpt import GPT
from . import tools
from .gpt import run_gpt

from .tools import full_path

from .merit import default_gpt_merit

from distgen import Generator   
from distgen.writers import write_gpt
from distgen.tools import update_nested_dict

from gpt.gpt_phasing import gpt_phasing

from h5py import File
import yaml
import os
import time



from gpt.tools import DEFAULT_KILL_MSGS 

def set_gpt_and_distgen(gpt, distgen_input, settings, verbose=False):
    """
    Searches gpt and distgen input for keys in settings, and sets their values to the appropriate input.
    """
    for k, v in settings.items():
        found=gpt.set_variable(k,v)
        #print(k,v,found)
        if verbose and found:
            print(k, 'is in gpt')
        
        if not found:
            distgen_input = update_nested_dict(distgen_input, {k:v}, verbose=bool(verbose))
            #set_nested_dict(distgen_input, k, v)    
    
    return gpt, distgen_input

def centroid(particle_group):
    good = particle_group.status == 1
    pg = particle_group[good]
    data = {key:pg.avg(key) for key in ['x', 'px', 'y', 'py', 'z', 'pz', 't']}
    data['species'] = pg.species
    data['weight'] = pg.charge
    data['status'] = 1
    return ParticleGroup(data=data)


def phase_gpt_with_distgen(settings=None,
                         gpt_input_file=None,
                         distgen_input_file=None,
                         workdir=None, 
                         use_tempdir=True,
                         gpt_bin='$GPT_BIN',
                         timeout=2500,
                         #auto_phase=False,
                         verbose=False,
                         gpt_verbose=False,
                         asci2gdf_bin='$ASCI2GDF_BIN',
                         kill_msgs=DEFAULT_KILL_MSGS,
                         load_fields=False,
                         parse_layout=True):

    # Call simpler evaluation if there is no generator:
    if not distgen_input_file:
        return run_gpt(settings=settings, 
                       gpt_input_file=gpt_input_file, 
                       workdir=workdir,
                       use_tempdir=use_tempdir,
                       gpt_bin=gpt_bin, 
                       timeout=timeout, 
                       verbose=verbose,
                       kill_msgs=kill_msgs,
                       load_fields=load_fields)
    
    if(verbose):
        print('Run GPT with Distgen:') 

    # Make gpt and generator objects
    G = GPT(gpt_bin=gpt_bin, 
        input_file=gpt_input_file,
        workdir=workdir, 
        use_tempdir=use_tempdir,
        kill_msgs=kill_msgs,
        load_fields=load_fields,
        parse_layout=parse_layout)

    
    G.timeout=timeout
    G.verbose = verbose

    # Distgen generator
    gen = Generator(verbose=verbose)
    f = tools.full_path(distgen_input_file)
    distgen_params = yaml.safe_load(open(f))

    # Set inputs
    if settings:
        G, distgen_params = set_gpt_and_distgen(G, distgen_params, settings, verbose=verbose)
    
    # Link particle files
    particle_file = tools.full_path(os.path.join(G.path, os.path.basename(G.get_dist_file())))
    phasing_particle_file = particle_file.replace('.gdf', '.phasing.gdf')

    if(verbose):
        print('Linking particle files, distgen output will point to -> "'+os.path.basename(particle_file)+'" in working directory.')

    G.set_dist_file(os.path.basename(particle_file))

    if('output' in distgen_params and verbose):
        print('Replacing Distgen output params')

    distgen_params['output'] = {'type':'gpt','file':particle_file}

    if(verbose):
        print('\nDistgen >------\n')
    # Configure distgen
    gen.parse_input(distgen_params)   
     
    # Attach distgen input. This is non-standard. Used for archiving
    G.distgen_input = gen.input        

    beam = gen.beam()

    if(verbose):
        print('------< Distgen\n')

    if(os.path.exists(particle_file)):
        
        if(os.path.islink(particle_file)):
            os.unlink(particle_file)
        else:
            os.remove(particle_file)

    write_gpt(beam, particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)

    if(verbose):
        print('\nAuto Phasing >------\n')
    t1 = time.time()

    # Create the distribution used for phasing
    if(verbose):
        print('****> Creating intiial distribution for phasing...')

    phasing_beam = get_distgen_beam_for_phasing(beam, n_particle=10, verbose=verbose)
    write_gpt(phasing_beam, phasing_particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)
    
    if(verbose):
        print('<**** Created intiial distribution for phasing.\n')    

    G.write_input_file()   # Write the unphased input file
       
    phased_file_name, phased_settings = gpt_phasing(G.input_file, 
                                                    path_to_gpt_bin=G.gpt_bin[:-3], 
                                                    path_to_phasing_dist=phasing_particle_file, 
                                                    verbose=verbose)

    G.set_variables(phased_settings)
    t2 = time.time()

    if(verbose):
        print(f'Time Ellapsed: {t2-t1} sec.')
        print('------< Auto Phasing\n')

    return G, phased_settings

    
def run_gpt_with_distgen(settings=None,
                         gpt_input_file=None,
                         distgen_input_file=None,
                         workdir=None, 
                         use_tempdir=True,
                         gpt_bin='$GPT_BIN',
                         timeout=2500,
                         auto_phase=False,
                         verbose=False,
                         gpt_verbose=False,
                         asci2gdf_bin='$ASCI2GDF_BIN',
                         kill_msgs=DEFAULT_KILL_MSGS,
                         load_fields=False,
                         parse_layout=True
                        ):
    """
    Run gpt with particles generated by distgen. 
    
        settings: dict with keys that can appear in an gpt or distgen Generator input file. 
        
    Example usage:
        G = run_gpt_with_distgen({'lspch':False},
                       gpt_input_file='$LCLS_LATTICE/gpt/models/gunb_eic/gpt.in',
                       distgen_input_file='$LCLS_LATTICE/distgen/models/gunb_gaussian/gunb_gaussian.json',
                       verbose=True,
                       timeout=None
                      )        
        
    """

    # Call simpler evaluation if there is no generator:
    if not distgen_input_file:
        return run_gpt(settings=settings, 
                       gpt_input_file=gpt_input_file, 
                       workdir=workdir,
                       use_tempdir=use_tempdir,
                       gpt_bin=gpt_bin, 
                       timeout=timeout, 
                       verbose=verbose,
                       kill_msgs=kill_msgs,
                       load_fields=load_fields)
    
    if(verbose):
        print('Run GPT with Distgen:') 

    # Make gpt and generator objects
    G = GPT(gpt_bin=gpt_bin, 
        input_file=gpt_input_file,
        workdir=workdir, 
        use_tempdir=use_tempdir,
        kill_msgs=kill_msgs,
        load_fields=load_fields,
        parse_layout=parse_layout)

    
    G.timeout=timeout
    G.verbose = verbose

    # Distgen generator
    gen = Generator(verbose=verbose)
    f = tools.full_path(distgen_input_file)
    distgen_params = yaml.safe_load(open(f))

    # Set inputs
    if settings:
        G, distgen_params = set_gpt_and_distgen(G, distgen_params, settings, verbose=verbose)
    
    # Link particle files
    particle_file = tools.full_path(os.path.join(G.path, os.path.basename(G.get_dist_file())))
    phasing_particle_file = particle_file.replace('.gdf', '.phasing.gdf')

    if(verbose):
        print('Linking particle files, distgen output will point to -> "'+os.path.basename(particle_file)+'" in working directory.')

    G.set_dist_file(os.path.basename(particle_file))

    if('output' in distgen_params and verbose):
        print('Replacing Distgen output params')

    distgen_params['output'] = {'type':'gpt','file':particle_file}

    if(verbose):
        print('\nDistgen >------\n')
    # Configure distgen
    gen.parse_input(distgen_params)   
     
    # Attach distgen input. This is non-standard. Used for archiving
    G.distgen_input = gen.input        

    beam = gen.beam()

    if(verbose):
        print('------< Distgen\n')

    if(os.path.exists(particle_file)):
        
        if(os.path.islink(particle_file)):
            os.unlink(particle_file)
        else:
            os.remove(particle_file)

    write_gpt(beam, particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)

    #print(beam['x'].mean())
    #print(beam['y'].mean())
    #print(beam['z'].mean())
    #print(beam['px'].mean())
    #print(beam['py'].mean())
    #print(beam['pz'].mean())
    #print(beam['t'].mean())

    if(auto_phase): 

        if(verbose):
            print('\nAuto Phasing >------\n')
        t1 = time.time()

        # Create the distribution used for phasing
        if(verbose):
            print('****> Creating intiial distribution for phasing...')

        phasing_beam = get_distgen_beam_for_phasing(beam, n_particle=10, verbose=verbose)
        write_gpt(phasing_beam, phasing_particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)
    
        if(verbose):
            print('<**** Created intiial distribution for phasing.\n')    

        G.write_input_file()   # Write the unphased input file
       
        phased_file_name, phased_settings = gpt_phasing(G.input_file, 
                                                        path_to_gpt_bin=G.gpt_bin[:-3], 
                                                        path_to_phasing_dist=phasing_particle_file, 
                                                        verbose=verbose)

        G.set_variables(phased_settings)
        t2 = time.time()

        if(verbose):
            print(f'Time Ellapsed: {t2-t1} sec.')
            print('------< Auto Phasing\n')


    # If here, either phasing successful, or no phasing requested
    G.run(gpt_verbose=gpt_verbose)
    
    return G

def evaluate_gpt_with_distgen(settings, 
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
             asci2gdf_bin='$ASCI2GDF_BIN',
             kill_msgs=DEFAULT_KILL_MSGS
             ):    
    """
    Simple evaluate GPT.
    
    Similar to run_astra_with_distgen, but returns a flat dict of outputs. 
    
    Will raise an exception if there is an error. 
    
    """
    
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
                             asci2gdf_bin=asci2gdf_bin,
                             kill_msgs=kill_msgs)
    
    if merit_f:
        merit_f = tools.get_function(merit_f)
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)
    
    if output['error']:
        raise ValueError('error occured!')
        
    #Recreate Generator object for fingerprint, proper archiving
    # TODO: make this cleaner
    gen = Generator(G.distgen_input)
    #gen = Generator()
    #gen.input = G.distgen_input    
    
    fingerprint = fingerprint_gpt_with_distgen(G, gen)
    output['fingerprint'] = fingerprint    
    
    if archive_path:
        path = tools.full_path(archive_path)
        assert os.path.exists(path), f'archive path does not exist: {path}'
        archive_file = os.path.join(path, fingerprint+'.h5')
        output['archive'] = archive_file
        
        # Call the composite archive method
        archive_gpt_with_distgen(G, gen, archive_file=archive_file)          
        
    return output

def get_distgen_beam_for_phasing(beam, n_particle=10, verbose=False):

    variables = ['x', 'y', 'z','px', 'py', 'pz', 't']

    transforms = { f'avg_{var}':{'type': f'set_avg {var}', f'avg_{var}': { 'value': beam.avg(var).magnitude, 'units':  str(beam.avg(var).units)  } } for var in variables }
    #for var in variables:
    #  
    #    avg_var = beam.avg(var)
    #    transforms[f'set avg {var}'] = {'variables':var, 'type': 'set_avg', 
    #                                    f'avg_{var}': {'value': float(avg_var.magnitude), 'units': str(avg_var.units) }} 

    phasing_distgen_input = {'n_particle':10, 'random_type':'hammersley', 'transforms':transforms,
                             'total_charge':{'value':0.0, 'units':'C'},
                             'start': {'type':'time', 'tstart':{'value': 0.0, 'units': 's'}},}
    
    gen = Generator(phasing_distgen_input, verbose=verbose) 
    pbeam = gen.beam()

    return pbeam


def fingerprint_gpt_with_distgen(gpt_object, distgen_object):
    """
    Calls fingerprint() of each of these objects
    """
    f1 = gpt_object.fingerprint()
    f2 = distgen_object.fingerprint()
    d = {'f1':f1, 'f2':2}
    return tools.fingerprint(d)



def archive_gpt_with_distgen(gpt_object,
                             distgen_object,
                             archive_file=None,
                             gpt_group ='gpt',
                             distgen_group ='distgen'):
    """
    Creates a new archive_file (hdf5) with groups for 
    gpt and distgen. 
    
    Calls .archive method of GPT and Distgen objects, into these groups.
    """
    
    h5 = File(archive_file, 'w')
    
    #fingerprint = tools.fingerprint(astra_object.input.update(distgen.input))
    
    g = h5.create_group(distgen_group)
    distgen_object.archive(g)
    
    g = h5.create_group(gpt_group)
    gpt_object.archive(g)
    
    h5.close()