from gpt import tools, parsers
from gpt.particles import particle_stats, raw_data_to_particle_groups
import gpt.archive 

from pmd_beamphysics.units import pg_units

import h5py
import os
import tempfile
from time import time
import numpy as np

from gpt.parsers import parse_gpt_string

class GPT:
    """ 
    GPT simulation object. Essential methods:
    .__init__(...)
    .configure()
    .run()
    
    Input deck is held in .input
    Output data is parsed into .output
    .load_screens() will load particle data into .screen[...]
    
    The GPT binary file can be set on init. If it doesn't exist, configure will check the
        $GPT_BIN
    environmental variable.
    
    
    """
    
    def __init__(self,
                 input_file=None,
                 initial_particles = None,
                 gpt_bin='$GPT_BIN',      
                 use_tempdir=True,
                 workdir=None,
                 timeout=None,
                 verbose=False):

        # Save init
        self.original_input_file = input_file
        self.initial_particles = initial_particles
        self.use_tempdir = use_tempdir

        self.workdir = workdir

        if workdir:
            assert os.path.exists(workdir), 'workdir does not exist: '+workdir  
            self.workdir = os.path.abspath(workdir)  
                
        self.verbose=verbose
        self.gpt_bin = gpt_bin

        # These will be set
        self.log = []
        self.output = {}
        #self.screen = [] # list of screens
        self.timeout=timeout
        self.error = False
       
        # Run control
        self.finished = False
        self.configured = False
        self.using_tempdir = False
      
        # Call configure
        if input_file:
            self.load_input(input_file)
            self.configure()
        else:
            self.vprint('Warning: Input file does not exist. Not configured.')

    def configure(self):
        self.configure_gpt(workdir=self.workdir)
 
    def configure_gpt(self, input_filePath=None, workdir=None):
        """ Configure the GPT object """
        if input_filePath:
            self.load_input(input_filePath)
        
        # Check that binary exists
        self.gpt_bin = tools.full_path(self.gpt_bin)
        assert os.path.exists(self.gpt_bin), 'ERROR: GPT binary does not exist:'+self.gpt_bin
              
        # Set paths
        if self.use_tempdir:

            # Need to attach this to the object. Otherwise it will go out of scope.
            self.tempdir = tempfile.TemporaryDirectory(dir=workdir)
            self.path = self.tempdir.name

        elif(workdir):

            # Use the top level of the provided workdir
            self.path = workdir

        else:

            # Work in location of the template file
            self.path = self.original_path         

        self.input_file = os.path.join(self.path, self.original_input_file) 

        parsers.set_support_files(self.input['lines'], self.original_path)              
        
        self.vprint('GPT.configure_gpt:')
        self.vprint(f'   Original input file "{self.original_input_file}" in "{self.original_path}"')
        self.vprint(f'   Configured to run in "{self.path}"')

        self.configured = True

    def load_input(self, input_filePath, absolute_paths=True):
        """ Load the GPT template file """
        f = tools.full_path(input_filePath)
        self.original_path, self.original_input_file = os.path.split(f) # Get original path, filename
        self.input = parsers.parse_gpt_input_file(f)            

    def get_dist_file(self):
        """ Find the distribution input file name in the GPT file """ 
        for line in self.input['lines']:
            if('setfile' in line):
                return parse_gpt_string(line)[1]
     
    def set_dist_file(self,dist_file):
        """ Set the input distirbution file name in a GPT file """
        dist_file_set = False
        for ii, line in enumerate(self.input['lines']):
            if('setfile' in line):
                gpt_strs = parse_gpt_string(line)
                assert len(gpt_strs)==2, "Couldn't find distribution input file strs." 
                assert gpt_strs[0]=='beam', "Could not find beam defintion in setfile str."
                self.input['lines'][ii] = f'setfile("beam", "{dist_file}");'
                dist_file_set = True
        
        if(not dist_file_set):
            self.input['lines'].append(f'setfile("beam", "{dist_file}");')  

    def set_variable(self,variable,value):
        """ Set variable in the GPT input file to a new value """
        if(variable in self.input["variables"]):
            self.input['variables'][variable]=value
            return True
        else:
            return False

    def set_variables(self, variables):
        """ Set a list of variables (variable.keys) to new values (variables.values()) in the GPT Input file """
        return {var:self.set_variable(var,variables[var]) for var in variables.keys()}
    
    def load_output(self, file='gpt.out.gdf'):
        """ loads the GPT raw data and puts it into particle groups """

        self.vprint(f'   Loading GPT data from {self.get_gpt_output_file()}')
        touts, screens=parsers.read_gdf_file(file, self.verbose)  # Raw GPT data

        self.output['particles'] = raw_data_to_particle_groups(touts, screens, verbose=self.verbose) 
        self.output['n_tout'] = len(touts)
        self.output['n_screen'] = len(screens)

    @property
    def n_tout(self):
        return self.output['n_tout']
 
    @property
    def n_screen(self):
        return self.output['n_screen']

    @property
    def tout(self):
        """ Returns output particle groups for touts """
        if('particles' in self.output):
            return self.output['particles'][:self.output['n_tout']]

    @property
    def screen(self):
        """ Returns output particle groups for screens """
        if('particles' in self.output):
            return self.output['particles'][self.output['n_tout']:]

    @property
    def particles(self):
        """ Returns output particle groups for touts + screens """
        if('particles' in self.output):
            return self.output['particles']

    def trajectory(self, pid, data_type='tout'):

        if(data_type=='tout'):
            particle_groups = self.tout
        elif(data_type=='screen'):
            particle_groups = self.screen
        else:
            raise ValueError(f'GPT.trajectory got an unsupported data type = {data_type}.')

        #for pg in particle_groups:
        #    print(pg, pid in pg['id'])
        pgs_with_pid = [pg for pg in particle_groups if(pid in pg['id'])]

        if(len(pgs_with_pid)==0):
            return None

        variables = ['x', 'y', 'z', 'px', 'py', 'pz', 't']
   
        trajectory = {var:np.zeros( (len(pgs_with_pid),) ) for var in variables }    

        for ii, pg in enumerate(pgs_with_pid):
            for var in variables:
                trajectory[var][ii] = pg[var][pg['id']==pid]

        return trajectory
   

    def run(self, gpt_verbose=False):
        if not self.configured:
            print('not configured to run')
            return
        pass
        self.run_gpt(verbose=self.verbose, timeout=self.timeout, gpt_verbose=gpt_verbose)

        
    def get_run_script(self, write_to_path=True):
        """
        Assembles the run script. Optionally writes a file 'run' with this line to path.
        """
        
        _, infile = os.path.split(self.input_file)
        
        tokens = infile.split('.')
        if(len(tokens)>1):
            outfile = '.'.join(tokens[:-1])+'.out.gdf'
        else:
            outfile = tokens[0]+'.out.gdf'
        
        runscript = [self.gpt_bin, '-j1', '-v', '-o', self.get_gpt_output_file(), self.input_file]
            
        if write_to_path:
            with open(os.path.join(self.path, 'run'), 'w') as f:
                f.write(' '.join(runscript))
            
        return runscript

    def get_gpt_output_file(self):
        """ get the name of the GPT output file """
        path, infile = os.path.split(self.input_file)
        tokens = infile.split('.')
        if(len(tokens)>1):
            outfile = '.'.join(tokens[:-1])+'.out.gdf'
        else:
            outfile = tokens[0]+'.out.gdf'
        return os.path.join(path, outfile)

    def run_gpt(self, verbose=False, parse_output=True, timeout=None, gpt_verbose=False):
        """ RUN GPT and read in results """
        self.vprint('GPT.run_gpt:')

        run_info = {}
        t1 = time()
        run_info['start_time'] = t1

        if self.initial_particles:
            fname = self.write_initial_particles() 
            print(fname)

            # Link input file to new particle file
            self.set_dist_file(fname)
        
        init_dir = os.getcwd()
        self.vprint(f'   Running GPT...')

        # Write input file from internal dict
        self.write_input_file()
            
        runscript = self.get_run_script()

        try:

            if timeout:
               
                self.vprint(f'   Running with timeout = {self.timeout} sec.')
                kill_msgs = ["gpt: Spacecharge3Dmesh:", 
                             'Error:',
                             'gpt: No valid GPT license',
                             'malloc',
                             'Segmentation fault'
                            ]
                
                run_time, exception, log = tools.execute3(runscript, kill_msgs=kill_msgs, timeout=timeout, verbose=gpt_verbose)
                
                if(exception is not None):
                    self.error=True
                    run_info["error"]=True
                    run_info['why_error']=exception.strip()
            else:
                # Interactive output, for Jupyter
                log = []
                for path in tools.execute(runscript):
                    self.vprint(path, end="")
                    log.append(path)
    
            self.log = log
                    
            if parse_output:
                self.load_output(file=self.get_gpt_output_file())

        except Exception as ex:
            
            print('Run Aborted:', ex)
            self.error = True
            run_info['why_error'] = str(ex)
            
        finally:
  

            run_info['run_time'] = time() - t1
            run_info['run_error'] = self.error
            self.vprint(f'   Run finished, total time ellapsed: {run_info["run_time"]:G} (sec)')

            
            # Add run_info
            self.output.update(run_info)
                     
        
        self.finished = True
    
    def fingerprint(self):
        """
        Data fingerprint using the input. 
        """
        return tools.fingerprint(self.input)
                
    def vprint(self, *args, **kwargs):
        # Verbose print
        if self.verbose:
            print(*args, **kwargs)    
    
    def stat(self, key, data_type='all'):
        """
        Calculates any statistic that the ParticleGroup class can calculate, on all particle groups.
        """
        if(data_type=='all'):
            particle_groups = self.output['particles']

        elif(data_type=='tout'):
            particle_groups = self.tout

        elif(data_type=='screen'):
            particle_groups = self.screen

        else:
            raise ValueError(f'Unsupported GPT data type: {data_type}')

        return particle_stats(particle_groups, key)
    
    def stat_units(self, key):
        """
        Calculates any statistic that the ParticleGroup class can calculate, on all particle groups.
        """
        """Returns a str decribing the physical units of a stat key."""
        return pg_units(key)
    
    
    def write_input_file(self):
        """ Write the updated GPT input file """
        self.vprint(f'   Writing gpt input file to "{self.input_file}"')
        parsers.write_gpt_input_file(self.input, self.input_file)
   
    def write_initial_particles(self, fname=None):
        """ Write the initial particle data to file for use with GPT """
        if not fname:
            fname = os.path.join(self.path, 'gpt.particles.gdf')
        self.initial_particles.write_gpt(fname,asci2gdf_bin='$ASCI2GDF_BIN', verbose=False)
        self.vprint(f'   Initial particles written to "{fname}"')
        return fname 

    def load_initial_particles(self, h5):
        """Loads a openPMD-beamphysics particle h5 handle or file"""
        P = ParticleGroup(h5=h5)
        self.initial_particles = P

    
    def load_archive(self, h5=None):
        """
        Loads input and output from archived h5 file.
        
        See: GPT.archive
        """
        if isinstance(h5, str):
            h5 = os.path.expandvars(h5)
            g = h5py.File(h5, 'r')
            
            glist = gpt.archive.find_gpt_archives(g)
            n = len(glist)
            if n == 0:
                # legacy: try top level
                message = 'legacy'
            elif n == 1:
                gname = glist[0]
                message = f'group {gname} from'
                g = g[gname]
            else:
                raise ValueError(f'Multiple archives found in file {h5}: {glist}')
            
            self.vprint(f'Reading {message} archive file {h5}')
        else:
            g = h5
        
        self.input = gpt.archive.read_input_h5(g['input'])
        
        if 'initial_particles' in g:
            self.initial_particles=ParticleGroup(g['initial_particles'])        
        
        self.output = gpt.archive.read_output_h5(g['output'])
        
        self.vprint('Loaded from archive. Note: Must reconfigure to run again.')
        self.configured = False
        

    
    def archive(self, h5=None):
        """
        Archive all data to an h5 handle or filename.
        
        If no file is given, a file based on the fingerprint will be created.
        
        """
        if not h5:
            h5 = 'gpt_'+self.fingerprint()+'.h5'
         
        if isinstance(h5, str):
            h5 = os.path.expandvars(h5)
            g = h5py.File(h5, 'w')
            self.vprint(f'Archiving to file {h5}')
        else:
            # store directly in the given h5 handle
            g = h5
        
        # Write basic attributes
        gpt.archive.gpt_init(g)
        
        # All input
        gpt.archive.write_input_h5(g, self.input, name='input')

        if self.initial_particles:
            self.initial_particles.write(g, name='initial_particles')        
        
        # All output
        gpt.archive.write_output_h5(g, self.output, name='output')

        return h5        
        
        
    def __str__(self):

        outstr = '\nGPT object:'
        outstr = outstr+ "\n   Original input file: "+self.original_input_file
        outstr = outstr+ "\n   Template location: "+self.original_path
        if(self.workdir):
            outstr = outstr+ "\n   Top leve work dir: "+self.workdir
 
        if(self.use_tempdir):
            outstr = outstr+f"\n   Use temp directory: {self.use_tempdir}"
        #else:
        #    outstr = outstr+f"\n   Work directory: {self.path}"
       
        # Run control
        outstr = outstr+"\n\nRun Control"
        outstr = outstr+f"\n   Run configured: {self.configured}"
        outstr = outstr+f"\n   Work location: {self.path}"
        outstr = outstr+f"\n   Timeout: {self.timeout} (sec)"

        # Results
        outstr = outstr+"\n\nResults"
        outstr = outstr+f"\n   Finished: {self.finished}"
        outstr = outstr+f"\n   Error occured: {self.error}"
        if(self.error):
            outstr=outstr+f'\n   Cause: {self.output["why_error"]}'
            errline = self.get_syntax_error_line(self.output["why_error"])
            if(errline):
                outstr = outstr+f'\n   Suspected input file line: "{errline}"'
        rtime = self.output['run_time']
        outstr = outstr+f'\n   Run time: {rtime} (sec)'
        
        #outstr = outstr+f"\n
        #outstr = outstr+f'\n   Log: {self.log}\n'
        return outstr

    def get_syntax_error_line(self,error_msg):

        s=error_msg.strip().replace('\n','')
        if(s.endswith('Error: syntax error')):
            error_line_index = int(s[s.find("(")+1:s.find(")")])
            return self.input['lines'][error_line_index]
        else:
            return None
            

def run_gpt(settings=None, 
            initial_particles=None,
            gpt_input_file=None,
            use_tempdir=True, 
            workdir=None, 
            gpt_bin='$GPT_BIN', 
            timeout=2500, 
            auto_phase=False,
            verbose=False,
            gpt_verbose=False,
            asci2gdf_bin='$ASCI2GDF_BIN'):
    """
    Run GPT. 
    
        settings: dict with keys that can appear in a GPT input file. 
    """
    if verbose:
        print('run_gpt') 

    

    # Make GPT object
    G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, workdir=workdir, verbose=verbose, use_tempdir=use_tempdir)
    
    G.timeout=timeout
    G.verbose = verbose
      
    # Set inputs
    if settings:
        G.set_variables(settings)
  
    if(auto_phase):

        if(verbose):
            print('\nAuto Phasing >------\n')
        t1 = time.time()

        # Create the distribution used for phasing
        if(verbose):
            print('****> Creating intiial distribution for phasing...')

        phasing_beam = get_distgen_beam_for_phasing(beam, n_particle=10, verbose=verbose)
        phasing_particle_file = os.path.join(G.path, 'gpt_particles.phasing.gdf')
        write_gpt(phasing_beam, phasing_particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)
    
        if(verbose):
            print('<**** Created intiial distribution for phasing.\n')    

        G.write_input_file()   # Write the unphased input file

        phased_file_name, phased_settings = gpt_phasing(G.input_file, path_to_gpt_bin=G.gpt_bin[:-3], path_to_phasing_dist=phasing_particle_file, verbose=verbose)
        G.set_variables(phased_settings)
        t2 = time.time()

        if(verbose):
            print(f'Time Ellapsed: {t2-t1} sec.')
            print('------< Auto Phasing\n')


    # Run
    G.run(gpt_verbose=gpt_verbose)
    
    return G  

