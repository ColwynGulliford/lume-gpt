from gpt import tools, parsers
from gpt.particles import touts_to_particlegroups, particle_stats
import gpt.archive 

from pmd_beamphysics.units import pg_units

import h5py
import os
import tempfile
from time import time



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
                 gpt_bin='$GPT_BIN',      
                 use_tempdir=True,
                 workdir=None,
                 verbose=False):

        # Save init
        self.original_input_file = input_file
        self.use_tempdir = use_tempdir
        self.workdir = workdir
        if workdir:
            assert os.path.exists(workdir), 'workdir does not exist: '+workdir        
        self.verbose=verbose
        self.gpt_bin = gpt_bin

        # These will be set
        self.log = []
        self.output = {}
        #self.screen = [] # list of screens
        self.timeout=None
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
        else:
            # Work in place
            self.path = self.original_path            

        self.input_file = os.path.join(self.path, self.original_input_file) 
        
        parsers.set_support_files(self.input['lines'],self.original_path)              
        
        self.vprint(f'Configured to run in {self.path}')
        self.configured = True

    def load_input(self, input_filePath, absolute_paths=True):
        f = tools.full_path(input_filePath)
        self.original_path, self.original_input_file = os.path.split(f) # Get original path, filename
        self.input = parsers.parse_gpt_input_file(f)            
            
    def set_variable(self,variable,value):
        if(variable in self.input["variables"]):
            self.input['variables'][variable]=value
            return True
        else:
            return False

    def set_variables(self, variables):
        return [self.set_variable(var,variables[var]) for var in variables.keys()]
    
    def load_output(self, file='gpt.out.gdf'):

        touts, screens=parsers.read_gdf_file(file)
        
        # Raw GPT 
        self.screen = screens
        self.tout = touts
        
        # particles as ParticleGroup objects
        self.output['particles'] = touts_to_particlegroups(touts)
        

    def run(self):
        if not self.configured:
            print('not configured to run')
            return
        pass
        self.run_gpt(verbose=self.verbose, timeout=self.timeout)
        
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
        
        runscript = [self.gpt_bin, '-v -o', self.get_gpt_output_file(), infile]
            
        if write_to_path:
            with open(os.path.join(self.path, 'run'), 'w') as f:
                f.write(' '.join(runscript))
            
        return runscript

    def get_gpt_output_file(self):
        path, infile = os.path.split(self.input_file)
        tokens = infile.split('.')
        if(len(tokens)>1):
            outfile = '.'.join(tokens[:-1])+'.out.gdf'
        else:
            outfile = tokens[0]+'.out.gdf'
        return os.path.join(path, outfile)

    def run_gpt(self, verbose=False, parse_output=True, timeout=None):
        
        run_info = {}
        t1 = time()
        run_info['start_time'] = t1
        
        # Move to local directory
        # Save init dir
        init_dir = os.getcwd()
        self.vprint('init dir: ', init_dir)

        os.chdir(self.path)
        # Debugging
        self.vprint('running GPT in '+os.getcwd())#

        # Write input file from internal dict
        self.write_input_file()
            
        runscript = self.get_run_script()

        try:
            if timeout:
                #print("Running with timeout")
                kill_msgs = ["gpt: Spacecharge3Dmesh:", 
                             'Error:','gpt: No valid GPT license']
                
                run_time,exception,log = tools.execute3(runscript, kill_msgs=kill_msgs, timeout=timeout)
                
                if(exception is not None):
                    self.error=True
                    run_info["error"]=True
                    run_info['why_error']=exception
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
            
            # Add run_info
            self.output.update(run_info)
            
            # Return to init_dir
            os.chdir(init_dir)                        
        
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
    
    def stat(self, key):
        """
        Calculates any statistic that the ParticleGroup class can calculate, on all particle groups.
        """
        particle_groups = self.output['particles']
        return particle_stats(particle_groups, key)
    
    def stat_units(self, key):
        """Returns a str decribing the physical units of a stat key."""
        return pg_units(key)
    
    
    def write_input_file(self):
        parsers.write_gpt_input_file(self.input, self.input_file)
   
    
    def load_archive(self, h5=None):
        """
        Loads input and output from archived h5 file.
        
        See: GPT.archive
        """
        if isinstance(h5, str):
            g = h5py.File(h5, 'r')
            self.vprint(f'Reading archive file {h5}')
        else:
            g = h5
        
        self.input = gpt.archive.read_input_h5(g['input'])
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
            g = h5py.File(h5, 'w')
            self.vprint(f'Archiving to file {h5}')
        else:
            g = h5
        
        # All input
        gpt.archive.write_input_h5(g, self.input, name='input')

        # All output
        gpt.archive.write_output_h5(g, self.output, name='output')

        return h5        
        
        
    def __str__(self):

        outstr = 'GPT object:'
        outstr = outstr+ "\n   Original input file: "+self.original_input_file
        outstr = outstr+f"\n   Use temp directory: {self.use_tempdir}"
        outstr = outstr+f"\n   Work directory: {self.workdir}"
       
        # Run control
        outstr = outstr+"\n\nRun Control"
        outstr = outstr+f"\n   Run configured: {self.configured}"
        outstr = outstr+f"\n   Using temp directory: {self.using_tempdir}"
        outstr = outstr+f"\n   Timeout: {self.timeout} (sec)"

        # Results
        outstr = outstr+"\n\nResults"
        outstr = outstr+f"\n   Finished: {self.finished}"
        outstr = outstr+f"\n   Error occured: {self.error}"
        rtime = self.output['run_time']
        outstr = outstr+f'\n   Run time: {rtime} (sec)'
        #outstr = outstr+f"\n
        #outstr = outstr+f'\n   Log: {self.log}\n'
        return outstr


def run_gpt(settings=None, 
            gpt_input_file=None, 
            workdir=None, 
            gpt_bin='$GPT_BIN', 
            timeout=2500, 
            verbose=False):
    """
    Run GPT. 
    
        settings: dict with keys that can appear in a GPT input file. 
    """
    if verbose:
        print('run_gpt') 

    # Make GPT object
    G =GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, workdir=workdir)
    
    G.timeout=timeout
    G.verbose = verbose
      
    # Set inputs
    if settings:
        G.set_variables(settings)
        #set_astra(A.input, {}, settings, verbose=verbose)
            
    # Run
    G.run()
    
    return G  

