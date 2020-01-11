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
