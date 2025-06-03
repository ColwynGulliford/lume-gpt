import easygdf
import time
import numpy as np
import re
import os

from gpt.tools import full_path

import shutil

# ------ Number parsing ------
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def find_path(line, pattern=r'"([^"]+\.gdf)"'):
    matches=re.findall(pattern, line)
    return matches
 
  
 
def set_support_files_orig(lines, 
                           original_path, 
                           target_path='', 
                           copy_files=False, 
                           pattern=r'"([^"]+\.gdf)"', 
                           verbose=False):

    print('set SUPPORT ORIG')
    
    for ii, line in enumerate(lines):

        support_files = find_path(line, pattern=pattern)
        
        for support_file in support_files:

            #print(full_path(support_file))

            abs_original_path = full_path( os.path.join(original_path, os.path.expandvars(support_file)) )
            
            if(copy_files):
            
                abs_target_path = os.path.join(target_path, support_file) 
                shutil.copyfile(abs_original_path, abs_target_path, follow_symlinks=True)            

                if(verbose):
                    print("Copying file: ", abs_original_path,'->',abs_target_path)   

            else:

                if(os.path.isfile(abs_original_path)):
                    lines[ii] = line.replace(support_file, abs_original_path)
                    if(verbose):
                        print("Set path to file: ",lines[ii])        
                        
def set_support_files(lines, 
                      original_path, 
                      target='', 
                      copy_files=False, 
                      pattern=r'"([^"]+\.gdf)"',
                      verbose=False):
    
    for ii, line in enumerate(lines):

        support_files = find_path(line, pattern=pattern)
        
        for support_file in support_files:

            abs_original_path = full_path( os.path.join(original_path, os.path.expandvars(support_file)) )
            
            if(copy_files):
                #print(support_file, target)
                abs_target_path = os.path.join(target, os.path.basename(support_file))

                if(os.path.exists(abs_original_path) and abs_original_path!=abs_target_path):

                    try:
                        shutil.copyfile(abs_original_path, abs_target_path, follow_symlinks=True)  
                    except shutil.SameFileError:
                        pass
                    lines[ii] = line.replace(support_file, os.path.basename(abs_target_path))
                    
                elif(abs_original_path!=abs_target_path and not os.path.exists(abs_target_path)):
                    print('Warning: possible missing support file: ', abs_original_path)


                if(verbose):
                    print("Copying file: ", abs_original_path,'->',abs_target_path)   

            else:

                if(os.path.isfile(abs_original_path)):
                    
                    #print(target, abs_original_path)
                    dest = full_path( os.path.join(target, os.path.basename(abs_original_path)) )

                    # Make symlink
                    # Replace old symlinks. 
                    if os.path.islink(dest):
                        os.unlink(dest)
                    elif os.path.exists(dest):
                        
                        if verbose:
                            print(dest, 'exists, will not symlink')

                        continue

                    # Note that the following will raise an error if the dest is an actual file that exists    
                    os.symlink(abs_original_path, dest)
                    if verbose:
                        print('Linked', abs_original_path, 'to', os.path.basename(dest) )

                    lines[ii] = line.replace(support_file, os.path.basename(dest))


def parse_gpt_input_file(filePath, condense=False, verbose=False):
    """
    Parses GPT input file 
    """

    finput={}

    with open(filePath, 'r') as f:

        clean_lines = []

        # Get lines without comments
        for line in f:
            tokens = line.strip().split('#')
            if(len(tokens[0])>0):
                clean_line = tokens[0].strip().replace('\n', '')
                clean_lines.append(clean_line)

    variables={}

    for ii,line in enumerate(clean_lines):
      
        tokens = line.split("=")

        if(len(tokens)==2 and isfloat(tokens[1][:-1].strip())):
 
            if(len(tokens[0].split(")"))>1):
                name = tokens[0].split(")")[-1].strip()
            else:
                name = tokens[0].strip()

            value = float(tokens[1][:-1].strip())
            
            if(name not in variables.keys()):
                variables[name]=value 
            elif(verbose):
                print(f'Warning: multiple definitions of variable {name} on line {ii}.')

    support_files={}
    for ii, line in enumerate(clean_lines):
        for sfile in find_path(line):
            if(sfile not in support_files):
                support_files[ii]=sfile
                
        
    finput['lines']=clean_lines
    finput['variables']=variables
    finput['support_files'] = support_files

    return finput


def write_gpt_input_file(finput, inputFile, ccs_beg='wcs'):

    #print(inputFile)
    for var in finput['variables'].keys():

        value=finput['variables'][var]
        for index, line in enumerate(finput['lines']):
            tokens = line.split('=')
            if(len(tokens)==2 and tokens[0].strip()==var):
            #if(len(tokens)==2 and tokens[0].strip().endswith(var)):
                finput["lines"][index]=f'{tokens[0].strip()[:-len(var)]}{var}={value};'
                #print(finput["lines"][index])
                break
            
    with open(inputFile,'w') as f:

        for line in finput["lines"]:
            f.write(line+"\n")

        if(ccs_beg!="wcs"):
            f.write(f'settransform("{ccs_beg}", 0,0,0, 1,0,0, 0,1,0, "beam");\n')

def read_particle_gdf_file(gdffile, 
                           verbose=0.0, 
                           extra_screen_keys=['q','nmacro'], 
                           load_files=False): #,'ID', 'm']):

    with open(gdffile, 'rb') as f:
        data = easygdf.load_initial_distribution(f, extra_screen_keys=extra_screen_keys)

    screen = {}
    n = len(data[0,:])
    if(n>0):

        q = data[7,:]          # elemental charge/macroparticle
        nmacro = data[8,:]     # number of elemental charges/macroparticle
                   
        weights = np.abs(data[7,:]*data[8,:])/np.sum(np.abs(data[7,:]*data[8,:]))

        screen = {"x":data[0,:],"GBx":data[1,:],
                  "y":data[2,:],"GBy":data[3,:],
                  "z":data[4,:],"GBz":data[5,:],
                  "t":data[6,:],
                  "q":data[7,:],
                  "nmacro":data[8,:],
                  "w":weights,
                  "G":np.sqrt(data[1,:]*data[1,:]+data[3,:]*data[3,:]+data[5,:]*data[5,:]+1)}
                
                    #screen["Bx"]=screen["GBx"]/screen["G"]
                    #screen["By"]=screen["GBy"]/screen["G"]
                    #screen["Bz"]=screen["GBz"]/screen["G"]

        screen["time"]=np.sum(screen["w"]*screen["t"])
        screen["n"]=n          

    return screen

def read_gdf_file(gdffile, verbose=False, load_fields=False, spin_tracking=False):
      
    # Read in file:

  
    #self.vprint("Current file: '"+data_file+"'",1,True)
    #self.vprint("Reading data...",1,False)
    t1 = time.time()

    extra_tout_keys = ['q', 'nmacro', 'ID', 'm']
    extra_screen_keys = ['q', 'nmacro', 'ID', 'm']
    
    with open(gdffile, 'rb') as f:
        
        if load_fields:
            extra_tout_keys = extra_tout_keys + ['fEx', 'fEy', 'fEz', 'fBx', 'fBy', 'fBz']

        if spin_tracking:
            extra_tout_keys = extra_tout_keys + ['spinx', 'spiny', 'spinz', 'sping']
            extra_screen_keys = extra_screen_keys + ['spinx', 'spiny', 'spinz', 'sping']

        touts, screens = easygdf.load(f, extra_screen_keys=extra_screen_keys, extra_tout_keys=extra_tout_keys)

    t2 = time.time()
    if(verbose):
        print(f'   GDF data loaded, time ellapsed: {t2-t1:G} (sec).')
            
    #self.vprint("Saving wcs tout and ccs screen data structures...",1,False)

    tdata = make_tout_dict(touts, load_fields=load_fields, spin_tracking=spin_tracking)
    pdata = make_screen_dict(screens, spin_tracking=spin_tracking)

    return (tdata, pdata)




def make_tout_dict(touts, load_fields=False, spin_tracking=False):

    tdata=[]
    count = 0

    spin_index = 11 + int(load_fields) * 6
    
    for data in touts:
        n=len(data[0,:])
        
        if(n>0):

            q = data[7,:]       # elemental charge/macroparticle
            nmacro = data[8,:]  # number of elemental charges/macroparticle
                    
            if(np.sum(q)==0 or np.sum(nmacro)==0):
                weights = data[10,:]/np.sum(data[10,:])  # Use the mass if no charge is specified
            else:
                weights = np.abs(data[7,:]*data[8,:])/np.sum(np.abs(data[7,:]*data[8,:]))

            tout = {"x":data[0,:],"GBx":data[1,:],
                    "y":data[2,:],"GBy":data[3,:],
                    "z":data[4,:],"GBz":data[5,:],
                    "t":data[6,:],
                    "q":data[7,:],
                    "nmacro":data[8,:],
                    "ID":data[9,:],
                    "m":data[10,:],
                    "w":weights,
                    "G":np.sqrt(data[1,:]*data[1,:]+data[3,:]*data[3,:]+data[5,:]*data[5,:]+1)}

            #tout["Bx"]=tout["GBx"]/tout["G"]
            #tout["By"]=tout["GBy"]/tout["G"]
            #tout["Bz"]=tout["GBz"]/tout["G"]

            tout["time"]=np.sum(tout["w"]*tout["t"])
            tout["n"]=len(tout["x"])
            tout["number"]=count
            
            if load_fields:
                tout['Ex'] = data[11,:]
                tout['Ey'] = data[12,:]
                tout['Ez'] = data[13,:]
                tout['Bx'] = data[14,:]
                tout['By'] = data[15,:]
                tout['Bz'] = data[16,:]

            if spin_tracking:
                tout['sx'] = data[spin_index,:]
                tout['sy'] = data[spin_index+1,:]
                tout['sz'] = data[spin_index+2,:]

            count=count+1
            tdata.append(tout)

    return tdata

def make_screen_dict(screens, spin_tracking=False):

    pdata=[]
         
    count=0
    for data in screens:
        n = len(data[0,:])
        if(n>0):

            q = data[7,:]          # elemental charge/macroparticle
            nmacro = data[8,:]     # number of elemental charges/macroparticle
                   
            if(np.sum(q)==0 or np.sum(nmacro)==0):
                weights = data[10,:]/np.sum(data[10,:])  # Use the mass if no charge is specified
            else:
                weights = np.abs(data[7,:]*data[8,:])/np.sum(np.abs(data[7,:]*data[8,:]))

            screen = {"x":data[0,:],"GBx":data[1,:],
                      "y":data[2,:],"GBy":data[3,:],
                      "z":data[4,:],"GBz":data[5,:],
                      "t":data[6,:],
                      "q":data[7,:],
                      "nmacro":data[8,:],
                      "ID":data[9,:],
                      "m":data[10,:],
                      "w":weights,
                      "G":np.sqrt(data[1,:]*data[1,:]+data[3,:]*data[3,:]+data[5,:]*data[5,:]+1)}
                
                    #screen["Bx"]=screen["GBx"]/screen["G"]
                    #screen["By"]=screen["GBy"]/screen["G"]
                    #screen["Bz"]=screen["GBz"]/screen["G"]

            if spin_tracking:
                screen['sx'] = data[11,:]
                screen['sy'] = data[12,:]
                screen['sz'] = data[13,:]

            screen["time"]=np.sum(screen["w"]*screen["t"])
            screen["n"]=n
            screen["number"]=count

            count=count+1
            pdata.append(screen)
                    
    t2 = time.time()
    #self.vprint("done. Time ellapsed: "+self.ptime(t1,t2)+".",0,True)

    ts=np.array([screen['time'] for screen in pdata])
    sorted_indices = np.argsort(ts)
    return [pdata[sii] for sii in sorted_indices]

def parse_gpt_string(line):
    return re.findall(r'\"(.+?)\"',line) 

def replace_gpt_string(line,oldstr,newstr):

    strs = parse_gpt_string(line)
    assert oldstr in strs, 'Could not find string '+oldstr+' for string replacement.'
    line.replace(oldstr,newstr)

