import copy
from . import easygdf
import time
import numpy as np
import re
import os

import shutil

# ------ Number parsing ------
def isfloat(value):
      try:
            float(value)
            return True
      except ValueError:
            return False

def find_path(line, pattern=r'"([^"]+\.gdf)"'):

    matches=re.findall(pattern,line)
    return matches
 
def set_support_files(lines, original_path, target_path='', copy_files=False, pattern=r'"([^"]+\.gdf)"', verbose=False):

    for ii, line in enumerate(lines):

        #print(line)
        support_files = find_path(line,pattern=pattern)

        for support_file in support_files:

            abs_original_path = os.path.join(original_path, support_file)

            if(copy_files):
            
                abs_target_path = os.path.join(target_path, support_file) 
                shutil.copyfile(abs_original_path, abs_target_path, follow_symlinks=True)            

                if(verbose):
                    print("Copying file: ",abs_original_path,'->',abs_target_path)   

            else:

                if(os.path.isfile(abs_original_path)):
                    lines[ii] = line.replace(support_file, abs_original_path)
                    if(verbose):
                        print("Set path to file: ",lines[ii])        

def parse_gpt_input_file(filePath, condense=False):
    """
    Parses GPT input file 
    """

    finput={}

    with open(filePath, 'r') as f:

        filestr=f.read()

        expressions = filestr.split(';')
        lines = []
        for ii,expression in enumerate(expressions):
            nlines = expression.strip().split('\n')
            for nline in nlines:
                lines.append(nline.strip())

        clean_lines=[]
    
        for line in lines:
            line = line.strip()
            line.replace('\n','')
            if(line!=''):
                tokens = line.split('#')
                if(tokens[0]!=''):
                    nline = tokens[0]
                    if(not (nline[-1]=='{' or nline[-1]=='}')):
                        nline = nline+';'

                    clean_lines.append(nline)
        #print(clean_lines)

    variables={}

    for ii,line in enumerate(clean_lines):
      
        tokens = line.split("=")

        if(len(tokens)==2 and isfloat(tokens[1][:-1].strip())):
 
            name = tokens[0].strip()
            value = float(tokens[1][:-1].strip())
            
            if(name not in variables.keys()):
                variables[name]=value #{"value":value,"index":ii}
                #print(name,value)
            else:
                print("Warning: multiple definitions of variable "+name+" on line "+str(ii)+".")

    for line in clean_lines:
        find_path(line)

    finput['lines']=clean_lines
    finput['variables']=variables

    return finput


def write_gpt_input_file(finput,inputFile):

    #print(inputFile)
    for var in finput["variables"].keys():

        value=finput["variables"][var]
        for index,line in enumerate(finput["lines"]):
            tokens = line.split("=")
            if(len(tokens)==2 and tokens[0].strip()==var):
                finput["lines"][index]=var+"="+str(value)+";"
                break

    with open(inputFile,'w') as f:

        for line in finput["lines"]:
            f.write(line+"\n")


def read_gdf_file(gdffile,verbose=0):
      
    # Read in file:
    #self.vprint("Reading data from files: ",0,True)
  
    #self.vprint("Current file: '"+data_file+"'",1,True)
    #self.vprint("Reading data...",1,False)
    t1 = time.time()
    with open(gdffile, 'rb') as f:
        touts, screens = easygdf.load(f, extra_screen_keys=['q','nmacro',"ID","m"], extra_tout_keys=['q','nmacro',"ID","m"])
    t2 = time.time()
    #self.vprint("done. Time ellapsed: "+self.ptime(t1,t2)+".",0,True)
            
    #self.vprint("Saving wcs tout and ccs screen data structures...",1,False)

    tdata=[]
    pdata=[]

    t1 = time.time()
    count=0
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

            count=count+1
            tdata.append(tout)
         
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

            screen["time"]=np.sum(screen["w"]*screen["t"])
            screen["n"]=n
            screen["number"]=count

            count=count+1
            pdata.append(screen)
                    
    t2 = time.time()
    #self.vprint("done. Time ellapsed: "+self.ptime(t1,t2)+".",0,True)


    ts=[screen['time'] for screen in pdata]
    inds={}

    ts_sorted = sorted(ts)
    pdata_temp=[]

    for t in ts_sorted:
        for screen in pdata:
            if(t == screen["time"]):
                pdata_temp.append(screen)
               
    pdata=pdata_temp
    #self.vprint("done.",0,True)

    return(tdata,pdata)
















