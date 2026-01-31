
# Script to phase a GPT input file as produced by the GUI
# USAGE:  (last two arguments are optional)
# python gpt_phasing input_file.in /path/to/gpt_exe/ /path/to/fields/

import re
import os
#import subprocess
import numpy
import scipy.optimize as sp
from optparse import OptionParser

from pathlib import Path

from gpt.executables import gpt, gdf2a
from gpt.parsers import read_gdf_file

def main():

    # Default setting
    parser = OptionParser()
    parser.add_option("-i", "--infile", dest="infilename", default="", 
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-f", "--file", dest="filename", default="", 
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")
    parser.add_option("-d", "--debug",
                      action="store_true", dest="debug_flag", default=False,
                      help="show GPT execution output on failure")

    (options, args) = parser.parse_args()

    gpt_bin = options.filename

    verbose = options.verbose
    debug_flag = options.debug_flag

    # Interpret input arguments
    path_to_input_file = options.infilename
    gpt_bin = options.filename

    gpt_phasing(path_to_input_file, verbose=True, debug_flag=False)

 
def gpt_phasing(path_to_input_file, 
                gpt_bin='$GPT_BIN', 
                gdf2a_bin='$GDF2A_BIN',
                path_to_phasing_dist=None, 
                verbose=False, 
                debug_flag=False):

    workdir = os.path.dirname(path_to_input_file)

    settings = {}

    #print(path_to_input_file, gpt_bin, path_to_phasing_dist)

    if verbose:
        print("\nPhasing: " + path_to_input_file )

    # Interpret input arguments
    split_input_file_path = path_to_input_file.split('/')
    gpt_input_filename = split_input_file_path[-1]

    extension = Path(gpt_input_filename).suffix

    #print(extension)

    assert gpt_input_filename[-3:]=='.in', 'Phasing script requires the input file end with .in'

    phase_input_filename = gpt_input_filename.replace('.in', '.temp.in')
    finished_phase_input_filename = gpt_input_filename.replace('.in', '.phased.in')
    #phase_input_filename = gpt_input_filename.replace(extension, '.temp{extension}')
    #finished_phase_input_filename = gpt_input_filename.replace(extension, '.phased{extension}')

    path_to_input_file = ''
    for x in range(len(split_input_file_path)-1):
        path_to_input_file = path_to_input_file + split_input_file_path[x] + '/'

    with open(path_to_input_file + gpt_input_filename, 'r') as hand:
        gpt_input_text = hand.readlines()

    # Add replace usual input distribution with single particle at centroid
    if(path_to_phasing_dist):
        dist_line_index = find_lines_containing(gpt_input_text, "setfile")[0]
        gpt_input_text[dist_line_index]=f'setfile("beam", "{os.path.basename(path_to_phasing_dist)}");'

    # Find all lines marked for phasing
    amplitude_flag_indices = find_lines_containing(gpt_input_text, "phasing_amplitude_")
    amplitude_flag_indices = sort_lines_by_first_integer(gpt_input_text, amplitude_flag_indices)
    
    #print(gpt_input_text[amplitude_flag_indices[0]])

    amplitude_indices = []
    desired_amplitude = []
    for index in amplitude_flag_indices:
        variable_name = get_variable_with_string_value(gpt_input_text[index])
        
        amp_index = find_line_with_variable_name(gpt_input_text, variable_name)
        amplitude_indices.append(amp_index)
        desired_amplitude.append(get_variable_on_line(gpt_input_text, amp_index))
        settings[variable_name]=desired_amplitude[-1]

    oncrest_flag_indices = find_lines_containing(gpt_input_text, "phasing_on_crest_")
    oncrest_flag_indices = sort_lines_by_first_integer(gpt_input_text, oncrest_flag_indices)

    oncrest_indices = []
    oncrest_names = []
    for index in oncrest_flag_indices:
        variable_name = get_variable_with_string_value(gpt_input_text[index])
        oncrest_names.append(variable_name)
        oncrest_index = find_line_with_variable_name(gpt_input_text, variable_name)
        oncrest_indices.append(oncrest_index)
        settings[variable_name]=0

    relative_flag_indices = find_lines_containing(gpt_input_text, "phasing_relative_")
    relative_flag_indices = sort_lines_by_first_integer(gpt_input_text, relative_flag_indices)

    relative_indices = []
    desired_relative_phase = []
    for index in relative_flag_indices:
        variable_name = get_variable_with_string_value(gpt_input_text[index])
        rel_index = find_line_with_variable_name(gpt_input_text, variable_name)
        relative_indices.append(rel_index)
        desired_relative_phase.append(get_variable_on_line(gpt_input_text, rel_index))
        settings[variable_name]=desired_relative_phase[-1]

    gamma_flag_indices = find_lines_containing(gpt_input_text, "phasing_gamma_")
    gamma_flag_indices = sort_lines_by_first_integer(gpt_input_text, gamma_flag_indices)

    gamma_indices = []
    gamma_names = []
    for index in gamma_flag_indices:
        variable_name = get_variable_with_string_value(gpt_input_text[index])
        gamma_names.append(variable_name)
        gamma_index = find_line_with_variable_name(gpt_input_text, variable_name)
        gamma_indices.append(gamma_index)
        settings[variable_name]=1

    # Set up phasing input file
    phase_input_text = gpt_input_text

    initial_space_charge = get_variable_by_name(phase_input_text, 'space_charge')
    initial_couplers_on = get_variable_by_name(phase_input_text, 'couplers_on')
    initial_viewscreens_on = get_variable_by_name(phase_input_text, 'viewscreens_on')

    phase_input_text = set_variable_by_name(phase_input_text, 'auto_phase', 1, True)
    phase_input_text = set_variable_by_name(phase_input_text, 'space_charge', 0, False)
    phase_input_text = set_variable_by_name(phase_input_text, 'couplers_on', 0, False)
    phase_input_text = set_variable_by_name(phase_input_text, 'viewscreens_on', 0, False)

    #print(gamma_names,oncrest_names)

    # turn off all cavities
    for index in amplitude_indices:
        phase_input_text = set_variable_on_line(phase_input_text, index, 0.0)

    # set relative phases to zero
    for index in relative_indices:
        phase_input_text = set_variable_on_line(phase_input_text, index, 0.0)

    # set gammas to one
    for index in gamma_indices:
        phase_input_text = set_variable_on_line(phase_input_text, index, 1.0)

    # phase the cavities

    phase_step = 20
    phase_test = numpy.arange(0, 360, phase_step)

    if verbose:
        print(" ")

    for cav_ii in range(len(amplitude_indices)):

        #print(cav_ii)

        if desired_amplitude[cav_ii] > 0:

            # Tell script which cavity we are phasing
            phase_input_text = set_variable_by_name(phase_input_text, 'cavity_phasing_index', cav_ii, False)
            
            # turn on the cavity
            phase_input_text = set_variable_on_line(phase_input_text, amplitude_indices[cav_ii], desired_amplitude[cav_ii])

            gamma_test = []
            for phase in phase_test:


                gamma = run_gpt_phase(phase, 
                                      gpt_bin, 
                                      gdf2a_bin,
                                      phase_input_text, 
                                      path_to_input_file + phase_input_filename, 
                                      oncrest_indices[cav_ii], 
                                      debug_flag,
                                      workdir)

                #print(gamma, phase)

                gamma_test.append(gamma)

  

            gamma_test_indices = numpy.argsort(gamma_test)

            best_phase = phase_test[gamma_test_indices[-1]]
            left_bound = best_phase - phase_step
            right_bound = best_phase + phase_step

            bracket = [left_bound, best_phase, right_bound]

            if verbose:
                print("Cavity " + str(cav_ii) + ": Bracketed between " + str(left_bound) + " and " + str(right_bound))
        
            if (numpy.std(gamma_test) == 0):
                if (gamma_test[0] == 1.0):
                    raise ValueError("GPT PHASING ERROR: No particles reached a screen for any attempted phase.")
                else:
                    raise ValueError("GPT PHASING ERROR: Gamma did not depend on cavity " + str(cav_ii) + " phase, gamma = " + str(gamma_test[0]))

            brent_output = sp.brent(func=neg_run_gpt_phase, args=(gpt_bin, gdf2a_bin, phase_input_text, path_to_input_file + phase_input_filename, oncrest_indices[cav_ii], debug_flag, workdir), brack=bracket, tol=1.0e-5, full_output=1, maxiter=1000)

            best_phase = brent_output[0]
            best_gamma = -brent_output[1]

            phase_input_text = set_variable_on_line(phase_input_text, oncrest_indices[cav_ii], best_phase)
            phase_input_text = set_variable_on_line(phase_input_text, relative_indices[cav_ii], desired_relative_phase[cav_ii])

            final_gamma = run_gpt(gpt_bin, gdf2a_bin, phase_input_text, path_to_input_file + phase_input_filename, debug_flag, workdir)

            if (len(gamma_indices) > 0):
                phase_input_text = set_variable_on_line(phase_input_text, gamma_indices[cav_ii], final_gamma)

            if verbose:
                print("Cavity " + str(cav_ii) + ": Best phase = " + str(best_phase) + ", final gamma = " + str(final_gamma))
                print(" ")

        else:
            
            best_phase = 0.0
            phase_input_text = set_variable_on_line(phase_input_text, oncrest_indices[cav_ii], best_phase)
            phase_input_text = set_variable_on_line(phase_input_text, relative_indices[cav_ii], desired_relative_phase[cav_ii])

            final_gamma = run_gpt(gpt_bin, phase_input_text, path_to_input_file + phase_input_filename, debug_flag, workdir)
            if (len(gamma_indices) > 0):
                phase_input_text = set_variable_on_line(phase_input_text, gamma_indices[cav_ii], final_gamma)

            if verbose:
                print("Skipping: Cavity " + str(cav_ii) + ": Best phase = " + str(best_phase) + ", final gamma = " + str(final_gamma))
                print(" ")

        settings[oncrest_names[cav_ii]]=best_phase
        settings[gamma_names[cav_ii]]=final_gamma

    # Put back in the original settings, turn off phasing flags, set reference gamma
    phase_input_text = set_variable_by_name(phase_input_text, 'auto_phase', 0, True)
    phase_input_text = set_variable_by_name(phase_input_text, 'space_charge', initial_space_charge, False)
    phase_input_text = set_variable_by_name(phase_input_text, 'couplers_on', initial_couplers_on, False)
    phase_input_text = set_variable_by_name(phase_input_text, 'viewscreens_on', initial_viewscreens_on, False)

    # Write phased input file
    with open(path_to_input_file + finished_phase_input_filename,'wt') as fid:
        fid.writelines(phase_input_text)

    # Delete temporary input file
    trashclean(path_to_input_file + phase_input_filename, True)

    return (finished_phase_input_filename, settings)
                
# ---------------------------------------------------------------------------- #
# Run GPT with a given phase for a cavity, returns value of (NEGATIVE) gamma
# ---------------------------------------------------------------------------- #
def neg_run_gpt_phase(phase, 
                      gpt_bin,
                      gdf2a_bin,
                      phase_input_text, 
                      filename, 
                      oncrest_index, 
                      debug_flag,
                      workdir):

    gamma = run_gpt_phase(phase, gpt_bin, gdf2a_bin, phase_input_text, filename, oncrest_index, debug_flag, workdir)
    
    return -gamma

# ---------------------------------------------------------------------------- #
# Run GPT with a given phase for a cavity, returns value of gamma
# ---------------------------------------------------------------------------- #
def run_gpt_phase(phase, 
                  gpt_bin, 
                  gdf2a_bin,
                  phase_input_text, 
                  filename, 
                  oncrest_index, 
                  debug_flag,
                  workdir):

    phase_input_text = set_variable_on_line(phase_input_text, oncrest_index, phase)
    
    return run_gpt(gpt_bin, gdf2a_bin, phase_input_text, filename, debug_flag, workdir)

# ---------------------------------------------------------------------------- #
# Just run GPT, given an input file to write
# ---------------------------------------------------------------------------- #
def run_gpt(gpt_bin, 
            gdf2a_bin,
            phase_input_text, 
            filename, 
            debug_flag,
            workdir):

    #print(path_to_gpt_bin, filename, debug_flag, workdir)
    with open(filename, 'wt') as fid:
        fid.writelines(phase_input_text)
    
    output_filename = filename.replace(".in", ".gdf")
    output_text_filename = output_filename.replace(".gdf", ".txt")

    gpt(filename, output_filename, verbose=False, workdir=workdir, gpt_bin=gpt_bin)
    _, pdata = read_gdf_file(output_filename)
    gamma = pdata[-1]['G'].mean()

    trashclean(output_filename, True)
    trashclean(output_text_filename, True)

    return gamma

# ---------------------------------------------------------------------------- #
# sets the value of a variable with a given name, returns the entire string array
# ---------------------------------------------------------------------------- #
def set_variable_by_name(gpt_input_text, name, value, crash_on_error):

    gpt_input_text_new = gpt_input_text

    index = find_line_with_variable_name(gpt_input_text_new, name)

    if (index > -1):
        gpt_input_text_new = set_variable_on_line(gpt_input_text_new, index, value)
    else:
        if crash_on_error:
            raise ValueError("GPT PHASING ERROR: variable " + name + " not found.")

    return gpt_input_text_new

# ---------------------------------------------------------------------------- #
# find line with variable name
# ---------------------------------------------------------------------------- #
def find_line_with_variable_name(gpt_input_text, name):

    if (len(name.strip()) == 0):
        raise ValueError("GPT PHASING ERROR: attempting to find variable with name = empty string.")

    gpt_input_text_new = gpt_input_text

    indices = []
    for ii in range(len(gpt_input_text_new)):
        line = gpt_input_text_new[ii]
        match = re.search(name + "[ ]*=[^=]", line)
        if (match):
                indices.append(ii)

    if len(indices) == 0:
        return -1

    if len(indices) > 1:
        raise ValueError("ERROR: variable " + name + " found on more than one line.")

    return indices[0]

# ---------------------------------------------------------------------------- #
# gets the value of a variable with a given name, returns the value
# ---------------------------------------------------------------------------- #
def get_variable_by_name(gpt_input_text, name):

    index = find_line_with_variable_name(gpt_input_text, name)

    if (index < 0):
        return 0

    gpt_input_text_new = gpt_input_text

    value = get_variable_on_line(gpt_input_text_new, index)

    return value

# ---------------------------------------------------------------------------- #
# Gets the value of a variable on a line, returns the value
# ---------------------------------------------------------------------------- #
def get_variable_on_line(gpt_input_text, index):

    line = gpt_input_text[index]
    
    split_on_comments = line.split('#')
    line_bare = split_on_comments[0].replace(';','').replace(' ', '')
    
    split_on_equals = line_bare.split('=')
    value_string = split_on_equals[-1]
    
    return float(value_string)
        


# ---------------------------------------------------------------------------- #
# sets the value of a variable on a line, returns the entire string array
# ---------------------------------------------------------------------------- #
def set_variable_on_line(gpt_input_text, index, value):

    line = gpt_input_text[index]
    
    split_on_comments = line.split('#')

    original_comment = '\n'
    if (len(split_on_comments) > 1):
            original_comment = ' #' + split_on_comments[-1]
    line_bare = split_on_comments[0].replace(';','').replace(' ', '')
    
    split_on_equals = line_bare.split('=')
    
    variable_name = split_on_equals[0]

    new_line = variable_name + '=' + str(value) + ';' + original_comment

    gpt_input_text_new = gpt_input_text
    gpt_input_text_new[index] = new_line

    return gpt_input_text_new


# ---------------------------------------------------------------------------- #
# Returns indices of lines that have been sorted by an integer that appears at the end of the line
# ---------------------------------------------------------------------------- #
def sort_lines_by_first_integer(lines, indices):

    numbers = []        
    
    for ii in indices:
        line = lines[ii]
        m = re.search(r"\d+", line)
        integer_string = m.group(0)
        numbers.append(float(integer_string))
   
    sorted_numbers_indices = numpy.argsort(numbers)

    sorted_indices = []
    for ii in sorted_numbers_indices:
        sorted_indices.append(indices[ii])
    
    return sorted_indices

# ---------------------------------------------------------------------------- #
# Returns the value of a variable that is a string
# ---------------------------------------------------------------------------- #
def get_variable_with_string_value(line):
    
    split_on_comments = line.split('#')
    line_bare = split_on_comments[0].replace(';','').replace(' ', '')
    
    split_on_equals = line_bare.split('=')
    value_string = split_on_equals[-1]
    
    return value_string.strip()

# ---------------------------------------------------------------------------- #
# Find lines containing a string, returns their indices, case insensitive
# ---------------------------------------------------------------------------- #
def find_lines_containing(lines, string):
    
    string_lower = string.lower()

    indices = []
    for ii in range(len(lines)):
        line = lines[ii].lower()
        split_on_comments = line.split('#')
        line = split_on_comments[0]
        match = re.search(string_lower, line)
        if match:
            indices.append(ii)
    
    return indices

# ---------------------------------------------------------------------------- #
# Deletes a file
# ---------------------------------------------------------------------------- #
def trashclean(trashname, control):
    if control:
        os.system("rm -f "+trashname)


# ---------------------------------------------------------------------------- #
# This allows the main function to be at the beginning of the file
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()


