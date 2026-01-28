import functools
import os
import subprocess

import functools

import os
import functools
import inspect
import warnings

# Define your fixed list of environment variable keys here
VALID_EXE_VARS = {'asci2gdf_bin', 'gdf2a_bin', 'gpt_bin'}

def expand_gpt_env_vars(func):
    """
    Automatically scans all function arguments. If an argument name matches
    our fixed list, it expands any environment variables found in its value.
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Map positional args and defaults into a single dictionary
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # 2. Iterate through the arguments
        for name, value in bound_args.arguments.items():
            # Only process if it's in our fixed list and is a string
            if name in VALID_EXE_VARS and isinstance(value, str):
                # Check for $VAR or %VAR%
                if '$' in value or '%' in value:
                    # Strip symbols to get the key (e.g., $ASCI2GDF_BIN -> ASCI2GDF_BIN)
                    env_key = value.replace('$', '').replace('%', '')
                    
                    expanded = os.environ.get(env_key)
                    if expanded:
                        # Expand and normalize the path for the current OS
                        bound_args.arguments[name] = os.path.normpath(expanded)
                    else:
                        # Optional: provide a fallback or warning
                        # RAISE WARNING HERE
                        warnings.warn(
                            f"Environment variable '{env_key}' is not set. "
                            f"Using literal value '{value}' for argument '{name}'.",
                            UserWarning
                        )

        # 3. Call the function with the updated arguments
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper

GPT_BINS = {k: os.path.normpath(os.path.expandvars(k)) for k in ['$GPT_BIN', '$GDF2A_BIN', '$ASCI2GDF_BIN']}


def check_file_suffix(filename, raise_on):

    if filename.endswith(raise_on):
        raise ValueError(f'File name has the wrong suffix ({raise_on}): {filename}')

@expand_gpt_env_vars
def asci2gdf(ascii_file, gdf_file, asci2gdf_bin='$ASCI2GDF_BIN', strict_file_suffixes=False):
    
    cmd_list = [asci2gdf_bin, "-o", gdf_file, ascii_file]

    if strict_file_suffixes:
        check_file_suffix(gdf_file, raise_on='.txt')
        check_file_suffix(ascii_file, raise_on='.gdf')

    try:
        result = subprocess.run(
            cmd_list, 
            check=True, 
            capture_output=True, 
            text=True
        )
    except subprocess.CalledProcessError as e:
        print("--- ASCI2GDF STDOUT ---")
        print(e.stdout)
        print("--- ASCI2GDF STDERR ---")
        print(e.stderr)  # <--- THIS is where the real answer is
        raise e

@expand_gpt_env_vars
def gdf2a(gdf_file, ascii_file, gdf2a_bin='$GDF2A_BIN', strict_file_suffixes=False):

    if strict_file_suffixes:
        check_file_suffix(gdf_file, raise_on='.txt')
        check_file_suffix(ascii_file, raise_on='.gdf')
    
    cmd_list = [gdf2a_bin, "-o", ascii_file, gdf_file]
    
    try:
        result = subprocess.run(
            cmd_list, 
            check=True, 
            capture_output=True, 
            text=True
        )
    except subprocess.CalledProcessError as e:
        print("--- GDF2A STDOUT ---")
        print(e.stdout)
        print("--- GDF2A STDERR ---")
        print(e.stderr)  # <--- THIS is where the real answer is
        raise e

    