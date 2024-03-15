################################################################################
# File: easygdf.py
# Description: easygdf provides methods that allow the reading of GDF files
#              directly into python/numpy atomic data types.
# Author: Christopher M. Pierce (cmp285@cornell.edu)
################################################################################

################################################################################
# Imports
################################################################################
import io
import numpy
import pickle
import struct
import unittest

################################################################################
# Format Specific Constants
################################################################################
# GDF file specifics
GDF_NAME_LEN = 16
GDF_MAGIC    = 94325877

# The GDF data type magic numbers
GDF_ASCII     = 1
GDF_CHAR      = 30
GDF_DOUBLE    = 3
GDF_FLOAT     = 90
GDF_INT16     = 50
GDF_INT64     = 80
GDF_LONG      = 2
GDF_NULL      = 10
GDF_UCHAR     = 20
GDF_UINT16    = 40
GDF_UINT32    = 60
GDF_UINT64    = 70
GDF_UNDEFINED = 0

# The bit masks for the types of GDF blocks
GDF_DIRECTORY = 256
GDF_END       = 512
GDF_SINGLE    = 1024
GDF_ARRAY     = 2048

################################################################################
# Function Definitions
################################################################################


def is_GDF_file(file):
    '''Returns whether or not the file is a GDF file.  Accepts an already open file in read, binary mode.'''
    # Rewind the file to the beginning
    file.seek(0)

    # Get the magic number
    magic_number, = struct.unpack('i', file.read(4))

    # Check it against the real magic number
    return magic_number == GDF_MAGIC


def load(file, tout_filter=lambda x: True, screen_filter=lambda x: True, screen_block_size=None, tout_block_size=None,
         extra_tout_keys=[], extra_screen_keys=[]):
    '''Reads all screens and touts  from the open file object file.  These are returned as a the tuple (touts,
    screens) where each of touts and screens is a list of numpy arrays of the phase space coordinates.  The output
    array has the format:

        [[x_0,   x_1,   x_2,   ..., x_(n-1)  ],
         [BGx_0, BGx_1, BGx_2, ..., BGx_(n-1)],
         [y_0,   y_1,   y_2,   ..., y_(n-1)  ],
         [BGy_0, BGy_1, BGy_2, ..., BGy_(n-1)],
         [z_0,   z_1,   z_2,   ..., z_(n-1)  ],
         [BGz_0, BGz_1, BGz_2, ..., BGz_(n-1)],
         [t_0,   t_1,   t_2,   ..., t_(n-1)  ]]

    The spatial coordinates have units of meters and BGx, BGy, BGz refer to the components of the normalized
    relativistic momentum Beta*Gamma and are unitless.  Here, time has units of seconds and charge has units of
    Coulombs.

    The touts and screens can be filtered by time and position.  The inputs `tout_filter` and `screen_filter` accept
    lambdas of one parameter.  That parameter is time/position and the lambda should return true or false indicating
    whether to read that screen or not.  Although this does speed up load times and reduces memory overhead in most
    cases, some large files can take a long time to load even a single screen or tout.  This is because the way the
    GDF file is formatted requires the parser to read every block (including variables in the touts and screens) once
    while going through the file.  To get around this, you can try specifying `screen_block_size` and
    `tout_block_size`.  These tell the parser how large a tout or screen is in bytes and causes it to skip over them
    instead of reading the header of each variable.  This can improve performance by an order of magnitude on some
    files.  Be careful however, since incorrectly setting the block size will cause the parser to load corrupted
    data.

    The `extra_tout_keys` and `extra_screen_keys` parameters are used to load addition particle variables such as
    scatter variables or field values at the particle locations.  It is a list of strings where each string is the
    key to be loaded.  The key should match up with the name of the variable as written to the GDF file.  The
    variables will be appended to the numpy array as rows in the order they appear in the list.

    Example Usage:

    # Load the EasyGDF module
    import easygdf

    # Open a GDF file
    with open('output.gdf', 'rb') as f:
      touts, screens = easygdf.load(f)
    '''

    # Check if the file is a real file object
    if (not isinstance(file, io.IOBase)):
        raise TypeError('Argument is not a file-like object')

    # If the file wasn't opened in binary mode
    if 'b' not in file.mode:
        # Raise an exception
        raise ValueError("File is not in binary mode.  "
                         "Try opening with option 'rb'")

    # Check if the file is readable
    try:
        file.read(1)

    except IOError:
        raise ValueError('Could not read from file')

    # Check it against the real magic number
    if (not is_GDF_file(file)):
        raise ValueError('File is not GDF formatted')

    # Jump to where the data actually begins
    file.seek(48)

    # Make a variable to keep track of state
    state = 'root'

    # Make holders for our phase space variables
    touts = []
    screens = []
    block_value = None

    # Make storage for the values we will use
    screen_tout_arrays = {}
    first_tout = True

    # Go into an infinite loop
    while True:
        # Read the block's header
        block_header = file.read(16 + 4 + 4)

        # If no data came back
        if block_header == b'':
            # If we were in a tout
            if (state == 'tout'):
                # Get the array
                phase_space_array = numpy.array([
                    screen_tout_arrays['x'],
                    screen_tout_arrays['Bx'] * screen_tout_arrays['G'],
                    screen_tout_arrays['y'],
                    screen_tout_arrays['By'] * screen_tout_arrays['G'],
                    screen_tout_arrays['z'],
                    screen_tout_arrays['Bz'] * screen_tout_arrays['G'],
                    numpy.ones(screen_tout_arrays['x'].shape[0]) * block_value,
                ])

                # For each extra one
                for key in extra_tout_keys:
                    # Append it
                    phase_space_array = numpy.append(phase_space_array, [screen_tout_arrays[key]], axis=0)

                # Append it to the list
                touts.append(phase_space_array)

            # Or, if we were in a screen
            elif (state == 'screen'):
                # Create the phase space array
                phase_space_array = numpy.array([
                    screen_tout_arrays['x'],
                    screen_tout_arrays['Bx'] * screen_tout_arrays['G'],
                    screen_tout_arrays['y'],
                    screen_tout_arrays['By'] * screen_tout_arrays['G'],
                    screen_tout_arrays['z'],
                    screen_tout_arrays['Bz'] * screen_tout_arrays['G'],
                    screen_tout_arrays['t'],
                ])

                # For each extra one
                for key in extra_screen_keys:
                    # Append it
                    phase_space_array = numpy.append(phase_space_array, [screen_tout_arrays[key]], axis=0)

                # Append it to the list
                screens.append(phase_space_array)

            # End the loop
            break

        else:
            # Clean up the name
            block_name = block_header[0:16]
            block_name = block_name.split(b'\0', 1)[0]
            block_name = block_name.decode('utf8')

            # Get the block's type_flag and size
            block_type_flag, = struct.unpack('i', block_header[16:20])
            block_size, = struct.unpack('i', block_header[20:24])

            # Break up the type flag into its parts
            directory = bool(block_type_flag & GDF_DIRECTORY)
            end = bool(block_type_flag & GDF_END)
            single = bool(block_type_flag & GDF_SINGLE)
            array = bool(block_type_flag & GDF_ARRAY)

            # If we are in the root state
            if (state == 'root'):
                # If we are the start of a tout block
                if (block_name == 'time'):
                    # Get the value of it
                    block_value, = struct.unpack('d', file.read(8))

                    # If the filter says that we will load it
                    if (tout_filter(block_value)):
                        # We are now in the tout state
                        state = 'tout'

                        # Clean out the array dict
                        screen_tout_arrays = {}
                    else:
                        # If we are given the number of particles and elements in a tout
                        if (tout_block_size is not None) and not first_tout:
                            # Try to skip by the correct amount
                            file.seek(tout_block_size, 1)

                    # We are no longer the first tout
                    first_tout = False

                # If we are the start of a screen
                elif (block_name == 'position'):
                    # Get the value of it
                    block_value, = struct.unpack('d', file.read(8))

                    # If the filter says that we will load it
                    if (screen_filter(block_value)):
                        # We are now in the screen state
                        state = 'screen'

                        # Clean out the array dict
                        screen_tout_arrays = {}
                    else:
                        # If we are given the number of particles and elements in a tout
                        if (screen_block_size is not None):
                            # Try to skip by the correct amount
                            file.seek(screen_block_size, 1)

                # If it's just some random block
                else:
                    # Move us to the next block
                    file.seek(block_size, 1)

            # If we are in the tout state
            elif ((state == 'tout') or (state == 'screen')):
                # If it's an array object
                if (array):
                    # Get the data-type
                    data_type = get_data_type(block_type_flag & 255)

                    # If it's a double
                    if (data_type == 'double'):
                        # Get the array
                        screen_tout_arrays[block_name] = numpy.fromfile(file, dtype=numpy.dtype('d'),
                                                                        count=block_size // 8)

                    # Otherwise
                    else:
                        # Skip ahead
                        file.seek(block_size, 1)

                # If it's time to end the block
                elif (end):
                    if (state == 'tout'):
                        # Get the array
                        phase_space_array = numpy.array([
                            screen_tout_arrays['x'],
                            screen_tout_arrays['Bx'] * screen_tout_arrays['G'],
                            screen_tout_arrays['y'],
                            screen_tout_arrays['By'] * screen_tout_arrays['G'],
                            screen_tout_arrays['z'],
                            screen_tout_arrays['Bz'] * screen_tout_arrays['G'],
                            numpy.ones(screen_tout_arrays['x'].shape[0]) * block_value,
                        ])

                        # For each extra one
                        for key in extra_tout_keys:
                            # Append it
                            phase_space_array = numpy.append(phase_space_array, [screen_tout_arrays[key]], axis=0)

                        # Append it to the list
                        touts.append(phase_space_array)

                    elif (state == 'screen'):
                        # Create the phase space array
                        phase_space_array = numpy.array([
                            screen_tout_arrays['x'],
                            screen_tout_arrays['Bx'] * screen_tout_arrays['G'],
                            screen_tout_arrays['y'],
                            screen_tout_arrays['By'] * screen_tout_arrays['G'],
                            screen_tout_arrays['z'],
                            screen_tout_arrays['Bz'] * screen_tout_arrays['G'],
                            screen_tout_arrays['t'],
                        ])

                        # For each extra one
                        for key in extra_screen_keys:
                            # Append it
                            phase_space_array = numpy.append(phase_space_array, [screen_tout_arrays[key]], axis=0)

                        # Append it to the list
                        screens.append(phase_space_array)

                    # Change the state back
                    state = 'root'

                # Otherwise
                else:
                    # Skip ahead
                    file.seek(block_size, 1)

    # Return everything
    return (touts, screens)


def load_dict(file, tout_filter=lambda x: True, screen_filter=lambda x: True, screen_block_size=None,
              tout_block_size=None):
    '''Reads all screens and touts  from the open file object file.  These are returned as python dictionaries where
    the keys to the dict are the keys to the arrays in the GDF file itself.  The spatial coordinates have units of
    meters and BGx, BGy, BGz refer to the components of the normalized relativistic momentum Beta*Gamma and are
    unitless.  Here, time has units of seconds and charge has units of Coulombs.

    The touts and screens can be filtered by time and position.  The inputs `tout_filter` and `screen_filter` accept
    lambdas of one parameter.  That parameter is time/position and the lambda should return true or false indicating
    whether to read that screen or not.  Although this does speed up load times and reduces memory overhead in most
    cases, some large files can take a long time to load even a single screen or tout.  This is because the way the
    GDF file is formatted requires the parser to read every block (including variables in the touts and screens) once
    while going through the file.  To get around this, you can try specifying `screen_block_size` and
    `tout_block_size`.  These tell the parser how large a tout or screen is in bytes and causes it to skip over them
    instead of reading the header of each variable.  This can improve performance by an order of magnitude on some
    files.  Be careful however, since incorrectly setting the block size will cause the parser to load corrupted
    data.

    The `extra_tout_keys` and `extra_screen_keys` parameters are used to load addition particle variables such as
    scatter variables or field values at the particle locations.  It is a list of strings where each string is the
    key to be loaded.  The key should match up with the name of the variable as written to the GDF file.  The
    variables will be appended to the numpy array as rows in the order they appear in the list.

    Example Usage:

    # Load the EasyGDF module
    import easygdf

    # Open a GDF file
    with open('output.gdf', 'rb') as f:
      touts, screens = easygdf.load(f)
    '''

    # Check if the file is a real file object
    if (not isinstance(file, io.IOBase)):
        raise TypeError('Argument is not a file-like object')

    # If the file wasn't opened in binary mode
    if 'b' not in file.mode:
        # Raise an exception
        raise ValueError("File is not in binary mode.  "
                         "Try opening with option 'rb'")

    # Check if the file is readable
    try:
        file.read(1)

    except IOError:
        raise ValueError('Could not read from file')

    # Check it against the real magic number
    if (not is_GDF_file(file)):
        raise ValueError('File is not GDF formatted')

    # Jump to where the data actually begins
    file.seek(48)

    # Make a variable to keep track of state
    state = 'root'

    # Make holders for our phase space variables
    touts = []
    screens = []
    block_value = None

    # Make storage for the values we will use
    screen_tout_arrays = {}
    first_tout = True

    # Go into an infinite loop
    while True:
        # Read the block's header
        block_header = file.read(16 + 4 + 4)

        # If no data came back
        if block_header == b'':
            # If we were in a tout
            if (state == 'tout'):
                # Append it to the list
                touts.append(screen_tout_arrays)

            # Or, if we were in a screen
            elif (state == 'screen'):
                # Append it to the list
                screens.append(screen_tout_arrays)

            # End the loop
            break

        else:
            # Clean up the name
            block_name = block_header[0:16]
            block_name = block_name.split(b'\0', 1)[0]
            block_name = block_name.decode('utf8')

            # Get the block's type_flag and size
            block_type_flag, = struct.unpack('i', block_header[16:20])
            block_size, = struct.unpack('i', block_header[20:24])

            # Break up the type flag into its parts
            directory = bool(block_type_flag & GDF_DIRECTORY)
            end = bool(block_type_flag & GDF_END)
            single = bool(block_type_flag & GDF_SINGLE)
            array = bool(block_type_flag & GDF_ARRAY)

            # If we are in the root state
            if (state == 'root'):
                # If we are the start of a tout block
                if (block_name == 'time'):
                    # Get the value of it
                    block_value, = struct.unpack('d', file.read(8))

                    # If the filter says that we will load it
                    if (tout_filter(block_value)):
                        # We are now in the tout state
                        state = 'tout'

                        # Clean out the array dict
                        screen_tout_arrays = {}
                    else:
                        # If we are given the number of particles and elements in a tout
                        if ((tout_block_size is not None) and not first_tout):
                            # Try to skip by the correct amount
                            file.seek(tout_block_size, 1)

                    # We are no longer the first tout
                    first_tout = False

                # If we are the start of a screen
                elif (block_name == 'position'):
                    # Get the value of it
                    block_value, = struct.unpack('d', file.read(8))

                    # If the filter says that we will load it
                    if (screen_filter(block_value)):
                        # We are now in the screen state
                        state = 'screen'

                        # Clean out the array dict
                        screen_tout_arrays = {}
                    else:
                        # If we are given the number of particles and elements in a tout
                        if (screen_block_size is not None):
                            # Try to skip by the correct amount
                            file.seek(screen_block_size, 1)

                # If it's just some random block
                else:
                    # Move us to the next block
                    file.seek(block_size, 1)

            # If we are in the tout state
            elif ((state == 'tout') or (state == 'screen')):
                # If it's an array object
                if (array):
                    # Get the data-type
                    data_type = get_data_type(block_type_flag & 255)

                    # If it's a double
                    if (data_type == 'double'):
                        # Get the array
                        screen_tout_arrays[block_name] = numpy.fromfile(file, dtype=numpy.dtype('d'),
                                                                        count=block_size // 8)

                    # Otherwise
                    else:
                        # Skip ahead
                        file.seek(block_size, 1)

                # If it's time to end the block
                elif (end):
                    if (state == 'tout'):
                        # Append it to the list
                        touts.append(screen_tout_arrays)

                    elif (state == 'screen'):
                        # Append it to the list
                        screens.append(screen_tout_arrays)

                    # Change the state back
                    state = 'root'

                # Otherwise
                else:
                    # Skip ahead
                    file.seek(block_size, 1)

    # Return everything
    return (touts, screens)


def load_initial_distribution(file, extra_screen_keys=[]):
    '''Reads an initial distributions file.  The output array has the format:

        [[x_0,   x_1,   x_2,   ..., x_(n-1)  ],
         [BGx_0, BGx_1, BGx_2, ..., BGx_(n-1)],
         [y_0,   y_1,   y_2,   ..., y_(n-1)  ],
         [BGy_0, BGy_1, BGy_2, ..., BGy_(n-1)],
         [z_0,   z_1,   z_2,   ..., z_(n-1)  ],
         [BGz_0, BGz_1, BGz_2, ..., BGz_(n-1)],
         [t_0,   t_1,   t_2,   ..., t_(n-1)  ]]

    The spatial coordinates have units of meters and BGx, BGy, BGz refer to the components of the normalized
    relativistic momentum Beta*Gamma and are unitless.  Here, time has units of seconds and charge has units of
    Coulombs.

    The parameter `extra_screen_keys` adds particle parameters to the array as described in the function `load`.

    # Load the EasyGDF module
    import easygdf

    # Open a GDF file
    with open('initial_distribution.gdf', 'rb') as f:
      initial_screen = easygdf.load_initial_distribution(f)

      '''

    # Check if the file is a real file object
    if(not isinstance(file, io.IOBase)):
        raise TypeError('Argument is not a file-like object')

    # If the file wasn't opened in binary mode
    if 'b' not in file.mode:
        # Raise an exception
        raise ValueError("File is not in binary mode.  "
                "Try opening with option 'rb'")

    # Check if the file is readable
    try:
        file.read(1)

    except IOError:
        raise ValueError('Could not read from file')

    # Rewind the file to the beginning
    file.seek(0)

    # Get the magic number
    magic_number, = struct.unpack('i', file.read(4))

    # Check it against the real magic number
    if(magic_number != GDF_MAGIC):
        raise ValueError('File is not GDF formatted')

    # Jump to where the data actually begins
    file.seek(48)

    # Make holders for our phase space variables
    values = {}
    block_value = None

    # Go into an infinite loop
    while True:
        # Read the block's header
        block_name = file.read(16)

        # If no data came back
        if block_name == b'':
            # Exit the loop
            break

        # Clean up the name
        block_name = block_name.split(b'\0',1)[0]
        block_name = block_name.decode('utf8')

        # Get the block's type_flag and size
        block_type_flag, = struct.unpack('i', file.read(4))
        block_size, = struct.unpack('i', file.read(4))

        # Break up the type flag into its parts
        directory = bool(block_type_flag & GDF_DIRECTORY)
        end       = bool(block_type_flag & GDF_END)
        single    = bool(block_type_flag & GDF_SINGLE)
        array     = bool(block_type_flag & GDF_ARRAY)

        # Write out the list of parameters we care about
        parameter_names = ['x', 'y', 'z', 'GBx', 'GBy', 'GBz', 't', 'q',
                'nmacro']

        # If we are the start of a tout block
        if(block_name in parameter_names):
            # Get the value of it
            val = numpy.fromfile(file, dtype=numpy.dtype('d'),
                    count=block_size//8)

            # Add it to the dict
            values[block_name]  = val

        # If it's just some random block
        else:
            # Move us to the next block
            file.seek(block_size, 1)

    # Generate the screen
    screen = numpy.array([
        values['x'],
        values['GBx'],
        values['y'],
        values['GBy'],
        values['z'],
        values['GBz'],
        values['t'],
        ])

    # Get the charge per macroparticle
    q = values['q'] * values['nmacro']

    # Add extra values
    for key in extra_screen_keys:
        # Add the key
        screen = numpy.append(screen, [values[key]], axis=0)

    # Return everything
    return screen


def load_initial_distribution_dict(file):
    '''Reads an initial distributions file.  The output of this function is a dict where the keys to the dict are the
    keys corresponding to the arrays in the GDF file. The spatial coordinates have units of meters and BGx, BGy,
    BGz refer to the components of the normalized relativistic momentum Beta*Gamma and are unitless.  Here,
    time has units of seconds and charge has units of Coulombs.

    The parameter `extra_screen_keys` adds particle parameters to the array as described in the function `load`.

    # Load the EasyGDF module
    import easygdf

    # Open a GDF file
    with open('initial_distribution.gdf', 'rb') as f:
      initial_screen = easygdf.load_initial_distribution(f)

      '''

    # Check if the file is a real file object
    if(not isinstance(file, io.IOBase)):
        raise TypeError('Argument is not a file-like object')

    # If the file wasn't opened in binary mode
    if 'b' not in file.mode:
        # Raise an exception
        raise ValueError("File is not in binary mode.  "
                "Try opening with option 'rb'")

    # Check if the file is readable
    try:
        file.read(1)

    except IOError:
        raise ValueError('Could not read from file')

    # Rewind the file to the beginning
    file.seek(0)

    # Get the magic number
    magic_number, = struct.unpack('i', file.read(4))

    # Check it against the real magic number
    if(magic_number != GDF_MAGIC):
        raise ValueError('File is not GDF formatted')

    # Jump to where the data actually begins
    file.seek(48)

    # Make holders for our phase space variables
    values = {}
    block_value = None

    # Go into an infinite loop
    while True:
        # Read the block's header
        block_name = file.read(16)

        # If no data came back
        if block_name == b'':
            # Exit the loop
            break

        # Clean up the name
        block_name = block_name.split(b'\0',1)[0]
        block_name = block_name.decode('utf8')

        # Get the block's type_flag and size
        block_type_flag, = struct.unpack('i', file.read(4))
        block_size, = struct.unpack('i', file.read(4))

        # Break up the type flag into its parts
        directory = bool(block_type_flag & GDF_DIRECTORY)
        end       = bool(block_type_flag & GDF_END)
        single    = bool(block_type_flag & GDF_SINGLE)
        array     = bool(block_type_flag & GDF_ARRAY)

        # Write out the list of parameters we care about
        parameter_names = ['x', 'y', 'z', 'GBx', 'GBy', 'GBz', 't', 'q',
                'nmacro']

        # If we are the start of a tout block
        if(block_name in parameter_names):
            # Get the value of it
            val = numpy.fromfile(file, dtype=numpy.dtype('d'),
                    count=block_size//8)

            # Add it to the dict
            values[block_name]  = val

        # If it's just some random block
        else:
            # Move us to the next block
            file.seek(block_size, 1)

    # Return everything
    return values


def get_data_type(flag):
    '''This method returns a string identifying the datatype of a GDF block
    given the last byte of it'''
    if(flag == GDF_ASCII):
        return 'ascii'
    elif(flag == GDF_CHAR):
        return 'char'
    elif(flag == GDF_DOUBLE):
        return 'double'
    elif(flag == GDF_FLOAT):
        return 'float'
    elif(flag == GDF_INT16):
        return 'int16'
    elif(flag == GDF_INT64):
        return 'int64'
    elif(flag == GDF_LONG):
        return 'long'
    elif(flag == GDF_NULL):
        return 'null'
    elif(flag == GDF_UCHAR):
        return 'uchar'
    elif(flag == GDF_UINT16):
        return 'uint16'
    elif(flag == GDF_UINT32):
        return 'uint32'
    elif(flag == GDF_UINT64):
        return 'uint64'
    elif(flag == GDF_UNDEFINED):
        return 'undefined'
    else:
        raise ValueError('Unknown data type')


################################################################################
# Unit Tests
################################################################################
class TestEasyGDF(unittest.TestCase):
    def test_load(self):
        '''Tests if the load method is working correctly by opening up a test
        file.'''
        # Make a dummy variable to store the data we will use in this test
        test_data = None
        validation_data = None

        # Attempt to open the file with the method under test
        with open('test.gdf', 'rb') as f:
            test_data = load(f)

        # Load the validation data
        with open('test.pickle', 'rb') as f:
            validation_data = pickle.load(f)

        # Compare them
        for test_tout, val_tout in zip(test_data[0], validation_data[0]):
            self.assertTrue( numpy.isclose(test_tout, val_tout[:-1]).all())

    def test_load_with_extra_keys(self):
        '''Tests if the load method is working correctly with additional keys by opening up a test
        file and looking at the charge variable.'''
        # Make a dummy variable to store the data we will use in this test
        test_data = None
        validation_data = None

        # Attempt to open the file with the method under test
        with open('test.gdf', 'rb') as f:
            test_data = load(f, extra_tout_keys=['q', 'nmacro'])

        # Load the validation data
        with open('test.pickle', 'rb') as f:
            validation_data = pickle.load(f)

        # Compare them
        for test_tout, val_tout in zip(test_data[0], validation_data[0]):
            # Make a new array to compare with
            modified_test_data = test_tout[:-1]
            modified_test_data[-1] = modified_test_data[-1] * test_tout[-1]

            self.assertTrue(numpy.isclose(modified_test_data, val_tout, atol=1e-18).all())

    def test_load_wrong_datatype(self):
        '''Test a user trying to plug the wrong datatype into the method.'''
        # Run it with something weird
        with self.assertRaises(TypeError):
            load('blah')

    def test_load_closed_file(self):
        '''Test the load method with a file that isn't open.'''
        # Open a file and close it
        closed_file = None
        with open('test.pickle', 'rb') as f:
            closed_file = f

        with self.assertRaises(ValueError):
            load(closed_file)

    def test_load_wrong_format(self):
        '''Test someone trying to load something that isn't a GDF file'''
        # Do it
        with self.assertRaises(ValueError):
            with open('test.pickle', 'rb') as f:
                load(f)

    def test_load_used_file(self):
        '''Tests the load method on a file that hasn't been re-wound'''
        # Do it
        with open('test.gdf', 'rb') as f:
            f.seek(1)
            load(f)

    def test_load_non_binary(self):
        '''Tests the load method on a file that is opened, but not in binary
        mode'''
        # Do it
        try:
            with open('test.gdf', 'r') as f:
                load(f)

        except ValueError as e:
            self.assertEqual(str(e), "File is not in binary mode.  "
                    "Try opening with option 'rb'")
            return

        # If we don't hit the exception, fail
        self.assertTrue(False)

    def test_load_initial_distribution(self):
        '''Tests if the load method is working correctly by opening up a test
        file.'''
        # Make a dummy variable to store the data we will use in this test
        test_data = None
        validation_data = None

        # Attempt to open the file with the method under test
        with open('test_cathode.gdf', 'rb') as f:
            test_data = load_initial_distribution(f)

        # Load the validation data
        with open('test_cathode.pickle', 'rb') as f:
            validation_data = pickle.load(f)

        # Compare them
        self.assertTrue(numpy.isclose(test_data, validation_data[:-1]).all())

    def test_load_initial_distribution_with_extra(self):
        '''Tests if the load method is working correctly by opening up a test
        file and loading charge as well.'''
        # Make a dummy variable to store the data we will use in this test
        test_data = None
        validation_data = None

        # Attempt to open the file with the method under test
        with open('test_cathode.gdf', 'rb') as f:
            test_data = load_initial_distribution(f, extra_screen_keys=['q', 'nmacro'])

        # Load the validation data
        with open('test_cathode.pickle', 'rb') as f:
            validation_data = pickle.load(f)

        # Make a new array to compare with
        modified_test_data = test_data[:-1]
        modified_test_data[-1] = modified_test_data[-1] * test_data[-1]

        # Compare them
        self.assertTrue(numpy.isclose(modified_test_data, validation_data, atol=1e-18).all())

    def test_load_curvilinear_screen(self):
        '''Some people use a custom GPT element to output screens in a curvilinear coordinates system.  These screens report the screen position as z=0, but list the particle z positions inside the GDF file which must be reported by the library.'''

        # Write down what we expect
        validation_data = numpy.array([[0., 0., 0.],
                                       [0., 0., 0.],
                                       [0., 0., 0.],
                                       [0., 0., 0.],
                                       [1., 2., 3.],
                                       [0., 0., 0.],
                                       [0., 0., 0.]])

        # Open up the file
        with open('curvilinear_screen.gdf', 'rb') as f:
            _, test_data = load(f)

        # Compare them
        self.assertTrue(numpy.isclose(test_data[0], validation_data).all())


################################################################################
# Script Begins Here
################################################################################
if __name__ == '__main__':
    unittest.main()

