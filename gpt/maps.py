import numpy as np
import os
import math
import tempfile
import subprocess

#from gpt import GPT
from gpt import tools
from gpt.tools import cvector
from gpt.tools import in_ecs
from gpt.element import Element
from gpt.element import Beg
from gpt.template import ztrack1_template

from matplotlib import pyplot as plt

from pmd_beamphysics import FieldMesh

from scipy.optimize import brent
import scipy.constants
from scipy.integrate import solve_ivp

MC2 = scipy.constants.value('electron mass energy equivalent in MeV')*1e6
c = scipy.constants.c
from math import pi

def gamma_to_beta(gamma):
    """ Converts relativistic gamma to beta"""
    return np.sqrt(1 - 1/gamma**2)

def beta_to_gamma(beta):
    """ Converts relativistic beta to gamma """
    return 1/np.sqrt(1-beta**2)

def KE_to_gamma(KE):
    """ Converts kinetic energy to relativistic gamma """
    return 1 + KE/mc2

def gamma_to_KE(gamma):
    """ Converts relativistic gamma to kinetic energy """
    return mc2*(gamma-1)

def KE_to_beta(KE):
    """ Converts kinetic energy to relativists beta """
    return gamma_to_beta(KE_to_gamma(KE))

def beta_to_KE(beta):
    """ Converts relativists beta to kinetic energy """
    return gamma_to_KE(beta_to_gamma(beta))

def p2e(p):
    return np.sqrt(p**2 + MC2**2)

def e2p(E):
    return np.sqrt(E**2 - MC2**2)

def e2g(E):
    return E/MC2

def e2b(E):
    return g2b(e2g(E))

def g2b(gamma):
    return np.sqrt(1-1/gamma**2)

def get_gdf_header(gdf_file, gdf2a_bin=os.path.expandvars('$GDF2A_BIN')):

    """Reads the header (column names) of gdf_file and returns them"""

    assert os.path.exists(gdf2a_bin), f'The gdf2a binary "{gdf2a_bin}" does not exist.'
    assert os.path.exists(gdf_file), f'The gdf file "{gdf_file}" does not exist'

    temp_ascii_file = f'{gdf_file}.temp.txt'
    cmd = f'{gdf2a_bin} -o {temp_ascii_file} {gdf_file}'
    subprocess.run(cmd, shell=True)

    with open(temp_ascii_file, 'r') as fp:
        columns = fp.readline().split()

    # cleanup    
    os.remove(temp_ascii_file)
    return columns

class GDFFieldMap(Element):

    """ General class for holding GDF field map data """

    def __init__(self, source_data_file, 
                 gdf2a_bin='$GDF2A_BIN', 
                 use_temp_file=True,
                 x0=0, y0=0, z0=0,
                 yaw=0, pitch=0, roll=0,
                 color='k'):

        super().__init__('GDFFieldMap',
                         x0=x0, y0=y0, z0=z0, 
                         yaw=yaw, pitch=pitch, roll=roll)
        
        assert os.path.exists(tools.full_path(gdf2a_bin)), f'GDF2A binary does not exist: {gdf2a_bin}'  

        self.source_data_file = tools.full_path(source_data_file)
        assert os.path.exists(self.source_data_file), f'Source GDF file {self.source_data_file} does not exist.' 
        
        remove_temp_gdf = False
        
        # If *.h5 file, load via FieldMesh, write a temporary GDF file and then load that
        if(source_data_file.endswith('.h5')):
            
            remove_temp_gdf = True
            
            FM = FieldMesh(source_data_file)
            temp_gdf_file = tempfile.NamedTemporaryFile().name + '.gdf'
            FM.write_gpt(temp_gdf_file, asci2gdf_bin='$ASCI2GDF_BIN', verbose=False)
            
            self.source_data_file = temp_gdf_file            
          
        if(use_temp_file):
            temp_ascii_file = tempfile.NamedTemporaryFile().name + '.txt'
        else:
            temp_ascii_file = f'{self.source_data_file}.temp.txt'

        cmd = f'{gdf2a_bin} -o {temp_ascii_file} "{self.source_data_file}"' 
        
        subprocess.run(cmd, shell=True)

        with open(temp_ascii_file, 'r') as fp:
            for i, line in enumerate(fp):
                if 'warning' in line.lower():
                    print(f'warning found in {temp_ascii_file}', line )
                else:
                    column_names = line.split()
                    break
        n_header = i+1
        self.column_names = column_names
        ndata = np.loadtxt(temp_ascii_file, skiprows=n_header)

        os.remove(temp_ascii_file)
        
        if(remove_temp_gdf):
            os.remove(temp_gdf_file)
            

        self.coordinates = [name for name in column_names if(name.lower() in ['r', 'x', 'y', 'z'])]
        self.field_components = [name for name in column_names if(name.lower() not in ['r', 'x', 'y', 'z'])]

        assert len(column_names)==len(self.coordinates) + len(self.field_components)

        for name in column_names:
            assert name in self.coordinates or name in self.field_components, f'Unknown variable: "{name}".'

        coordinate_sizes = {}
        coordinate_count_step = {}
            
        self.data={}

        # Get the coordinate vectors:
        for var in self.coordinates:
 
            v0 = ndata[0, column_names.index(var)]
            for ii, v in enumerate(ndata[:, column_names.index(var)]):
                if(v!=v0):
                    coordinate_count_step[var] = ii
                    break

            value = ndata[:, column_names.index(var)] 
            #setattr(self, var, value ) 
            #setattr(self, var, value)
            self.data[var]=value

            coordinate_sizes[var] = len(np.unique(ndata[:, column_names.index(var)]))

        coordinate_names = list(coordinate_count_step.keys())
        coordinate_steps = list(coordinate_count_step.values())

        sort_indices = np.argsort( np.array(coordinate_steps) )
        sorted_coordinate_names = [coordinate_names[index] for index in sort_indices]

        sorted_coordinate_sizes = [coordinate_sizes[sc] for sc in sorted_coordinate_names]
        data_shape = tuple(sorted_coordinate_sizes)
        alphabetical_order_coordinates = sorted(sorted_coordinate_names, key=str.lower)
        self.ordering = [alphabetical_order_coordinates.index(c) for c in sorted_coordinate_names]
        self.data_shape = data_shape

        for component in self.field_components:
            #setattr(self, component, ndata[:,column_names.index(component)])
            self.data[component]=ndata[:,column_names.index(component)]

    def __getitem__(self, key):

        column_names = [name.lower() for name in self.column_names]

        if(key.lower() in column_names):
            return self.data[self.column_names[column_names.index(key.lower())]]
            #return getattr(self, self.column_names[column_names.index(key.lower())])
        else:
            print(f'Field map does not contain item "{key}"')

    def is_in_map(self, var):

        column_names = [name.lower() for name in self.column_names]
        if(var.lower() in column_names):
            return True
        else:
            return False

        #return np.transpose(np.reshape(getattr(self, v), self.data_shape, 'F'), self.ordering)


    def scale_coordinates(self, scale):
        """ Scales all position coordinates in field map data by scale """
 
        for coordinate in self.coordinates:
            #setattr(self, coordinate, scale*getattr(self, coordinate))
            self.data[coordinate]=scale*self.data[coordinate]

    def scale_fields(self, scale):
        """ Scales all field components in field map data by scale """

        for component in self.field_components:
            #setattr(self, component, scale*getattr(self, component))
            self.data[component]=scale*self.data[component]

    def write_gdf(self, new_gdf_file, asci2gdf_bin='$ASCI2GDF_BIN', verbose=True):

        """ Writes a new GDF file"""

        temp_ascii_file = new_gdf_file + '.txt'

        data = np.zeros( (len(self.data[self.field_components[0]]), len(self.column_names)) )

        headers = []
        for ii, var in enumerate(self.coordinates+self.field_components):
            headers.append(var)

            #data[:,ii] = getattr(self, self.column_names[var])
            data[:,ii] = self.data[var]

        headers = '     '.join(headers)
        #np.savetxt(temp_ascii_file, data, header=headers, comments=' ')


        # Write file. 
        # Hack to delete final newline
        # https://stackoverflow.com/questions/28492954/numpy-savetxt-stop-newline-on-final-line
        with open(temp_ascii_file, 'w') as fout:
            #print('wooot', temp_ascii_file)
            NEWLINE_SIZE_IN_BYTES = 1 # 2 on Windows?
            np.savetxt(fout, data, header=headers, comments='')
            fout.seek(0, os.SEEK_END) # Go to the end of the file.
            # Go backwards one byte from the end of the file.
            fout.seek(fout.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
            fout.truncate() # Truncate the file to this point. 

        subprocess.run(f'{asci2gdf_bin} -o {new_gdf_file} {temp_ascii_file}', shell=True)
        os.remove(temp_ascii_file)

    def gpt_label_to_fieldmap_label(self, name):

        if(name in self.required_columns): 
            for cn in self.column_names:
                if(name.lower()==cn.lower()):
                    return cn
        else:
            return name

    def gpt_lines(self, ccs=None, gdf_file=None, e1=[1, 0, 0], e2=[0, 1, 0], scale =None, user_vars=[]):

        """ Creates ASCII lines defininig field map element in GPT syntax """

        element = self._name

        if(ccs is None):
            ccs = self.ccs_beg

        ds = in_ecs(self._p_beg, self._ccs_beg_origin, self.M_beg)[2]
 
        ccs_beg_e3 = cvector([0,0,1])
        r = ds*ccs_beg_e3

        map_line = f'{self.type}("{self.ccs_beg}", "GxyzXYZ", '
        extra_lines={}

        for ii, coordinate in enumerate(['x', 'y', 'z']):
            #if(coordinate in user_vars):
            if(coordinate=='z'):
                val = r[ii][0]-self['z'][0]
                #print(self['z'][0])
            else:
                val = r[ii][0]
            extra_lines[coordinate] = f'{element}_{coordinate} = {val};'
            map_line = map_line + f'{element}_{coordinate}, '
            #else:
            #    map_line = map_line + f'{str(r[ii][0])}, '

        extra_lines['yaw'] = f'{element}_yaw = {self._ecs["yaw"]};'
        extra_lines['pitch'] = f'{element}_pitch = {self._ecs["pitch"]};'
        extra_lines['roll'] = f'{element}_roll = {self._ecs["roll"]};'

        for v in ['yaw', 'pitch', 'roll']:

            map_line = map_line + f'{element}_{v}, '

        #for ii, m in enumerate(['e11', 'e12', 'e13']):
        #    if(m in user_vars):
        #        extra_lines[m] = f'{element}_{m} = {e1[ii]};'
        #        map_line = map_line + f'{element}_{m}, '
        #    else:
        #        map_line = map_line + f'{str(e1[ii])}, '

        #for ii, m in enumerate(['e21', 'e22', 'e23']):
        #    if(m in user_vars):
        #        extra_lines[m] = f'{element}_{m} = {e2[ii]};'
        #        map_line = map_line + f'{element}_{m}, '
        #    else:
        #        map_line = map_line + f'{str(e2[ii])}, '

        if(gdf_file is None):
            gdf_file = self.source_data_file

        map_line = map_line + f'"{gdf_file}", '

        for ii, rc in enumerate(self.required_columns):

            name = self.gpt_label_to_fieldmap_label(rc)
            map_line = map_line + f'"{name}", '
        
        if(scale is None):
            scale = self._scale

        #if('scale' in user_vars):
        extra_lines['scale'] = f'{element}_scale = {scale};'
        map_line = map_line + f'{element}_scale);'
        #else:
        #    map_line = map_line + f'{scale});'

        return [extra_line for extra_line in extra_lines.values()] + [map_line]

    def to_dict(self):

        desc = {'type':self._type, 
        's_beg': float(self.s_beg), 
        's_end': float(self.s_end), 
        's': float(0.5*(self.s_beg + self.s_end)),
        'fieldmap': os.path.basename(self.source_data_file)}

        return desc

class Map1D(GDFFieldMap):

    """ Class for storing 1D GDF field maps, derives from GDFfieldMap """

    def __init__(self, source_data, required_columns, 
                 gdf2a_bin='$GDF2A_BIN', 
                 x0=0, y0=0, z0=0,
                 yaw=0, pitch=0, roll=0,
                 color=None):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, 
                         x0=x0, y0=y0, z0=z0,
                         yaw=yaw, pitch=pitch, roll=roll)

        self._type = 'Map1D'

        self.required_columns = required_columns
        assert 'z' in required_columns

        if('Ez' in required_columns):
            self.Fz_str='Ez'
            self.Fz_unit='V/m'
        else:
            self.Fz_str='Bz'
            self.Fz_unit='T'

        self._color=color

    def plot_floor(self, axis=None, alpha=1.0, ax=None, xlim=None, ylim=None, style='tao'):
        """ Plots the bounding box of the field map in z-x floor coordinates """
        plot_clyindrical_map_floor(self, axis=axis, alpha=alpha, ax=ax, xlim=xlim, ylim=ylim, style=style)

    def plot_field_profile(self, ax=None, normalize=False):
        """ Plots the z, Fz field profile on axis """
        plot_clyindrical_map_field_profile(self, ax=ax, normalize=normalize)

    @property
    def field_integral(self):
        """ Returns the on axis integral of the Fz """
        return np.trapezoid(self.Fz, self.z0)

    @property
    def z0(self):
        """ Returns the vector of unique z points in map"""
        return self['z']

    @property
    def Fz(self):
        """ Returns the on axis field z component """
        return self[self.Fz_str]*self._scale

    @property
    def max_abs_Fz(self):
        return np.max(np.abs(self._scale*self.Fz))
    

class Map1D_E(Map1D):

    """
    Defines a 1D [z, Ez] cylindrically symmetric electric field map object
    """

    def __init__(self, name, source_data, 
                 gdf2a_bin='$GDF2A_BIN', 
                 width=0.3, 
                 scale=1, 
                 color='#1f77b4', 
                 style=None,
                 yaw=0, pitch=0, roll=0
                ):

        super().__init__(source_data, 
                         gdf2a_bin=gdf2a_bin, 
                         required_columns=['z', 'Ez'],
                         x0=x0, y0=y0, z0=z0,
                         yaw=yaw, pitch=pitch, roll=roll)
       
        self._name = name
        self._type = 'Map1D_E'
        self._length = self.z0[-1]-self.z0[0]
        self._width = 0.2
        self._height = self._width
        self._style = style

        if(style=='tao'):
            color='#1f77b4'

        self._color=color
        self._scale = scale

        self.place() 

    def place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg'):
        place(self, ref_element=ref_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)

    @property
    def Ez0(self):
        return self.Fz

    def to_dict(self):

        desc = super().to_dict()
        desc['max(|Ez|)'] = float(self.max_abs_Fz)

        return desc

    def fields(self, x, y, s, t):
        fields = np.zeros( (6, len(s)) )
        fields[2,:] = np.interp(s, self.z0, self.Ez0)
        return fields

    def track_on_axis_rk45(self, t, p):

        pass



    def energy_gain(self, t, E):

        zspan = (self.z0[0],  self.z0[-1])
        z = np.linspace(zspan[0], zspan[-1], 100)

        dat = solve_ivp(fun, zspan, [t, E], t_eval=z)

        z = dat['t']
        u = dat['y']

        t = u[0,:]
        E = u[1,:]

        return (z, t, E)


def fun(z, u):

    t = u[0]
    E = u[1]

    b = E2b(E)
    dtdz = 1/b/c
    dEdz = 500e3

    return [dtdz, dEdz]



class Map1D_B(Map1D):

    """
    Defines a 1D [z, Bz] cylindrically symmetric magnetic field map object
    """

    def __init__(self, name, source_data, 
                 gdf2a_bin='$GDF2A_BIN', 
                 width=0.4, 
                 scale=1, 
                 color='#2ca02c', 
                 style='tao',
                 x0=0, y0=0, z0=0,
                 yaw=0, pitch=0, roll=0):

        super().__init__(source_data, 
                         gdf2a_bin=gdf2a_bin, 
                         required_columns=['z', 'Bz'],
                         x0=x0, y0=y0, z0=z0,
                         yaw=yaw, pitch=pitch, roll=roll)
        
        self._name = name
        self._type = 'Map1D_B'
        self._length = self.z0[-1]-self.z0[0]
        self._width = width
        self._height = self._width
        self._style = style

        if(style=='tao'):
            color='blueviolet'

        self._color = color


        self._scale=scale

        self.place()

    def place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg'):
        place(self, ref_element=ref_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)

    @property
    def Bz0(self):
        return self.Fz

    def larmor_angle(self, p=1e6):
        return larmor_angle(self, p)

    @property
    def Bz2_integral(self):
        return np.trapezoid(self.Fz**2, self.z0)

    def to_dict(self):

        desc = super().to_dict()
        desc['max(|Bz|)'] = float(self.max_abs_Fz)

        return desc


class Map1D_TM(Map1D):

    """
    Defines a 1D [z, Ez] cylindrically symmetric TM cavity field map object
    """

    def __init__(self, 
        name, 
        source_data, 
        frequency, 
        scale=1,
        relative_phase=0,
        oncrest_phase=0,
        gdf2a_bin='$GDF2A_BIN', 
        color='darkorange',
        x0=0, y0=0, z0=0,
        
        style=None):

        super().__init__(source_data, 
                         gdf2a_bin=gdf2a_bin, 
                         required_columns=['z', 'Ez'],
                         x0=x0, y0=y0, z0=z0,
                         yaw=yaw, pitch=pitch, roll=roll)

        self._name = name
        self._type='Map1D_TM'
        self._length = self.z0[-1]-self.z0[0]
        self._width = 0.2
        self._height = self._width
        self._style=style

        if(style=='tao'):
            color = 'lawngreen'

        self._color = color

        self._frequency = frequency
        self._w = 2*math.pi*frequency
        self._oncrest_phase=oncrest_phase
        self._relative_phase=relative_phase
        self._scale=scale

        self.place()

    def place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg'):
        place(self, ref_element=ref_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)

    @property
    def Ez0(self):
        return self.Fz

    def plot_field_profile(self, ax=None, normalize=False):
        return plot_clyindrical_map_field_profile(self, ax=ax, normalize=normalize)

    def track_on_axis(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, n_screen=1, workdir=None):
        return track_on_axis(self, t, p, xacc=xacc,  GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, n_screen=n_screen, workdir=workdir)

    def autophase_track1(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, oncrest_phase=0, workdir=None, n_screen=1):
        return autophase_track1(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, oncrest_phase=oncrest_phase, workdir=workdir, n_screen=n_screen)

    def autophase(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, workdir=None, n_screen=100, verbose=False):
        return autophase(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, workdir=workdir, n_screen=n_screen, verbose=verbose)

    def gpt_lines(self):

        name = self.name

        base_lines = super().gpt_lines()
        extra_lines = base_lines[:-1]
        map_line = base_lines[-1].replace(');','')

        extra_lines.append(f'{name}_oncrest_phase = {self._oncrest_phase};')
        extra_lines.append(f'{name}_relative_phase = {self._relative_phase};')
        extra_lines.append(f'{name}_phase = ({name}_oncrest_phase + {name}_relative_phase)*pi/180;')

        map_line = map_line + f', {name}_phase, '

        extra_lines.append(f'{name}_frequency = {self._frequency};')
        extra_lines.append(f'{name}_gamma = 1;')
        map_line = map_line + f'2*pi*{name}_frequency);'

        return extra_lines + [map_line]

    @property
    def relative_phase(self):
        return self._relative_phase

    @relative_phase.setter
    def relative_phase(self, phi):
        self._relative_phase=phi

    def to_dict(self):

        desc = super().to_dict()
        desc['max(|Ez|)'] = float(self.max_abs_Fz)

        return desc

    @property
    def cavity_voltage(self):
        return cavity_voltage(self)


class Map2D(GDFFieldMap):

    """ Base class for all 2D cylindrically symmetric fields pointed along z """

    def __init__(self, source_data_file, required_columns, gdf2a_bin='$GDF2A_BIN'):

        super().__init__(source_data_file, gdf2a_bin=gdf2a_bin)

        self.required_columns=required_columns

        assert 'z' in required_columns
        assert 'r' in required_columns

        if('Ez' in required_columns):
            self.Fz_str='Ez'
            self.Fz_unit='V/m'
        else:
            self.Fz_str='Bz'
            self.Fz_unit='T'

    def plot_floor(self, axis=None, alpha=1.0, ax=None, xlim=None, ylim=None, style='tao'):
        """ 
        Function for plotting the relevant field/object region for cylindrically symmetric map object
        Inputs: 
            axis: None or str('equal') to set axes to same scale
            alpha: alpha parameter to matplotlib pyplot.plot
            ax: matplotlib axis object, if None provided usess gca().
        Outputs:
            ax: returns current axis handle being used
        """
        
        return plot_clyindrical_map_floor(self, axis=axis, alpha=alpha, ax=ax, xlim=xlim, ylim=ylim, style=style)

    def plot_field_profile(self, ax=None, normalize=False):
        """ 
        Function for plotting the Ez or Bz on axis profile for cylindrically symmetric field map
        Inputs: 
            element (object, Map2D_* )
            ax: matplotlib axis object, if None provided usess gca().
            normalize: boolean, normalize field to 1 or not
        Outputs:
            ax: returns current axis handle being used
        """
        return plot_clyindrical_map_field_profile(self, ax=ax, normalize=normalize)

    @property
    def field_integral(self):
        """
        Computs the on axis field integral through a cylindrically symmetric field map 
        Outputs:
            float, integral ( Fz(r=0) dz)
        """
        return np.trapezoid(self.Fz, self.z0)

    @property
    def z0(self):

        z = self['z']
        r = self['r']

        return np.squeeze(z[r==0])

    @property
    def r(self):
        return np.unique(self['r'])

    @property
    def Fz(self):
        return np.squeeze(self[self.Fz_str][self['r']==0])

    def write_1d_map(self, filename=None):
        write_1d_map(self, filename=filename)

    @property
    def max_abs_Fz(self):
        return float(np.max(np.abs(self._scale*self.Fz)))
    

    #def track_on_axis(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, n_screen=1, workdir=None):
    #    return track_on_axis(self, t, p, xacc=xacc,  GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, n_screen=n_screen, workdir=workdir)

class Map2D_E(Map2D):

    """
    Defines a 2D (r,z), (Er, Ez) cylindrically electric symmetric field map object
    """
    
    def __init__(self, name, source_data, gdf2a_bin='$GDF2A_BIN', scale=1, r=[0,0,0], style=None, color='#1f77b4'):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, required_columns=['r', 'z', 'Er', 'Ez'])

        self._name = name
        self._type = 'Map2D_E'
        self._length = self.z0[-1]-self.z0[0]
        self._width = 0.3
        self._height = self._width
        self._style = style

        if(style=='tao'):
            color=color

        self._color = color
        self._scale=scale

        self.place() 

    def place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg'):
        place(self, ref_element=ref_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)

    @property
    def Ez0(self):
        return self.Fz

    def to_dict(self):

        desc = super().to_dict()
        desc['max(|Ez|)'] = float(self.max_abs_Fz)

        return desc


class Map2D_B(Map2D):

    """
    Defines a 2D (r,z), (Br, Bz) cylindrically magnetic symmetric field map object
    """

    def __init__(self, name, source_data, gdf2a_bin='$GDF2A_BIN', field_pos='center', scale=1, style='tao', color='#2ca02c'):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, required_columns=['r', 'z', 'Br', 'Bz'])

        self._type='Map2D_B'
        self._name = name
        self._length = self.z0[-1]-self.z0[0]
        self._width = 0.4
        self._height = self._width
        self._style = style

        if(style=='tao'):
            color = 'blueviolet'

        self._color = color
        self._scale=scale

        self.place()

    def place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg'):
        place(self, ref_element=ref_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)

    def larmor_angle(self, p):
        """ 
        Computs the Larmor angle through a solenoid field map 
        Inputs: 
            p: float, momentum in [eV/c]
        Outputs:
            float, larmor angle in [rad]
        """
        return larmor_angle(self, p)

    @property
    def Bz0(self):
        return self.Fz

    def to_dict(self):

        desc = super().to_dict()
        desc['max(|Bz|)'] = float(self.max_abs_Fz)

        return desc


class Map25D_TM(Map2D):

    def __init__(self, 
        name,
        source_data, 
        frequency, 
        scale=1,
        relative_phase=0,
        oncrest_phase=0,
        gdf2a_bin='$GDF2A_BIN', 
        column_names={'z':'z', 'r':'r', 'Ez':'Ez', 'Er':'Er', 'Bphi':'Bphi'}, 
        required_columns=['r', 'z', 'Er', 'Ez', 'Bphi'],
        e1=[1, 0, 0], 
        e2=[0, 1, 0],
        k=0,
        color='darkorange',
        style=None):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, required_columns=required_columns)

        self._name=name
        self._type='Map25D_TM'

        self._scale = scale
        self._relative_phase = relative_phase

        self._frequency = frequency
        self._w = 2*math.pi*frequency

        self._oncrest_phase=oncrest_phase

        self._length = self.z0[-1]-self.z0[0]
        self._width = 0.2
        self._height = self._width
        self._style=style
        if(style=='tao'):
            color='lawngreen'

        self._color = color

        self._k=k

        self.Fz_str='Ez'
        self.Fz_unit='V/m'

        self._e1=e1
        self._e2=e2

        self.place()

    def place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg'):
        place(self, ref_element=ref_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)

    #@property
    #def on_axis_Ez(self):
    #    return np.squeeze(self.Ez[self.r==0])

    #def integrate_Ez(self, phi):
    #    return integrate_Ez(self['z'], self.on_axis_Ez, self._w, phi, self._beta)

    #def energy_gain(self, phi):
    #    return energy_gain(self['z'], self.on_axis_Ez, self._w, phi, self._beta)

    #@property
    #def oncrest_phase(self):
    #    return (-cmath.phase(self.integrate_Ez(0))*180/math.pi)%360

    #@property
    #def oncrest_energy_gain(self):
    #    return self.energy_gain(self.oncrest_phase)

    #@property
    #def beta(self):
    #    return self._beta

    #@beta.setter
    #def beta(self, beta):
    #    self._beta = beta
    #    self._kinetic_energy = beta_to_KE(beta)

    #@property
    #def kinetic_energy(self):
    #    return self._kinetic_energy

    #@kinetic_energy.setter
    #def kinetic_energy(self, kinetic_energy):
    #    self._kinetic_energy = kinetic_energy
    #    self._beta = KE_to_beta(kinetic_energy)

    @property
    def Ez0(self):
        return self.Fz

    @property
    def scale(self):
        return self._scale

    def gpt_lines(self, oncrest_phase=None, relative_phase=None, gdf_file=None):

        name = self.name

        base_lines = super().gpt_lines(e1=self._e1, e2=self._e2, gdf_file=gdf_file)
        extra_lines = base_lines[:-1]
        map_line = base_lines[-1].replace(');','')

        if(oncrest_phase is None):
            oncrest_phase = self._oncrest_phase

        if(relative_phase is None):
            relative_phase = self._relative_phase

        extra_lines.append(f'{name}_oncrest_phase = {oncrest_phase};')
        extra_lines.append(f'{name}_relative_phase = {relative_phase};')
        extra_lines.append(f'{name}_phase = ({name}_oncrest_phase + {name}_relative_phase)*pi/180;')
        extra_lines.append(f'{name}_k = {self._k};')
        extra_lines.append(f'{name}_gamma = 1;')

        map_line = map_line + f', {name}_k, {name}_phase, '
        extra_lines.append(f'{name}_frequency = {self._frequency};')
        map_line = map_line + f'2*pi*{name}_frequency);'

        lines = extra_lines + [map_line]

        return lines

    def to_dict(self):

        desc = super().to_dict()
        desc['max(|Ez|)'] = self.max_abs_Fz
        desc['frequency'] = self._frequency
        desc['relative_phase'] = self._relative_phase

        return desc

    def track_on_axis(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, n_screen=1, workdir=None):
        return track_on_axis(self, t, p, xacc=xacc,  GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, n_screen=n_screen, workdir=workdir)

    def autophase_track1(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, oncrest_phase=0, workdir=None, n_screen=1):
        return autophase_track1(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, oncrest_phase=oncrest_phase, workdir=workdir, n_screen=n_screen)

    def autophase(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, workdir=None, n_screen=100, verbose=False):
        return autophase(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, workdir=workdir, n_screen=n_screen, verbose=verbose)

    @property
    def cavity_voltage(self):
        return cavity_voltage(self)

    def to_superfish_t7(self, filename, column_names={'Ez':'Ez', 'Er':'Er', 'E':'E', 'Bphi':'Bphi'}, fmt='%10.8e'):

        zmin, zmax, nz = self.z0[0]*100, self.z0[-1]*100, len(self.z0)   # z [cm]
        rmin, rmax, nr = self.r[0]*100,  self.r[-1]*100,  len(self.r)    # r [cm]

        header=f'{zmin} {zmax} {nz-1}\n{self._frequency/1e6}\n{rmin} {rmax} {nr-1}'

        scales = {'Ez':1e6, 'Er':1e6, 'E':1e6, 'Bphi':1}

        dat=np.zeros( (nz*nr, 4) )
        for ii, k in enumerate(column_names):
            v = column_names[k]
            dat[:,ii] = self[v]/scales[k]

        np.savetxt(filename, dat, header=header, comments='',  fmt = fmt)




class Map3D_E(GDFFieldMap):

    def __init__(self, 
        name, 
        source_data, 
        scale=1,
        gdf2a_bin='$GDF2A_BIN', 
        column_names={'x':'x', 'y':'y', 'z':'z', 'Ex':'Ex', 'Ey':'Ey', 'Ez':'Ez'}, 
        required_columns=['x', 'y', 'z', 'Ex', 'Ey', 'Ez'],
    #             e1=[1, 0, 0], 
    #             e2=[0, 1, 0],
        color='tab:blue',
        style=None):

        self.column_names = column_names

        GDFFieldMap.__init__(self, source_data, gdf2a_bin=gdf2a_bin, use_temp_file=True)

        self._name = name
        self._scale = scale
        self._type = 'Map3D_E'
        self._length = max(self['z']) - min(self['z'])
        self._width =  max(self['x']) - min(self['x'])
        self._height = max(self['y']) - min(self['y']) 
        self._style = style
        self._color = color

        #self.column_names = column_names
        self.required_columns = required_columns

    @property
    def x(self):
        return np.unique(self['x'])

    @property
    def y(self):
        return np.unique(self['y'])

    @property
    def z(self):
        return np.unique(self['z'])

    @property
    def Ex(self):
        return np.unique(self['Ex'])

    @property
    def Ey(self):
        return np.unique(self['Ey'])

    @property
    def Ez(self):
        return np.unique(self['Ez'])

    @property
    def Ez0(self):
        return self['Ez'][ (np.isclose(self['x'], 0)) & (np.isclose(self['y'], 0))]

    @property
    def z0(self):
        return self['z'][ (np.isclose(self['x'], 0)) & (np.isclose(self['y'], 0))]
    

    def plot_field_profile(self, ax=None, normalize=True):

        if ax is None:
            ax=plt.gca()

        if normalize:
            f = 1/max(np.abs(self.Ez0))
        else:
            f=1

        ax.plot(self.z, f*self.Ez0)
        ax.set_xlabel('z (m)')
        if normalize:
            ax.set_ylabel('Field Profile (arb)')
        else:
            ax.set_ylabel('$E_z(x=y=0)$ (V/m)')

        return ax


class Map3D_B(GDFFieldMap):

    def __init__(self, 
        name, 
        source_data, 
        scale=1,
        gdf2a_bin='$GDF2A_BIN', 
        column_names={'x':'x', 'y':'y', 'z':'z', 'Bx':'Bx', 'By':'By', 'Bz':'Bz'}, 
        required_columns=['x', 'y', 'z', 'Bx', 'By', 'Bz'],
        yaw=0, pitch=0, roll=0,
        color='tab:blue',
        style=None):

        self.column_names = column_names

        GDFFieldMap.__init__(self, source_data, 
                             gdf2a_bin=gdf2a_bin, 
                             yaw=yaw, pitch=pitch, roll=roll,
                             use_temp_file=True)

        self._name = name
        self._scale = scale
        self._type = 'Map3D_B'
        self._length = max(self['z']) - min(self['z'])
        self._width =  max(self['x']) - min(self['x'])
        self._height = max(self['y']) - min(self['y']) 
        self._style = style
        self._color = color

        #self.column_names = column_names
        self.required_columns = required_columns

    @property
    def x(self):
        return np.unique(self['x'])

    @property
    def y(self):
        return np.unique(self['y'])

    @property
    def z(self):
        return np.unique(self['z'])

    @property
    def Bx(self):
        return np.unique(self['Ex'])

    @property
    def By(self):
        return np.unique(self['Ey'])

    @property
    def Bz(self):
        return np.unique(self['Ez'])

    @property
    def By0(self):
        return self['By'][ (np.isclose(self['x'], 0)) & (np.isclose(self['y'], 0))]
    
    @property
    def Bz0(self):
        return self['Bz'][ (np.isclose(self['x'], 0)) & (np.isclose(self['y'], 0))]

    @property
    def z0(self):
        return self['z'][ (np.isclose(self['x'], 0)) & (np.isclose(self['y'], 0))]
    

    def plot_field_profile(self, ax=None, normalize=True):

        if ax is None:
            ax=plt.gca()

        if normalize:
            f = 1/max(np.abs(self.Bz0))
        else:
            f=1

        ax.plot(self.z, f*self.Bz0)
        ax.set_xlabel('z (m)')
        if normalize:
            ax.set_ylabel('Field Profile (arb)')
        else:
            ax.set_ylabel('$E_z(x=y=0)$ (V/m)')

        return ax
    


    

        
    






def write_1d_map(element, filename=None, asci2gdf_bin=os.path.expandvars('$ASCI2GDF_BIN')):

    if(element.type in ['Map2D_E', 'Map2D_B', 'Map25D_TM']):

        data = np.zeros( (len(element.z0), 2))

        data[:,0] = element.z0
        data[:,1] = element.Fz

        fstr = element.Fz_str
        zstr = 'z'

        for column in element.column_names:
            if(column.lower()=='z'):
                zstr=column

        header = zstr+'     '+fstr

        if(filename is None):
            tempfile = element.name+'_1D.txt'
            filename = tempfile.replace('.txt', '.gdf')

        elif(filename.endswith('.gdf')):
            tempfile = filename.replace('.gdf', '.txt')

        with open(tempfile, 'w') as fout:

            NEWLINE_SIZE_IN_BYTES = 1

            np.savetxt(tempfile, data, header=header, comments=' ')
            fout.seek(0, os.SEEK_END) # Go to the end of the file.
            # Go backwards one byte from the end of the file.
            fout.seek(fout.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
            fout.truncate() # Truncate the file to this point.

        subprocess.run(f'{asci2gdf_bin} -o {filename} {tempfile}', shell=True)
        os.remove(tempfile)




def place(ele, ref_element=None, ds=0, ref_origin='end', element_origin='origin'):

    if(ref_element is None):
        ref_element=Beg()

    if(ds>=0):

        e3 = ref_element.e3_end
        M = ref_element.M_end
        ele._ccs_beg = ref_element.ccs_end 
        ele._ccs_beg_origin = ref_element.ccs_end_origin

    else:

        e3 = ref_element.e3_beg
        M = ref_element.M_beg
        ele._ccs_beg = ref_element.ccs_beg  
        ele._ccs_beg_origin = ref_element.ccs_beg_origin

    ele._M_beg = M
    ele._M_end = M

    if(ref_origin=='end'):

        s_ref = ref_element.s_end
        p_ref = ref_element.p_end

    elif(ref_origin=='center'):

        s_ref = ref_element.s_beg + ref_element.length/2.0
        p_ref = ref_element.p_beg + (ref_element.length/2.0)*e3 

    else:

        s_ref = ref_element.s_beg
        p_ref = ref_element.p_beg

    if(element_origin=='beg'):
        
        ele._s_beg = s_ref + ds
        ele._s_end = ele.s_beg + ele.length

        ele._p_beg = p_ref + ds*e3
        ele._p_end = ele.p_beg + ele.length*e3

    elif(element_origin=='center'):

        ele._s_beg = s_ref + ds - ele.length/2.0
        ele._s_end = ele.s_beg + ele.length

        ele._p_beg = p_ref + (ds - ele.length/2.0)*e3
        ele._p_end = ele.p_beg + ele.length*e3 

    elif(element_origin=='origin'):

        ele._s_beg = s_ref + ds + ele['z'][0]
        ele._s_end = ele.s_beg + ele.length

        ele._p_beg = p_ref + (ds + ele['z'][0])*e3
        ele._p_end = ele.p_beg + ele.length*e3 

    ele._ccs_beg = ref_element.ccs_end
    ele._ccs_end = ref_element.ccs_end  # Straight line, no ccs flips

    #print(ele._ccs_beg)

    ele._ds = np.linalg.norm(ele._p_beg - ele._ccs_beg_origin)
    ele.set_ref_trajectory()


def plot_clyindrical_map_floor(element, axis=None, alpha=1.0, ax=None, xlim=None, ylim=None, style=None):

    """ 
    Function for plotting the relevant field/object region for cylindrically symmetric map object
    Inputs: 
        element (object, Map1D_* or Map2D_* or Map25D_TM)
        axis: None or str('equal') to set axes to same scale
        alpha: alpha parameter to matplotlib pyplot.plot
        ax: matplotlib axis object, if None provided usess gca().
    Outputs:
        ax: returns current axis handle being used
    """

    if(style is not None):
        element._style=style

    f = 0.01

    zs = element.z0
    p00 = np.array([[0],[0],[0]])
    if(hasattr(element, '_e1') and hasattr(element, '_e2')):
        e3 = np.cross(element._e1, element._e2)
        if(e3[2]==-1):
            zs = np.flip(element.z0) 
            p00 = np.array([[0],[0],[-element.z0[-1]]])


    Fz = element.Fz

    maxF = max(np.abs(Fz))

    zL = zs[0]
    if(abs(Fz[0])<f*maxF):
        for ii, z in enumerate(zs):
            if(np.abs(Fz[ii]) >= f*maxF):
                zL = z
                break

    zs = np.flip(zs)
    Fz = np.flip(Fz)

    for ii,z in enumerate(zs):
        if(np.abs(Fz[ii]) >= f*maxF):
            zR = z
            break

    effective_plot_length = zR-zL

    #print(effective_plot_length)

    if ax is None:
        ax = plt.gca()

    #pc = 0.5*(element.p_beg + element.p_end)

    p1 = element.p_beg + (element._width/2)*element.e1_beg + p00
    p2 = element.p_beg - (element._width/2)*element.e1_beg + p00
    p3 = element.p_end + (element._width/2)*element.e1_beg + p00
    p4 = element.p_end - (element._width/2)*element.e1_beg + p00

    ps1 = np.concatenate( (p1, p3, p4, p2, p1), axis=1)

    p0 = (zL-element.z0[0])*element.e3_beg + element.p_beg + p00

    p1 = p0 + (element._width/2)*element.e1_beg 
    p2 = p1 + effective_plot_length*element.e3_beg
    p3 = p2 - (element._width)*element.e1_beg
    p4 = p3 - effective_plot_length*element.e3_beg

    ps2 = np.concatenate( (p1, p2, p3, p4, p1), axis=1)

    #print(element.color)

    if(element._style=='tao'):
        ax.plot(ps1[2], ps1[0], element.color, alpha=0.2*alpha)
        ax.plot(ps2[2], ps2[0], element.color, alpha=alpha)

        ax.plot([p1[2], p3[2]], [p1[0], p3[0]], element.color, alpha=alpha)
        ax.plot([p2[2], p4[2]], [p2[0], p4[0]], element.color, alpha=alpha)
    #ax.plot([ps2[]], )

    ax.set_xlabel('z (m)')
    ax.set_ylabel('x (m)')

    if(axis=='equal'):
        ax.set_aspect('equal')
        
    if(xlim is not None):
        ax.set_xlim(xlim)

    return ax


def plot_clyindrical_map_field_profile(element, ax=None, normalize=False):
  
    """ 
    Function for plotting the Ez or Bz on axis profile for cylindrically symmetric field map
    Inputs: 
        lement (object, Map1D_* or Map2D_* or Map25D_TM)
        ax: matplotlib axis object, if None provided usess gca().
        normalize: boolean, normalize field to 1 or not
    Outputs:
        ax: returns current axis handle being used
    """

    if ax is None:
        ax = plt.gca()

    Fz = element.Fz

    if(hasattr(element, '_e1') and hasattr(element, '_e2')):

        e3 = np.cross(element._e1, element._e2)
        if(e3[2]==-1):
            zs = element.s_beg + np.flip(element.z0) - element.z0[-1]
        else:
            zs = element.s_beg + element.z0 - element.z0[0]

    else:
        zs = element.s_beg + element.z0 - element.z0[0]

    if(normalize):

        Fz0 = np.max(np.abs(Fz))

        index = np.where( np.abs(Fz) == Fz0 )[0][0]

        signFz = np.sign(Fz[index])

        Fz = signFz*Fz/Fz0

    ax.plot(zs, Fz, element._color)
    ax.set_xlabel('s (m)')

    if(not normalize):
        ax.set_xlabel('z (m)')
        ax.set_ylabel(f'{element.Fz_str} ({element.Fz_unit})')
    
    return ax

def larmor_angle(solenoid, p):
    '''
    Computs the Larmor angle through a solenoid field map 
    Inputs:
        solenoid: Map[1,2]D_B object
        p: float, particle momentum
    Outputs:
        theta: float, larmor angle [rad]
    '''

    return -c*solenoid.field_integral/p/2.0

def electrostatic_energy_gain(efield):
    '''
    Computs the Larmor angle through a solenoid field map 
    Inputs:
        solenoid: Map[1,2]D_B object
        p: float, particle momentum
    Outputs:
        theta: float, larmor angle [rad]
    '''
    pass

def cavity_voltage(cavity):

    zs = cavity.z0
    f = cavity._frequency
    w = 2*pi*f

    Ez = cavity.Ez0*np.exp(1j*w*zs/c)

    return np.absolute( np.trapezoid(Ez , zs) )


def track_on_axis(element, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, n_screen=1, workdir=None):

    if(workdir is None):
        tempdir = tempfile.TemporaryDirectory(dir=workdir)  
        gpt_file = os.path.join(tempdir.name, f'{element.name}.gpt.in')
        workdir = tempdir.name

    else:

        gpt_file = os.path.join(workdir, f'{element.name}.gpt.in' )

    element.write_element_to_gpt_file(ztrack1_template(gpt_file))

    G = GPT(gpt_file, ccs_beg=element.ccs_beg, workdir=workdir, use_tempdir=False)
    G.set_variables({'xacc':xacc, 'GBacc':GBacc, 'dtmin':dtmin, 'dtmax':dtmax})

    z_beg = np.linalg.norm(element._ccs_beg_origin - element._p_beg)  

    G.track1_in_ccs(z_beg=z_beg, 
    z_end=z_beg+element.length, 
    ccs=element.ccs_beg, 
    pz0=p, 
    t0=t, 
    weight=1, 
    status=1, 
    species='electron',
    xacc=xacc,
    GBacc=GBacc,
    n_screen=n_screen)

    return G

def autophase_track1(cavity, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, oncrest_phase=0, workdir=None, n_screen=1):

    cavity._oncrest_phase = oncrest_phase

    G = cavity.track_on_axis(t, p,  
        xacc=xacc, 
        GBacc=GBacc, 
        dtmin=dtmin, 
        dtmax=dtmax,
        workdir=workdir,
        n_screen=1)

    if(G.n_screen>=1 and len(G.screen[-1]['x'])==1):
        energy_gain = G.screen[-1]['mean_energy']-p2e(p)
    else:
        energy_gain = -8e88

    return -energy_gain

def autophase(cavity, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, workdir=None, n_screen=100, verbose=True):

    """ Auto phases a cavity for a particle entering the fieldmap at time = t with total momentum = p """

    if(verbose):

        print(f'\n> Phasing: {cavity.name}')
        print(f'   t_beg = {t} sec.')
        print(f'   s_beg = {cavity.s_beg} m.')
        print(f'   scale = {cavity._scale}.')
        print(f'   relative phase = {cavity._relative_phase} deg.')

    cavity._oncrest_phase=0
    cavity._momentum_beg=p
    cavity._t_beg = t

    relative_phase = cavity._relative_phase

    cavity._relative_phase=0

    oncrest_phase = brent(lambda x: cavity.autophase_track1(t, p,  
        xacc=xacc, 
        GBacc=GBacc, 
        dtmin=dtmin, 
        dtmax=dtmax,
        oncrest_phase=x,
        workdir=workdir,
        n_screen=1), brack=(-180, 180))

    cavity._oncrest_phase =  (oncrest_phase +180) % (360) - 180

    G = cavity.track_on_axis(t, p,  
        xacc=xacc, 
        GBacc=GBacc, 
        dtmin=dtmin, 
        dtmax=dtmax,
        workdir=workdir,
        n_screen=1)

    if(G.n_screen>=1):
        if(G.screen[-1]['mean_pz']<0):
            raise ValueError(f'Autophasing {cavity._name} failed: particle has pz<0 at oncresst phase.')
    else:
        raise ValueError(f'Autophasing {cavity._name} failed: no data found when cavity tracked with resulting oncrest phase = {oncrest_phase}.')

    cavity._relative_phase=relative_phase

    G = cavity.track_on_axis(t, p,  
        xacc=xacc, 
        GBacc=GBacc, 
        dtmin=dtmin, 
        dtmax=dtmax,
        workdir=workdir,
        n_screen=n_screen)

    if(G.n_screen>=1):
        if(G.screen[-1]['mean_pz']<0):
            raise ValueError(f'Autophasing {cavity._name} failed: particle has pz<0 at relative phase.')
    else:
        raise ValueError(f'Autophasing {cavity._name} failed: no data found when cavity tracked with relative phase = {relative_phase}.')

    cavity._momentum_end=G.screen[-1]['mean_p']
    cavity._t_end=G.screen[-1]['mean_t']

    if(verbose):
        print(f'\n   t_end = {cavity._t_end} m.')
        print(f'   s_end = {cavity.s_end} m.')
        print(f'   oncrest phase = {cavity._oncrest_phase}')
        print(f'   energy gain =  {p2e(cavity._momentum_end)-p2e(p):0.3f} eV.')

    return G   

