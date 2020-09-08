import numpy as np
import os
import math, cmath
from scipy.integrate import cumtrapz
import tempfile

from gpt import GPT
from gpt import tools
from gpt.tools import cvector
from gpt.element import Element
from gpt.template import basic_template
from gpt.template import ztrack1_template

from matplotlib import pyplot as plt

from pmd_beamphysics import single_particle

from scipy.optimize import brent

c = 299792458
mc2 = 0.51e6

def gamma_to_beta(gamma):
    """ Converts relativistic gamma to beta"""
    return np.sqrt(1 - 1/gamm**2)

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

def get_gdf_header(gdf_file, gdf2a_bin=os.path.expandvars('$GDF2A_BIN')):

    """Reads the header (column names) of gdf_file and returns them"""

    assert os.path.exists(gdf2a_bin), f'The gdf2a binary "{gdf2a_bin}" does not exist.'
    assert os.path.exists(gdf_file), f'The gdf file "{gdf_file}" does not exist'

    temp_ascii_file = f'{gdf_file}.temp.txt'
    rc = os.system(f'{gdf2a_bin} -o {temp_ascii_file} {gdf_file}')

    with open(temp_ascii_file, 'r') as fp:
        columns = fp.readline().split()
    os.system(f'rm {temp_ascii_file}')
    return columns

def scale_field_map(in_gdf_file, out_gdf_file):
    pass

def integrate_Ez(z, Ez, w, phi, beta=1):

        """Integrates a complex field map on axis:
        z  = vector of z points of on axis field map 
        Ez = vector of Ez(z)
        w  = cavity angular frequency [rad*Hz]
        phi = phase offset
        beta = relativistic beta, defaults to 1

        Computes: Integral ( Ez(z) * exp(iwz /cbeta + phi) dz)
        """

        phi = phi*math.pi/180

        if(beta==1):
            phase = (w/c)*z + phi
            return np.trapz(Ez*np.exp(np.complex(0,1)*phase), z)

def energy_gain(z, Ez, w, phi, beta=1):

        if(isinstance(phi, float)):
            return np.real(integrate_Ez(z, Ez, w, phi, beta))
        else:
            return np.array( list(map(lambda p: np.real(integrate_Ez(z, Ez, w, p, beta=beta)), phi)) )

class GDFFieldMap(Element):
    """ General class for holding GDF field map data """

    def __init__(self, source_data, column_names = None, gdf2a_bin='$GDF2A_BIN', use_temp_file=True):
        
        assert os.path.exists(tools.full_path(gdf2a_bin)), f'GDF2A binary does not exist: {gdf2a_bin}'  
        self.source_data_file = source_data

        if(use_temp_file):
            temp_ascii_file = tempfile.NamedTemporaryFile().name
        else:
            temp_ascii_file = f'{self.source_data_file}.temp.txt'
        
        os.system(f'{gdf2a_bin} -o {temp_ascii_file} {self.source_data_file}')

        with open(temp_ascii_file, 'r') as fp:
            columns = fp.readline().split()
        
        if(column_names is None):
            column_names = {c:c for c in columns}

        self.column_names = column_names
        ndata = np.loadtxt(temp_ascii_file, skiprows=1)

        os.remove(temp_ascii_file)

        self.coordinates = [name for name in columns if(name.lower() in ['r', 'x', 'y', 'z'])]
        self.field_components = [name for name in columns if(name.lower() not in ['r', 'x', 'y', 'z'])]

        assert len(columns)==len(self.coordinates) + len(self.field_components)
        for name in column_names:
            assert name in self.coordinates or name in self.field_components, name

        coordinate_sizes = {}
        coordinate_count_step = {}
            
        # Get the coordinate vectors:
        for var in self.coordinates:
            name = column_names[var]
 
            v0 = ndata[0, columns.index(var)]
            for ii,v in enumerate(ndata[:, columns.index(var)]):
                if(v!=v0):
                    coordinate_count_step[var] = ii
                    break

            value = ndata[:, columns.index(var)] 
            setattr(self, name, value ) 
            #self.__setitem__(name, value)

            coordinate_sizes[var] = len(np.unique(ndata[:, columns.index(var)]))

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
            setattr(self, column_names[component], ndata[:,columns.index(component)])

    def invert_column_names(self, var):
        pass

    def __getitem__(self, key):

        if(key in self.coordinates):
            return np.unique(getattr(self, self.column_names[key]))
        elif(key in self.field_components):
            return np.transpose(np.reshape(getattr(self, self.column_names[key]), self.data_shape, 'F'), self.ordering)
        else:

            for k, v in self.column_names.items():
                
                if(key==v):
                    
                    if(k in self.coordinates):
                        return np.unique(getattr(self, self.column_names[k]))
                    elif(k in self.field_components):
                        return np.transpose(np.reshape(getattr(self, self.column_names[k]), self.data_shape, 'F'), self.ordering)

            print(f'Field map does not contain item "{key}"')

    def scale_coordinates(self, scale):
        for coordinate in self.coordinates:
            setattr(self, coordinate, scale*getattr(self, coordinate))

    def scale_fields(self, scale):
        for component in self.field_components:
            setattr(self, component, scale*getattr(self, component))

    def write_gdf(self, new_gdf_file, asci2gdf_bin='$ASCI2GDF_BIN', verbose=True):

        temp_ascii_file = new_gdf_file + '.txt'

        data = np.zeros( (len(getattr(self, self.field_components[0])), len(self.column_names)) )

        headers = []
        for ii, var in enumerate(self.coordinates+self.field_components):
            headers.append(var)
            data[:,ii] = getattr(self, self.column_names[var])

        headers = '     '.join(headers)
        np.savetxt(temp_ascii_file, data, header=headers, comments=' ')

        os.system(f'{asci2gdf_bin} -o {new_gdf_file} {temp_ascii_file}')
        os.system(f'rm {temp_ascii_file}')

    def gpt_lines(self, ccs=None, gdf_file=None, e1=[1, 0, 0], e2=[0, 1, 0], scale =None, user_vars=[]):

        element = self._name

        inverse_column_names = {value:key for key,value in self.column_names.items()}

        if(ccs is None):
            ccs = self.ccs_beg

        ds = np.linalg.norm((self._p_beg - self._ccs_beg_origin))
        ccs_beg_e3 = cvector([0,0,1])
        r = ds*ccs_beg_e3

        map_line = f'{self.type}("{self.ccs_beg}", '
        extra_lines={}

        if(self._field_pos=='center'):
            zoff = self._length/2.0
        elif(self._field_pos=='end'):
            zoff = self._length
        else:
            zoff = 0

        for ii, coordinate in enumerate(['x', 'y','z']):
            #if(coordinate in user_vars):
            if(coordinate=='z'):
                val = r[ii][0]+zoff
            else:
                val = r[ii][0]
            extra_lines[coordinate] = f'{element}_{coordinate} = {val};'
            map_line = map_line + f'{element}_{coordinate}, '
            #else:
            #    map_line = map_line + f'{str(r[ii][0])}, '

        for ii, m in enumerate(['e11', 'e12', 'e13']):
            if(m in user_vars):
                extra_lines[m] = f'{element}_{m} = {e1[ii]};'
                map_line = map_line + f'{element}_{m}, '
            else:
                map_line = map_line + f'{str(e1[ii])}, '

        for ii, m in enumerate(['e21', 'e22', 'e23']):
            if(m in user_vars):
                extra_lines[m] = f'{element}_{m} = {e2[ii]};'
                map_line = map_line + f'{element}_{m}, '
            else:
                map_line = map_line + f'{str(e2[ii])}, '

        if(gdf_file is None):
            gdf_file = self.source_data_file

        map_line = map_line + f'"{gdf_file}", '

        for ii, rc in enumerate(self.required_columns):
            map_line = map_line + f'"{inverse_column_names[rc]}", '
        
        if(scale is None):
            scale = self._scale

        #if('scale' in user_vars):
        extra_lines['scale'] = f'{element}_scale = {scale};'
        map_line = map_line + f'{element}_scale);'
        #else:
        #    map_line = map_line + f'{scale});'

        return [extra_line for extra_line in extra_lines.values()] + [map_line]


class Map1D(GDFFieldMap):

    """ Class for storing 1D GDF field maps, derives from GDFfieldMap """

    def __init__(self, source_data, required_columns, gdf2a_bin='$GDF2A_BIN', column_names=None, zpos=0):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names)

        self.zpos = zpos
        self._type = 'Map1D'

        self.required_columns = required_columns
        assert 'z' in required_columns
        assert len(column_names)==len(self.required_columns)
        for k,v in column_names.items():
            assert v in self.required_columns, f'User defined column names must have a key name for {v}'
            if(v!='z'):
                self.fieldstr = v
    @property
    def on_axis_integral(self):
        return np.trapz(getattr(self, self.fieldstr), self.zpos + getattr(self, 'z'))

    def cum_on_axis_integral(self):
        return cumptrapz(getattr(self, self.fieldstr), self.zpos + getattr(self, 'z'), initial=0)

    def plot_floor(self, axis=None, alpha=1.0, ax=None):

        f = 0.01
        zs = getattr(self, 'z')
        Fz = getattr(self, self.fieldstr)

        maxF = max(np.abs(Fz))

        for ii,z in enumerate(zs):
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

        if(ax == None):
            ax = plt.gca()

        pc = 0.5*(self.p_beg + self.p_end)

        p1 = self.p_beg + (self._width/2)*self.e1_beg
        p2 = self.p_beg - (self._width/2)*self.e1_beg
        p3 = self.p_end + (self._width/2)*self.e1_beg
        p4 = self.p_end - (self._width/2)*self.e1_beg

        ps1 = np.concatenate( (p1, p3, p4, p2, p1), axis=1)

        p1 = pc + (self._width/2)*self.e1_beg - (effective_plot_length/2.0)*self.e3_beg
        p2 = p1 - (self._width)*self.e1_beg
        p3 = p2 + (effective_plot_length)*self.e3_beg
        p4 = p3 + (self._width)*self.e1_beg

        ps2 = np.concatenate( (p1, p2, p3, p4, p1), axis=1)

        ax.plot(ps1[2], ps1[0], self.color, alpha=0.2)
        ax.plot(ps2[2], ps2[0], self.color, alpha=alpha)
        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')

        if(axis=='equal'):
            ax.set_aspect('equal')

    #@property
    #def z(self):
    #    return self._z
    

class Map1D_E(Map1D):

    def __init__(self, name, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'Ez':'Ez'}, zpos=0, width=0.2):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Ez'], zpos=zpos)
       
        self._name = name
        self._type = 'Map1D_E'
        self._length = self.z[-1]-self.z[0]
        self._width = 0.2
        self._height = self._width
        self._color = '#1f77b4'

    @property
    def energy_gain(self, r=0, field_order =1):

        if(r==0):  # integrate on-axis -> trapz field map
            return np.trapz(getattr(self, 'Ez'), self.zpos+getattr(self, 'z'))
        else:
            print('wtf?')

class Map1D_B(Map1D):

    def __init__(self, name, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'Bz':'Bz'}, zpos=0, width=0.2, field_pos='center', scale=1):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Bz'], zpos=zpos)
        
        self._name = name
        self._type = 'Map1D_B'
        self._length = self.z[-1]-self.z[0]
        self._width = width
        self._height = self._width
        self._color = '#2ca02c'
        self._field_pos = 'center'
        self._scale=scale

        self.place()

    def plot_field_profile(self, ax=None, normalize=False):

        if(ax == None):
            ax = plt.gca()

        if(self._field_pos=='center'):
            zoff = self._length/2.0
        elif(self._field_pos=='end'):
            zoff = self._length
        else:
            zoff = 0

        zs = self.s_beg + zoff + self['z']

        Bz = self['Bz']

        if(normalize):
            Bz = Bz/np.max(np.abs(Bz))
        else:
            Bz = self._scale*Bz

        ax.plot(zs, Bz, self._color)
        ax.set_xlabel('s (m)')

        return ax

        


class Map1D_TM(Map1D):

    def __init__(self, 
        name, 
        source_data, 
        frequency, 
        scale=1,
        relative_phase=0,
        oncrest_phase=0,
        gdf2a_bin='$GDF2A_BIN', 
        column_names={'z':'z', 'Ez':'Ez'}, 
        kinetic_energy=float('Inf'), 
        color='darkorange',
        field_pos='center',
        legacy_phasing_lines=False):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Ez'])

        self._name = name
        self._type='Map1D_TM'
        self._length = self.z[-1]-self.z[0]
        self._width = 0.2
        self._height = self._width
        self._color = color

        self._frequency = frequency
        self._w = 2*math.pi*frequency
        self._kinetic_energy = kinetic_energy

        if(self._kinetic_energy==float('Inf')):
            self._beta = 1
        else:
            gamma = 1+self._kinetic_energy/mc2
            self._beta=np.sqrt(1 - 1/gamma**2)

        self._field_pos=field_pos
        self._oncrest_phase=oncrest_phase
        self._scale=1
        self._relative_phase=relative_phase
        self._scale=scale

        self.place()

    def integrate_Ez(self, phi):
        return integrate_Ez(self.z, self.Ez, self._w, phi, self._beta)

    def energy_gain(self, phi):
        return energy_gain(self.z, self.Ez, self._w, phi, self._beta)

    @property
    def oncrest_phase(self):
        return (-cmath.phase(self.integrate_Ez(0))*180/math.pi)%360

    @property
    def oncrest_energy_gain(self):
        return self.energy_gain(self.oncrest_phase)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self._kinetic_energy = beta_to_KE(beta)

    @property
    def kinetic_energy(self):
        return self._kinetic_energy

    @kinetic_energy.setter
    def kinetic_energy(self, kinetic_energy):
        self._kinetic_energy = kinetic_energy
        self._beta = KE_to_beta(kinetic_energy)

    def plot_field_profile(self, ax=None, normalize=False):

        if(ax == None):
            ax = plt.gca()

        if(self._field_pos=='center'):
            zoff = self._length/2.0
        elif(self._field_pos=='end'):
            zoff = self._length
        else:
            zoff = 0

        zs = self.s_beg + zoff + self['z']

        Ez = self['Ez']

        if(normalize):
            Ez = Ez/np.max(np.abs(Ez))

        ax.plot(zs, Ez, self._color)
        ax.set_xlabel('s (m)')

        return ax


    def track_on_axis(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, n_screen=1, workdir=None):
        return track_on_axis(self, t, p, xacc=xacc,  GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, n_screen=n_screen, workdir=workdir)

    def autophase_track1(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, oncrest_phase=0, workdir=None, n_screen=1):
        return autophase_track1(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, oncrest_phase=oncrest_phase, workdir=workdir, n_screen=n_screen)

    def autophase(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, workdir=None, n_screen=100):
        return autophase(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, workdir=workdir, n_screen=n_screen)

    def gpt_lines(self):

        name = self.name

        base_lines = super().gpt_lines()
        extra_lines = base_lines[:-1]
        map_line = base_lines[-1].replace(');','')

        extra_lines.append(f'{name}_oncrest_phase = {self._oncrest_phase};')
        extra_lines.append(f'{name}_relative_phase = {self._relative_phase};')
        extra_lines.append(f'{name}_phase = ({name}_oncrest_phase + {name}_relative_phase)*pi/180;')

        map_line = map_line + f', {name}_phase, '

        extra_lines.append(f'{name}_frequency = {self._frequency};');
        map_line = map_line + f'2*pi*{name}_frequency);'

        return extra_lines + [map_line]


class Map2D(GDFFieldMap):

    def __init__(self, source_data, required_columns, gdf2a_bin='$GDF2A_BIN', column_names=None, zpos=0):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names)
    
        self.required_columns = required_columns
        assert 'z' in required_columns
        assert 'r' in required_columns
        assert len(column_names)>=len(self.required_columns)

        for rc in required_columns:
            assert rc in column_names.values(),f'User must specify a key name for required column {rc}'

    def plot_floor(self, axis=None, alpha=1.0, ax=None):

        f = 0.01
        zs = getattr(self, 'z')
        Fz = getattr(self, self.fieldstr)

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

        if(ax == None):
            ax = plt.gca()

        pc = 0.5*(self.p_beg + self.p_end)

        max_radius = np.max(self['R'])

        if(max_radius is None):
            max_radius = np.max(self['r'])

        pc = 0.5*(self.p_beg + self.p_end)

        p1 = self.p_beg + (self._width/2)*self.e1_beg
        p2 = self.p_beg - (self._width/2)*self.e1_beg
        p3 = self.p_end + (self._width/2)*self.e1_beg
        p4 = self.p_end - (self._width/2)*self.e1_beg

        ps1 = np.concatenate( (p1, p3, p4, p2, p1), axis=1)

        max_radius = np.max(self['R'])

        if(self._field_pos=='center'):
            p0=pc
        elif(self._field_pos=='end'):
            p0=self.p_beg + cvector([0,0,effective_plot_length/2.0])

        p1 = p0 + (self._width/2)*self.e1_beg - (effective_plot_length/2.0)*self.e3_beg
        p2 = p1 - (self._width)*self.e1_beg
        p3 = p2 + (effective_plot_length)*self.e3_beg
        p4 = p3 + (self._width)*self.e1_beg

        ps2 = np.concatenate( (p1, p2, p3, p4, p1), axis=1)

        ax.plot(ps1[2], ps1[0], self.color, alpha=0.2)
        ax.plot(ps2[2], ps2[0], self.color, alpha=alpha)
        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')

        if(axis=='equal'):
            ax.set_aspect('equal')

        return ax

    def plot_field_profile(self, ax=None, normalize=False):

        if(ax == None):
            ax = plt.gca()

        if(self._field_pos=='center'):
            zoff = self._length/2.0
        elif(self._field_pos=='end'):
            zoff = self._length
        else:
            zoff = 0

        zs = self.z
        Fz = self.on_axis_Ez

        zs = self.s_beg + zoff + self['z']

        if(normalize):
            Fz = np.abs(Fz/np.max(np.abs(Fz)))

        ax.plot(zs, Fz, self._color)
        ax.set_xlabel('s (m)')

        if(not normalize):
            ax.set_xlabel('z (m)')
            ax.set_ylabel('$E_z$ (V/m)')
    
        return ax

    def track_on_axis(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, n_screen=1, workdir=None):
        return track_on_axis(self, t, p, xacc=xacc,  GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, n_screen=n_screen, workdir=workdir)

class Map2D_E(Map2D):
    
    def __init__(self, name, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'r':'r', 'Ez':'Ez', 'Er':'Er'}, field_pos='center', scale=1):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['r', 'z', 'Er', 'Ez'])
        self._name = name
        self._type = 'Map2D_E'
        self._length = self.z[-1]-self.z[0]
        self._width = 0.2
        self._height = self._width
        self._color = '#1f77b4'
        self._field_pos = field_pos

        self.fieldstr='Ez'
        self._scale=scale

        self.place()

    def z(self):
        return np.squeeze(self.z[self.r==0])

    @property
    def on_axis_Ez(self):
        return np.squeeze(self.Ez[self.r==0])

    @property
    def on_axis_integral(self):
        return np.trapz(self.on_axis_Ez, self.z[self.r==0])

    def plot_field_profile(self, ax=None, normalize=False):

        if(ax == None):
            ax = plt.gca()

        if(self._field_pos=='center'):
            zoff = self._length/2.0
        elif(self._field_pos=='end'):
            zoff = self._length
        else:
            zoff = 0

        zs = self.z
        Fz = self.on_axis_Ez

        zs = self.s_beg + zoff + self['z']

        if(normalize):
            Fz = np.abs(Fz/np.max(np.abs(Fz)))

        ax.plot(zs, Fz, self._color)
        ax.set_xlabel('s (m)')

        return ax

class Map2D_B(Map2D):

    def __init__(self, name, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'r':'r', 'Bz':'Bz', 'Br':'Br'}, field_pos='center'):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['r', 'z', 'Br', 'Bz'])

        self._type='Map2D_B'
        self._name = name
        self._length = self.z[-1]-self.z[0]
        self._width = 0.2
        self._height = self._width
        self._color = '#2ca02c'
        self._field_pos = field_pos

        self.place()

    @property
    def on_axis_Bz(self):
        return np.squeeze(self.Bz[self.r==0])

    @property
    def on_axis_integral(self):
        return np.trapz(self.on_axis_Bz, self.z[self.r==0])

class Map25D_TM(Map2D):

    def __init__(self, 
        name,
        source_data, 
        frequency, 
        scale=1,
        relative_phase=0,
        gdf2a_bin='$GDF2A_BIN', 
        column_names={'z':'z', 'r':'r', 'Ez':'Ez', 'Er':'Er', 'Bphi':'Bphi'}, 
        kinetic_energy=float('Inf'),
        field_pos='center',
        k=0,
        color='darkorange',
        legacy_phasing_lines=False):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['r', 'z', 'Er', 'Ez', 'Bphi'])

        self._name=name
        self._type='Map25D_TM'

        self._scale = scale
        self._relative_phase = relative_phase

        self._frequency = frequency
        self._w = 2*math.pi*frequency
        self._kinetic_energy = kinetic_energy

        if(self._kinetic_energy==float('Inf')):
            self._beta = 1
        else:
            gamma = 1+self._kinetic_energy/mc2
            self._beta=np.sqrt(1 - 1/gamma**2)

        self._field_pos=field_pos
        self._oncrest_phase=0

        self._length = self.z[-1]-self.z[0]
        self._width = 0.2
        self._height = self._width
        self._color = color

        self._k=k

        self.fieldstr='Ez'
        self.rstr = 'r'

        self.place()

    @property
    def on_axis_Ez(self):
        return np.squeeze(self.Ez[self.r==0])

    def integrate_Ez(self, phi):
        return integrate_Ez(self['z'], self.on_axis_Ez, self._w, phi, self._beta)

    def energy_gain(self, phi):
        return energy_gain(self['z'], self.on_axis_Ez, self._w, phi, self._beta)

    @property
    def oncrest_phase(self):
        return (-cmath.phase(self.integrate_Ez(0))*180/math.pi)%360

    @property
    def oncrest_energy_gain(self):
        return self.energy_gain(self.oncrest_phase)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self._kinetic_energy = beta_to_KE(beta)

    @property
    def kinetic_energy(self):
        return self._kinetic_energy

    @kinetic_energy.setter
    def kinetic_energy(self, kinetic_energy):
        self._kinetic_energy = kinetic_energy
        self._beta = KE_to_beta(kinetic_energy)

    @property
    def scale(self):
        return self._scale

    def gpt_lines(self, oncrest_phase=None, relative_phase=None):

        name = self.name

        base_lines = super().gpt_lines()
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

        map_line = map_line + f', {name}_k, {name}_phase, '
        extra_lines.append(f'{name}_frequency = {self._frequency};')
        map_line = map_line + f'2*pi*{name}_frequency);'

        lines = extra_lines + [map_line]

        return lines

    def autophase_track1(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, oncrest_phase=0, workdir=None, n_screen=1):
        return autophase_track1(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, oncrest_phase=oncrest_phase, workdir=workdir, n_screen=n_screen)

    def autophase(self, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, workdir=None, n_screen=100):
        return autophase(self, t, p, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, workdir=workdir, n_screen=n_screen)




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
    n_screen=n_screen, 
    s_beg=element.s_beg)

    return G

def autophase_track1(cavity, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, oncrest_phase=0, workdir=None, n_screen=1):

    cavity._oncrest_phase = oncrest_phase

    G = cavity.track_on_axis(t, p,  
        xacc=xacc, 
        GBacc=GBacc, 
        dtmin=dtmin, 
        dtmax=dtmax,
        workdir=workdir,
        n_screen=n_screen)

    if(G.n_screen>=1):
        return -G.screen[-1]['mean_energy']
    else:
        return 8e88

def autophase(cavity, t, p, xacc=6.5, GBacc=12, dtmin=1e-15, dtmax=1e-8, workdir=None, n_screen=100):

    """ Auto phases a cavity for a particle entering the fieldmap at time = t with total momentum = p """

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
        n_screen=1), )

    cavity._oncrest_phase = oncrest_phase

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

    return G   

