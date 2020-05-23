import numpy as np
import os
import math, cmath
from scipy.integrate import cumtrapz
from gpt import tools

c = 299792458
mc2 = 0.51e6

def gamma_to_beta(gamma):
    return np.sqrt(1 - 1/gamm**2)

def beta_to_gamma(beta):
    return 1/np.sqrt(1-beta**2)

def KE_to_gamma(KE):
    return 1 + KE/mc2

def gamma_to_KE(gamma):
    return mc2*(gamma-1)

def KE_to_beta(KE):
    return gamma_to_beta(KE_to_gamma(KE))

def beta_to_KE(beta):
    return gamma_to_KE(beta_to_gamma(beta))

def get_gdf_header(gdf_file, gdf2a_bin='$GDF2A_BIN'):

    temp_ascii_file = f'{gdf_file}.temp.txt'
    rc = os.system(f'{gdf2a_bin} -o {temp_ascii_file} {gdf_file}')

    with open(temp_ascii_file, 'r') as fp:
        columns = fp.readline().split()
    os.system(f'rm {temp_ascii_file}')
    return columns

def scale_field_map(in_gdf_file, out_gdf_file):
    pass

def integrate_Ez(z, Ez, w, phi, beta=1):

        phi = phi*math.pi/180

        if(beta==1):
            phase = (w/c)*z + phi
            return np.trapz(Ez*np.exp(np.complex(0,1)*phase), z)

def energy_gain(z, Ez, w, phi, beta):

        if(isinstance(phi, float)):
            return np.real(integrate_Ez(z, Ez, w, phi, beta))
        else:
            return np.array( list(map(lambda p: np.real(integrate_Ez(z, Ez, w, p, beta=beta)), phi)) )

class GDFFieldMap():

    def __init__(self, source_data, column_names = None, gdf2a_bin='$GDF2A_BIN'):
        
        assert os.path.exists(tools.full_path(gdf2a_bin)), f'GDF2A binary does not exist: {gdf2a_bin}'  

        self.source_data_file = source_data
        temp_ascii_file = f'{self.source_data_file}.temp.txt'
        os.system(f'{gdf2a_bin} -o {temp_ascii_file} {self.source_data_file}')

        with open(temp_ascii_file, 'r') as fp:
            columns = fp.readline().split()
        
        if(column_names is None):
            column_names = {c:c for c in columns}

        self.column_names = column_names
        ndata = np.loadtxt(temp_ascii_file, skiprows=1)

        os.system(f'rm {temp_ascii_file}')

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

    def __getitem__(self, key):

        if(key in self.coordinates):
            return np.unique(getattr(self, self.column_names[key]))
        elif(key in self.field_components):
            return np.transpose(np.reshape(getattr(self, self.column_names[key]), self.data_shape, 'F'), self.ordering)

    def scale_coordinates(self, scale):
        for coordinate in self.coordinates:
            setattr(self, coordinate, scale*getattr(self, coordinate))

    def scale_fields(self, scale):
        for component in self.field_components:
            setattr(self, component, scale*getattr(self, component))

    def write_gdf(self, new_gdf_file, asci2gdf_bin='$ASCI2GDF_BIN'):

        temp_ascii_file = new_gdf_file + '.txt'

        headers = ' '.join(self.column_names)
        data = np.zeros( (len(getattr(self, self.field_components[0])), len(self.column_names)) )
        np.savetxt(temp_ascii_file, data, header=headers, comments=' ')

        os.system(f'{asci2gdf_bin} -o {new_gdf_file} {temp_ascii_file}')
        os.system(f'rm {temp_ascii_file}')


class Map1D(GDFFieldMap):

    def __init__(self, source_data, required_columns, gdf2a_bin='$GDF2A_BIN', column_names=None, zpos=0):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names)

        self.zpos = zpos

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


class Map1D_E(Map1D):

    def __init__(self, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'Ez':'Ez'}, zpos=0):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Ez'], zpos=zpos)

    @property
    def energy_gain(self, r=0, field_order =1):

        if(r==0):  # integrate on-axis -> trapz field map
            return np.trapz(getattr(self, 'Ez'), self.zpos+getattr(self, 'z'))
        else:
            print('wtf?')


class Map1D_B(Map1D):

    def __init__(self, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'Bz':'Bz'}, zpos=0):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Bz'], zpos=zpos)


class Map1D_TM(Map1D):

    def __init__(self, source_data, frequency, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'Ez':'Ez'}, kinetic_energy=float('Inf')):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Ez'])

        self._frequency = frequency
        self._w = 2*math.pi*frequency
        self._kinetic_energy = kinetic_energy

        if(self._kinetic_energy==float('Inf')):
            self._beta = 1
        else:
            gamma = 1+self._kinetic_energy/mc2
            self._beta=np.sqrt(1 - 1/gamma**2)

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


class Map2D(GDFFieldMap):

    def __init__(self, source_data, required_columns, gdf2a_bin='$GDF2A_BIN', column_names=None, zpos=0):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names)
    
        self.required_columns = required_columns
        assert 'z' in required_columns
        assert 'r' in required_columns
        assert len(column_names)>=len(self.required_columns)

        for rc in required_columns:
            assert rc in column_names.values(),f'User must specify a key name for required column {rc}'
    

class Map2D_E(Map2D):
    
    def __init__(self, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'r':'r', 'Ez':'Ez', 'Er':'Er'}):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Ez', 'r', 'Er'])

    def z(self):
        return np.squeeze(self.z[self.r==0])

    @property
    def on_axis_Ez(self):
        return np.squeeze(self.Ez[self.r==0])

    @property
    def on_axis_integral(self):
        return np.trapz(self.on_axis_Ez, self.z[self.r==0])


class Map2D_B(Map2D):

    def __init__(self, source_data, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'r':'r', 'Bz':'Bz', 'Br':'Br'}):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Bz', 'r', 'Br'])

    @property
    def on_axis_Bz(self):
        return np.squeeze(self.Bz[self.r==0])

    @property
    def on_axis_integral(self):
        return np.trapz(self.on_axis_Bz, self.z[self.r==0])
    

class Map25D_TM(Map2D):

    def __init__(self, source_data, frequency, gdf2a_bin='$GDF2A_BIN', column_names={'z':'z', 'r':'r', 'Ez':'Ez', 'Er':'Er', 'Bphi':'Bphi'}, kinetic_energy=float('Inf')):

        super().__init__(source_data, gdf2a_bin=gdf2a_bin, column_names=column_names, required_columns=['z', 'Ez', 'r', 'Er', 'Bphi'])

        self._frequency = frequency
        self._w = 2*math.pi*frequency
        self._kinetic_energy = kinetic_energy

        if(self._kinetic_energy==float('Inf')):
            self._beta = 1
        else:
            gamma = 1+self._kinetic_energy/mc2
            self._beta=np.sqrt(1 - 1/gamma**2)

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



