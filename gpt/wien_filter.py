from gpt.bstatic import Rectmagnet
from gpt.element import Element
from gpt.estatic import Erect

import numpy as np

from scipy.constants import c
from scipy.constants import physical_constants

MC2 = physical_constants['electron mass energy equivalent in MeV'][0]*1e6

#class WienFilter(Rectmagnet, Erect):

class WienFilter(Element):

    def __init__(self, 
                 name, 
                 B0,
                 length, 
                 selection_energy=None,
                 selection_velocity=None,
                 width=0.2, # a
                 height=0.2,  # b
                 dipole_gap=None,
                 dipole_b1=0,
                 dipole_b2=0,
                 dipole_dl=0,
                 c1='r',
                 c2='b',
                 place=False,
                 x0=0, y0=0, z0=0,
                 yaw=0, pitch=0, roll=0,  
                ):

        Element.__init__(self, name, length=length, width=width, height=height, 
                         x0=x0, y0=y0, z0=z0,
                         yaw=yaw, pitch=pitch, roll=roll, global_element=True)


        if selection_energy is not None:
            
            assert selection_velocity is None

            g0 = selection_energy/MC2
            b0 = np.sqrt(1-1/g0**2)
            self._v0 = c * b0
            
        else:
            
            assert selection_velocity is not None

            self._v0 = selection_velocity

        #self._rectmagnet = Rectmagnet(name+'_B', width, length, B0,
        #                    b1=dipole_b1,
        #                    b2=dipole_b2,
        #                    dl=dipole_dl,
        #                    gap=dipole_gap,
        #                    x0=x0, y0=y0, z0=z0,
        #                    yaw=yaw, pitch=pitch, roll=roll,
        #                    color=c1,
        #                    global_element=True,
        #                    place=place)

        self._B = B0 
        self._E = self._v0 * B0  # Need to replace with field integral

        self._dipole_b1=dipole_b1
        self._dipole_b2=dipole_b2
        self._dipole_dl=dipole_dl

        #self._erect = Erect(name+'_E', length+0.01, height+0.01, width, E0, 
        #                    x0=x0, y0=y0, z0=z0, 
        #                    yaw=yaw, pitch=pitch + np.pi/2, roll=roll,
        #                    color='b',
        #                    global_element=True)

    #def place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg'):

    #    Element.place(self, ref_element=None, ds=0, ref_origin='end', element_origin='beg')

    #    self._rectmagnet.place(ref_element=None, ds=0, ref_origin='end', element_origin='beg')
    #    self._erect.place(ref_element=None, ds=0, ref_origin='end', element_origin='beg')


    def gpt_lines(self):

        N = self.name

        lines = ['\n']

        lines.append(f'{N}_x = {self._ecs["x0"]};\n')
        lines.append(f'{N}_y = {self._ecs["y0"]};\n')

        lines.append(f'{N}_yaw = {self._ecs["yaw"]};\n')
        lines.append(f'{N}_pitch = {self._ecs["pitch"]};\n')
        lines.append(f'{N}_roll = {self._ecs["roll"]};\n')

        ds = np.linalg.norm((self._p_beg - self._ccs_beg_origin)) + self.length/2 + self._ecs["z0"]

        lines.append(f'{N}_a = {self.width};\n')
        lines.append(f'{N}_b = {self.height};\n')
        lines.append(f'{N}_L = {self.length};\n')
        lines.append(f'{N}_z = {ds};\n')
        lines.append(f'{N}_E0 = {self._E};\n')
        lines.append(f'{N}_B0 = {self._B};\n')

        print(self._dipole_dl)
    
        lines.append(f'{N}_dipole_dl = {self._dipole_dl};\n')
        lines.append(f'{N}_dipole_b1 = {self._dipole_b1};\n')
        lines.append(f'{N}_dipole_b2 = {self._dipole_b2};\n')
        lines.append('\n')
        lines.append(f'erect("wcs", "GxyzXYZ", {N}_x, {N}_y, {N}_z, {N}_yaw, {N}_pitch + {np.pi/2}, {N}_roll, {N}_L, {N}_b, {N}_a, {N}_E0 );\n')
        lines.append(f'rectmagnet("wcs", "GxyzXYZ", {N}_x, {N}_y, {N}_z, {N}_yaw, {N}_pitch, {N}_roll, {N}_a, {N}_L, {N}_B0, {N}_dipole_dl, {N}_dipole_b1, {N}_dipole_b2 );\n')

    #    lines = self._rectmagnet.gpt_lines()
    #    lines = lines + self._erect.gpt_lines()

        return lines

    #def plot_floor(self, axis=None, alpha=1.0, ax=None, xlim=None, ylim=None, style=None):

    #    self._rectmagnet.plot_floor(axis=axis, alpha=alpha, ax=ax, xlim=xlim, ylim=ylim, style=style)
        #self._erect.plot_floor(axis=axis, alpha=alpha, ax=ax, xlim=xlim, ylim=ylim, style=style)

    #    return ax

    @property
    def selection_energy(self):
        b0 = self._v0/c
        g0 = 1/np.sqrt(1-b0**2)

        return MC2*g0

    @property
    def selection_momentum(self):
        return np.sqrt(self.selection_energy**2 - MC2**2)
        
        
