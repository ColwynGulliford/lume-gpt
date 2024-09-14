from matplotlib import pyplot as plt
import numpy as np

from gpt.element import Element

class Aperture(Element):

    def __init__(self, name, a=None, b=None, aperture_file=None, xmax=None, ymax=None, color='k'):
        
        self._aperture_file = aperture_file

        if(a is None and b is None):
            assert aperture_file is not None, "If no radii specified, user must give a aperture bitmap file."

        elif(aperture_file is None):
            assert (a is not None) or (b is not None), "If no aperture file, user must give the aperture radii."
            if(b is None):
                b = a
            elif(a is None):
                a = b
                
            self._a = a
            self._b = b

            if(xmax is None):
                xmax = 2*a
            if(ymax is None):
                ymax = 2*b
                
            self._xmax = xmax
            self._ymax = ymax

        super().__init__(name, length=0, width=xmax, height=ymax, color=color)

    def gpt_lines(self):

        lines = []
    
        lines.append(f'{self.name}_x = 0;')
        lines.append(f'{self.name}_y = 0;')
    
        s = 0.5*(self._s_beg + self._s_end)
        ds = np.linalg.norm( 0.5*(self.p_end + self.p_beg) - self._ccs_beg_origin) 

        if(self._a):
            lines.append(f'{self.name}_z = {ds};')
            lines.append(f'{self.name}_a = {self._a};')
            lines.append(f'{self.name}_b = {self._b};')
            lines.append(f'aperture("{self._ccs_beg}", 0, 0, {self.name}_z, 1, 0, 0, 0, 1, 0, {self.name}_a, {self.name}_b);')
                
        elif(self._aperture_file):
            raise ValueError('Aperture file untested')
    
        return lines

    
    def plot_floor(self, axis='equal', ax=None, alpha=1, xlim=None, ylim=None, style='tao'):

        if ax is None:
            ax = plt.gca()

        pc = self.p_beg

        ppr = pc + self.e1_beg * self._a
        ppmax = pc + self.e1_beg * 2*self._xmax

        pmr = pc - self.e1_beg * self._a
        pmmax = pc - self.e1_beg * 2*self._xmax

        pps = np.concatenate( (ppr, ppmax), axis=1)
        pms = np.concatenate( (pmr, pmmax), axis=1)

        ax.plot(pps[2], pps[0], self.color, alpha=alpha )
        ax.plot(pms[2], pms[0], self.color, alpha=alpha )
        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')

        if(axis=='equal'):
            ax.set_aspect('equal')

        return ax

    def plot_field_profile(self, ax=None, normalize=False):
        if ax is None:
            ax = plt.gca()
        return ax    

        

class CircularAperture(Element):

    def __init__(self, name, R, color='k', Rmax = None):

        if(Rmax is None):
            Rmax = 2*R

        super().__init__(name, length=0, width=Rmax, height=Rmax, color=color)

        self._type='Circular Aperture'
        self._R = R
        self._Rmax = Rmax

    def gpt_lines(self):

        lines = []

        lines.append(f'{self.name}_x = 0;')
        lines.append(f'{self.name}_y = 0;')

        s = 0.5*(self._s_beg + self._s_end)
        ds = np.linalg.norm( 0.5*(self.p_end + self.p_beg) - self._ccs_beg_origin) 

        lines.append(f'{self.name}_z = {ds};')
        lines.append(f'{self.name}_radius = {self._R};')
        lines.append(f'aperture("{self._ccs_beg}", 0, 0, {self.name}_z, 1, 0, 0, 0, 1, 0, {self.name}_radius);')

        return lines

    def plot_floor(self, axis='equal', ax=None, alpha=1, xlim=None, ylim=None, style='tao'):

        if ax is None:
            ax = plt.gca()

        pc = self.p_beg

        ppr = pc + self.e1_beg * self._R
        ppmax = pc + self.e1_beg * 2*self._Rmax

        pmr = pc - self.e1_beg * self._R
        pmmax = pc - self.e1_beg * 2*self._Rmax

        pps = np.concatenate( (ppr, ppmax), axis=1)
        pms = np.concatenate( (pmr, pmmax), axis=1)

        ax.plot(pps[2], pps[0], self.color, alpha=alpha )
        ax.plot(pms[2], pms[0], self.color, alpha=alpha )
        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')

        if(axis=='equal'):
            ax.set_aspect('equal')

        return ax

    def plot_field_profile(self, ax=None, normalize=False):
        if ax is None:
            ax = plt.gca()
        return ax
        