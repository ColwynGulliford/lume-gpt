from matplotlib import pyplot as plt
import numpy as np

from gpt.element import Element

class CircularAperture(Element):

    def __init__(self, name, R, color='k', Rmax = None):

        if(Rmax is None):
            Rmax = 2*R

        super().__init__(name, length=0, width=Rmax, height=Rmax, angles=[0, 0, 0], color=color)

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
        