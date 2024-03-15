import numpy as np
from gpt.tools import rotation_matrix
from gpt.tools import cvector

from matplotlib import pyplot as plt



class ScreenZXS():

    def __init__(self, name, ccs_name, s=0, ecs_origin=[0,0,0], ecs_e1=[1, 0, 0], ecs_e2=[0, 1, 0], ccs_out=None, ccs_origin=None, ccs_e1=None, screen_width=0.2, ds=0):

        #print(name, ccs_name, s, ecs_origin, ecs_e1, ecs_e2, ccs_out, ccs_e1, screen_width)

        self._type = 'ScreenZXS'

        self._name = name
        self._ccs_name = ccs_name
        self._s = s

        self._ecs_origin=cvector(ecs_origin)        
        self._ds = ds
        self._e1=cvector(ecs_e1)
        self._e2=cvector(ecs_e2)

        self._ccs_out = ccs_out
        self._ccs_e1 = cvector(ccs_e1)
        self._ccs_origin = cvector(ccs_origin)
        self._screen_width = screen_width


    def plot(self, axis=None, alpha=1.0, ax=None):

        if ax is None:
            ax = plt.gca()

        if(self._ccs_origin is not None):


            #print(self._ccs_e1.T)

            ccs_e1 = self._ccs_e1
            ccs_e1 = ccs_e1/np.linalg.norm(ccs_e1)

            p = self._ccs_origin + self._ds*ccs_e1

            p1 = p + self._screen_width*np.matmul(rotation_matrix(+90), ccs_e1)
            p2 = p + self._screen_width*np.matmul(rotation_matrix(-90), ccs_e1)

            ax.plot([p1[2][0], p2[2][0]], [p1[0][0], p2[0][0]], 'g')

    def gpt_lines(self):

        lines = []
        sline = f'screen("{self._ccs_name}", {self._ecs_origin[0][0]}, {self._ecs_origin[0][0]}, {self._ecs_origin[2][0]}, '

        #if(is_floatable(self._ds)):
        sline = sline+f'{self._e1[0][0]}, {self._e1[1][0]}, {self._e1[2][0]}, {self._e2[0][0]}, {self._e2[1][0]}, {self._e2[2][0]}, 0,'

        if(self._ccs_out is not None):
            sline = sline+f' "{self._ccs_out}");'
        else:
            sline = sline+f' "{self._ccs_name}");'

        return lines+[sline]

    @property
    def name(self):
        return self._name

    @property
    def s(self):
        return self._s
    
    







