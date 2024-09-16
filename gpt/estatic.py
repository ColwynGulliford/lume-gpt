from gpt.element import Element

import numpy as np


class Erect(Element):

    def __init__(self, name, a, b, L, E0, 
                 x0=0, y0=0, z0=0, 
                 theta_x=0, theta_y=0, theta_z=0,
                 color='b',
                 global_element=False
                ):

        Element.__init__(self, name, length=L, width=a, height=b, color=color, global_element=global_element)
        
        self._E0 = E0
        self._type='erect'

        print(self._E0, self._type)


    def gpt_lines(self):

        lines = []
  
        lines = lines + ['\n#***********************************************']
        lines = lines + [ f'#             Erect: {self.name}           ']
        lines = lines + [  '#***********************************************']

        lines = lines + [f'{self.name}_a = {self.width};']
        lines = lines + [f'{self.name}_b = {self.height};']
        lines = lines + [f'{self.name}_L = {self.length};']
        lines = lines + [f'{self.name}_E = {self._E0};']

        ds = np.linalg.norm((self._p_beg - self._ccs_beg_origin)) + self.length/2

        e_line = f'\nerect("{self.ccs_beg}", {self.ecs_str} {self.name}_a, {self.name}_b, {self.name}_L, {self.name}_E);'

        lines.append(e_line)

        return lines