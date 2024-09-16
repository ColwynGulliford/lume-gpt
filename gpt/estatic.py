from gpt.element import Element


class Erect(Element):

    def __init__(self, name, a, b, L, E0, 
                 x0=0, y0=0, z0=0, 
                 theta_x=0, theta_y=0, theta_z=0,
                 color='b'):

        Element.__init__(self, name, length=L, widnt=a, height=b, color=color)

        self._E0 = E0


    def gpt_lines(self):

        lines = []
  
        lines = lines + ['\n#***********************************************']
        lines = lines + [ f'#             Erect: {self.name}           ']
        lines = lines + [  '#***********************************************']

        lines = lines + [f'{self.name}_a = {self.width};']
        lines = lines + [f'{self.name}_b = {self.height};']
        lines = lines + [f'{self.name}_L = {self.length};']
        lines = lines + [f'{self.name}_E = {self._E};']

        ds = np.linalg.norm((self._p_beg - self._ccs_beg_origin)) + self.length/2

        bend_line = f'\nerect("{self.ccs_beg}", 0, 0, {ds}, 1, 0, 0, 0, 1, 0, '
        bend_line = bend_line + f'{self.name}_a, {self.name}_b, {self.name}_E,'
        bend_line = bend_line + f'{self.name}_fringe_dl, {self.name}_fringe_b1, {self.name}_fringe_b2);'

        lines.append(bend_line)