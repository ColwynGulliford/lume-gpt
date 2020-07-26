import numpy as np
import os
import math, cmath
from scipy.integrate import cumtrapz
from gpt.tools import is_floatable

from matplotlib import pyplot as plt

from numpy.linalg import norm 

c = 299792458  # Speed of light 

def line_segment_intersect_2d(p1, p2, q1, q2, tol=1e-12):

    X1 = p1[0]
    Y1 = p1[2]

    X2 = p2[0]
    Y2 = p2[2]

    X3 = q1[0]
    Y3 = q1[2]

    X4 = q2[0]
    Y4 = q2[2]

    I1 = [min(X1,X2), max(X1,X2)]
    I2 = [min(X3,X4), max(X3,X4)]

    Ia = [max( min(X1,X2), min(X3,X4) ),
      min( max(X1,X2), max(X3,X4) )]

    if (max(X1,X2) < min(X3,X4)):
        return (False, None)  # There is no mutual abcisses

    if(X1==X2):
        print('fart')
        return (False, None)
    if(X3==X4):
        print('werd')
        return (False, None)

    A1 = (Y1-Y2)/(X1-X2)  # Pay attention to not dividing by zero
    A2 = (Y3-Y4)/(X3-X4)  # Pay attention to not dividing by zero
    b1 = Y1-A1*X1 # = Y2-A1*X2
    b2 = Y3-A2*X3 # = Y4-A2*X4

    if (A1 == A2):
        print('parallel')
        return (False, None)  # Parallel segments

    #Ya = A1 * Xa + b1
    #Ya = A2 * Xa + b2
    #A1 * Xa + b1 = A2 * Xa + b2
    Xa = (b2 - b1) / (A1 - A2)   # Once again, pay attention to not dividing by zero
    Ya = A1 * Xa + b1

    if ( (Xa < max( min(X1,X2), min(X3,X4) )) or (Xa > min( max(X1,X2), max(X3,X4) )) ):
        return (False, None)  # intersection is out of bound
    else:
        return (True, np.array([Xa, Ya]))

def deg(theta_rad):
    return theta_rad*180/math.pi

def rad(theta_deg):
    return theta_deg*math.pi/180

class Sectormagnet_zx():

    def __init__(self, 
        name, 
        R, 
        angle, 
        p, 
        ccs_1_name, 
        ccs_1_origin, 
        ccs_1_angle, 
        ds=0, 
        s_ccs_1=0, 
        ccs_2_name=None, 
        q = 'e',
        phi_in=0,
        phi_out=0,
        b1=0, 
        b2=0, 
        dl=0,
        dipole_width=0.2):

        assert np.abs(angle)>0 and np.abs(angle)<=90, 'Bend angle must be 0 < abs(angle) <= 90'
        assert R>0, 'Bend radius must be > 0, if you set it negative, check the angle.'
        assert np.abs(phi_in) < 90, 'Entrance edge angle must be < 90'
        assert np.abs(phi_out) < 90, 'Entrance edge angle must be < 90'
        assert dipole_width < R, 'Dipole width must be < R'

        self.name = name

        self.s_ccs_1 = s_ccs_1
        self.ds = ds

        if(q == 'e'):
            self.q = 1.60217662e-19
        else:
            self.q = q
            
        self._p = p
        self._R = R
        self._B = p/R/c

        self._angle = rad(angle)    # input angle [deg], self.angle in [rad]
        self._phi_in = rad(phi_in)
        self._phi_out = rad(phi_out)
        self._L0 = R*self.angle



        self.b1 = b1
        self.b2 = b2
        self.dl = dl

        self.ccs_1_name = ccs_1_name

        if( isinstance(ccs_1_origin, list)):
            ccs_1_origin = np.array([np.array(ccs_1_origin)]).T

        self.ccs_1_origin = ccs_1_origin
        self.ccs_1_rotation_matrix = self.set_ccs_rotation_matrix(ccs_1_angle)

        self.ccs_2_name = ccs_2_name

        self.dipole_width = dipole_width

        self.set_entrance_point()
        self.set_exit_point()
        self.set_arc_points()

    @property
    def B(self):
        return self._B
    
    @property
    def R(self):
        return self._R

    @property
    def p(self):
        return self._p

    @property
    def angle(self):
        return deg(self._angle)

    @property
    def phi_in(self):
        return deg(self._phi_in)

    @property
    def phi_out(self):
        return deg(self._phi_out)

    def set_ccs_rotation_matrix(self, ccs_rotation):

        if(is_floatable(ccs_rotation)):

            print(f'Sectormagnet {self.name} CCS rotation matrix was specified by a single number: assuming this is the z-x rotation angle [deg]...')

            self.plane = 'zx'

            C = np.cos(ccs_rotation*math.pi/180)
            S = np.sin(ccs_rotation*math.pi/180)

            ccs_rotation = np.identity(3)

            ccs_rotation[0,0] = +C
            ccs_rotation[0,2] = -S
            ccs_rotation[2,0] = +S
            ccs_rotation[2,2] = +C

        elif(ccs_rotation.shape==(9,)):

            ccs_roation = np.reshape(ccs_rotation, (3, 3), order='F')

        return ccs_rotation

    def set_entrance_point(self):

        v1 = np.array([self.ccs_1_rotation_matrix[2,:]]).T
        self.p1 = self.ccs_1_origin + self.ds*v1

    def set_exit_point(self):

        v1 = np.array([self.ccs_1_rotation_matrix[2,:]]).T
        e1 = v1/np.linalg.norm(v1)

        M1 = self.rotation_matrix(-np.sign(self._angle)*90)
        self.R1 = self.R*np.matmul(M1,e1)

        M12 = self.rotation_matrix(self.angle)
        self.R2 = np.matmul(M12, self.R1)
        self.p2 = self.p1 + self.R2 - self.R1

    def set_arc_points(self):

        self.arc_s = self.s_ccs_1 + self.ds + np.linspace(0, self._angle*self.R, 100)
        self.arc_thetas = np.linspace(0, self.angle, 100)

        self.n1 = self.R1/norm(self.R1)
        self.n2 = self.R2/norm(self.R2)

        self.c = self.p1 - self.R*self.n1
        self.arc = self.get_arc(self.R, self.p1, self.n1, 0, self.angle)

        h1 = 1.5*self.dipole_width/np.cos(self._phi_in)
        h2 = 1.5*self.dipole_width/np.cos(self._phi_out)

        M1 = self.rotation_matrix(+self.phi_in)
        M2 = self.rotation_matrix(-self.phi_out)

        pa10 = self.p1 - self.dipole_width*self.n1
        pb10 = self.p1 + self.dipole_width*self.n1

        self.arc_a = self.get_arc(self.R-self.dipole_width, pa10, self.n1, 0, self.angle)
        self.arc_b = self.get_arc(self.R+self.dipole_width, pb10, self.n1, 0, self.angle)

        self.pa1 = self.p1 + h1*np.matmul(M1, self.n1)
        self.pb1 = self.p1 - h1*np.matmul(M1, self.n1)

        self.pa2 = self.p2 + h2*np.matmul(M2, self.n2)
        self.pb2 = self.p2 - h2*np.matmul(M2, self.n2)

        self.box = np.concatenate( (self.arc_a, np.fliplr(self.arc_b)), axis=1)
        self.box = np.concatenate( (self.box, np.array([self.box[:,0]]).T), axis=1)

    def rotation_matrix(self, theta, units='deg'):

        if(units=='deg'):
            theta = theta*math.pi/180

        C = np.cos(theta)
        S = np.sin(theta)

        M = np.identity(3)

        M[0,0] = +C
        M[0,2] = +S
        M[2,0] = -S
        M[2,2] = +C

        return M

    def get_arc(self, R, p1, n1, theta1, theta2, npts=100):

        thetas = np.linspace(theta1, theta2, npts)

        arc = np.zeros( (3, npts) )

        for ii, theta in enumerate(thetas):

            Mth = self.rotation_matrix(theta)
            Rii = np.matmul(Mth, R*n1)
            pii = p1 + Rii  - R*n1
            arc[:,ii] = np.squeeze(pii)

        return arc

    def plot_components(self):

        approach_z = np.array([self.ccs_1_origin[2], self.p1[2]])
        approach_x = np.array([self.ccs_1_origin[0], self.p1[0]])

        entrance_pole_face_z = [self.pa1[2,0], self.pb1[2,0]] 
        entrance_pole_face_x = [self.pa1[0,0], self.pb1[0,0]] 

        exit_pole_face_z = [self.pa2[2,0], self.pb2[2,0]] 
        exit_pole_face_x = [self.pa2[0,0], self.pb2[0,0]] 

        plot_components = {}

        plot_components['z'] = {
            'approach':approach_z,
            'box':self.box[2],
            #'entrance_face':entrance_face_z, 
            'entrance_pole_face':entrance_pole_face_z,
            #'exit_face':exit_face_z, 
            'exit_pole_face':exit_pole_face_z, 
            #'arc_a':self.arc_a[2], 
            #'arc_b':self.arc_b[2], 
            'arc':self.arc[2]
            }

        plot_components['x'] = {
            'approach':approach_x,
            'box':self.box[0],
            #'entrance_face':entrance_face_x, 
            'entrance_pole_face':entrance_pole_face_x,
            #'exit_face':exit_face_x, 
            'exit_pole_face':exit_pole_face_x, 
            #'arc_a':self.arc_a[0], 
            #'arc_b':self.arc_b[0], 
            'arc':self.arc[0]
            }

        return plot_components

    def plot(self, axis=None, alpha=1.0):

        components = self.plot_components()
        
        for comp in components['x']:

            if(comp=='arc' or comp=='approach' or comp=='entrance_pole_face' or comp=='exit_pole_face'):
                color='k'
            else:
                color='b'

            plt.plot(components['z'][comp], components['x'][comp], color, alpha=alpha)

        plt.plot(self.ccs_1_origin[2], self.ccs_1_origin[0], 'ob')
        plt.plot(self.p1[2], self.p1[0], 'og')
        plt.plot(self.p2[2], self.p2[0], 'or')

        plt.xlabel('z (m)')
        plt.ylabel('x (m)')

        if(axis=='equal'):
            plt.gca().set_aspect('equal')


    def gpt_lines(self, ):

        lines = []

        bname = self.name

        if(self.ccs_2_name is None):
            ccs_2_name = f'{self.name}_exit_ccs' 
  
        lines = lines + [f'#***********************************************']
        lines = lines + [f'#               Sectorbend: {self.name}         ']
        lines = lines + [f'#***********************************************']

        lines = lines + [f'{bname}_exit_x = {self.p2[0][0]};']    
        lines = lines + [f'{bname}_exit_y = {self.p2[1][0]};']  
        lines = lines + [f'{bname}_exit_z = {self.p2[2][0]};']      

        lines = lines + [f'{bname}_exit_theta = {self.angle}']

        exit_ccs_line = f'\nccs("{self.ccs_1_name}", {self.name}_exit_x, {bname}_exit_y, {bname}_exit_z'
        exit_ccs_line = exit_ccs_line + f', cos({bname}_exit_theta/deg), 0, -sin({bname}_exit_theta/deg), 0, 1, 0, "{ccs_2_name}");'

        lines = lines + [exit_ccs_line+'\n']
        
        lines = lines + [f'{bname}_radius = {self._R};']
        lines = lines + [f'{bname}_Bfield = {self._B};']
        lines = lines + [f'{bname}_phi_in = {self.phi_in};']    
        lines = lines + [f'{bname}_phi_out = {self.phi_out};'] 
        lines = lines + [f'{bname}_fringe_dl = {self.dl};'] 
        lines = lines + [f'{bname}_fringe_b1 = {self.b1};']    
        lines = lines + [f'{bname}_fringe_b2 = {self.b2};']  

        bend_line = f'\nsectormagnet("{self.ccs_1_name}", "{ccs_2_name}"'
        bend_line = bend_line + f', {bname}_radius, {bname}_Bfield, {bname}_phi_in/deg, {bname}_phi_out/deg'
        bend_line = bend_line + f', {bname}_fringe_dl, {bname}_fringe_b1, {bname}_fringe_b2);'

        lines = lines + [bend_line]

        return lines

        







        







