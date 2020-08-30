import numpy as np
import os
import math, cmath
from scipy.integrate import cumtrapz
from gpt.tools import is_floatable
from gpt.tools import cvector
from gpt.tools import rotation_matrix
from gpt.tools import deg, rad
from gpt.tools import get_arc
from gpt.tools import write_ecs
from gpt.tools import in_ecs
from gpt.template import basic_template

from matplotlib import pyplot as plt

from numpy.linalg import norm 

from gpt.element import SectorBend
from gpt.element import Element
from gpt.element import Quad
from gpt.element import Beg

from gpt import GPT

import tempfile

from pmd_beamphysics import single_particle


c = 299792458  # Speed of light 

class Sectormagnet(SectorBend):

    def __init__(self, 
        name, 
        R, 
        angle, 
        p,
        width=0.2, 
        height=0, 
        phi_in=0, 
        phi_out=0, 
        gap=None,
        b1=0,
        b2=0,
        dl=0,
        n_screens=0,
        species='electron',
        plot_pole_faces=True,
        color='r',
        fix=False,
        place=False
        ):

        assert np.abs(angle)>0 and np.abs(angle)<180, 'Bend angle must be 0 < abs(angle) < 180'
        assert R>0, 'Bend radius must be > 0, if you set it negative, check the angle.'
        assert np.abs(phi_in) <= 90, 'Entrance edge angle must be < 90'
        assert np.abs(phi_out) <= 90, 'Entrance edge angle must be < 90'
        assert width < R, 'Dipole width must be < R'
        assert n_screens>=0, 'Number of extra screens must be >= 0.'

        super().__init__(name, R, angle, width=width, height=0, phi_in=phi_in, phi_out=phi_out, M=np.identity(3), plot_pole_faces=True, color=color)

        self._type = 'Sectormagnet'

        if(species == 'electron'):
            self._q = -1.60217662e-19
        else:
            raise ValueError(f'Unknown particle type: {species}')

        self._species = species

        self._B = p/R/c
        self._p = p

        self._gap=gap

        if(gap == None):
            self._b1 = b1
        elif(gap>0 and gap < float('Inf')):
            self._b1 = 2/gap
        else:
            self._b1 = 0

        self._b2 = b2
        self._dl = dl

        self._n_screens = n_screens
        self._theta_screen=None
        self._s_screen=None

        self._fix = fix

        if(place):
            self.place()

    @property
    def p(self):
        return self._p

    @property
    def b1(self):
        return self._b1

    @property
    def b2(self):
        return self._b2

    @property
    def dl(self):
        return self._dl
    
    @property
    def phi_in(self):
        return self._phi_in

    @property
    def phi_out(self):
        return self._phi_out

    @property
    def s_screen(self):
        return self._s_screen

    @property
    def p_screen_center(self):
        return self._p_screen_center

    @property
    def p_screen_a(self):
        return self._p_screen_a

    @property
    def p_screen_b(self):
        return self._p_screen_b

    @property
    def n_screens(self):
        return self._n_screens

    @property
    def species(self):
        return self._species
    
    
    def place(self, previous_element=Beg(), ds=0):

        super().place(previous_element=previous_element, ds=ds)
        self.set_screens()
  
    def set_screens(self):

        e1 = self.e1_beg

        thetas = np.linspace(0, self._theta, self._n_screens)

        p_screen_a = np.zeros( (3,len(thetas)) )
        p_screen_b = np.zeros( (3,len(thetas)) )

        arc1_beg = self.p_beg + np.sign(self._theta)*(self._width/2)*cvector(self.e1_beg)
        self._p_screen_a = get_arc(self._R-self._width/2, arc1_beg, self.e1_beg, self._theta, npts=self._n_screens)

        arc2_beg = self.p_beg - np.sign(self._theta)*(self._width/2)*cvector(self.e1_beg)
        self._p_screen_b = get_arc(self._R+self._width/2, arc2_beg, self.e1_beg, self._theta, npts=self._n_screens )

        self._p_screen_center = get_arc(self._R, self.p_beg, self.e1_beg, self._theta, npts=self._n_screens )

        self._theta_screen = thetas
        self._s_screen = self.s_beg + np.linspace(0, self._length, self._n_screens)

    def plot_floor(self, axis='equal', ax=None):

        ax = super().plot_floor(axis=axis, ax=ax)
        for ii in range(len(self.p_screen_center[1,:])):
            ax.plot([self.p_screen_a[2,ii], self.p_screen_b[2,ii]], [self.p_screen_a[0,ii], self.p_screen_b[0,ii]], 'g')


    def plot_field_profile(self, ax=None, normalize=False):

        if(ax == None):
            ax = plt.gca()

        

    def gpt_lines(self):

        lines = []

        bname = self.name
  
        lines = lines + [f'\n#***********************************************']
        lines = lines + [f'#               Sectorbend: {self.name}         ']
        lines = lines + [f'#***********************************************']

        exit_ccs_line = f'\nccs("{self.ccs_beg}", {self.name}_end_x, {bname}_end_y, {bname}_end_z'

        if(self.ccs_beg=='wcs'):

            M = np.linalg.inv(self.M_end)

            lines = lines + [f'{bname}_end_x = {self.p_end[0][0]};']    
            lines = lines + [f'{bname}_end_y = {self.p_end[1][0]};']  
            lines = lines + [f'{bname}_end_z = {self.p_end[2][0]};']     

            exit_ccs_line = exit_ccs_line + f', {M[0,0]}, {M[0,1]}, {M[0,2]}, 0, 1, 0, "{self.ccs_end}");' 

        else:

            ds = np.linalg.norm((self._p_beg - self._ccs_beg_origin))

            ccs_beg_e1 = cvector([1,0,0])
            ccs_beg_e3 = cvector([0,0,1])
            p_beg_ccs = ds*ccs_beg_e3

            dM = rotation_matrix(self._theta)

            p_end_ccs = p_beg_ccs + np.sign(self._theta)*self.R*(ccs_beg_e1-np.matmul(dM, ccs_beg_e1))

            lines = lines + [f'{bname}_end_x = {p_end_ccs[0][0]};']    
            lines = lines + [f'{bname}_end_y = {p_end_ccs[1][0]};']  
            lines = lines + [f'{bname}_end_z = {p_end_ccs[2][0]};'] 

            dM_inv = np.linalg.inv(dM)

            exit_ccs_line = exit_ccs_line + f', {dM_inv[0,0]}, {dM_inv[0,1]}, {dM_inv[0,2]}, 0, 1, 0, "{self.ccs_end}");'

        lines = lines + [exit_ccs_line+'\n']
        
        lines = lines + [f'{bname}_radius = {self._R};']
        lines = lines + [f'{bname}_Bfield = {self._B};']
        lines = lines + [f'{bname}_phi_in = {self.phi_in};']    
        lines = lines + [f'{bname}_phi_out = {self.phi_out};'] 
        lines = lines + [f'{bname}_fringe_dl = {self.dl};'] 
        lines = lines + [f'{bname}_fringe_b1 = {self.b1};']    
        lines = lines + [f'{bname}_fringe_b2 = {self.b2};']  

        if(self._fix):
            btype = 'sectormagnet_fix'
        else:
            btype = 'sectormagnet'

        bend_line = f'\n{btype}("{self.ccs_beg}", "{self.ccs_end}"'
        bend_line = bend_line + f', {bname}_radius, {bname}_Bfield, {bname}_phi_in/deg, {bname}_phi_out/deg'
        bend_line = bend_line + f', {bname}_fringe_dl, {bname}_fringe_b1, {bname}_fringe_b2);'

        lines = lines + [bend_line]
        
        """
        p_end_ccs_beg = in_ecs(self.p_end, self._ccs_beg_origin, self.M_beg)

        for ii, theta in enumerate(self._theta_screen):

            dM = rotation_matrix(theta)
            pii_ccs_beg = in_ecs(cvector(self.p_screen_center[:,ii]), self._ccs_beg_origin, self.M_beg)

            if(np.abs(theta)<=np.abs(self._theta)/2.0):

                ccs_line = f'ccs("{self.ccs_beg}", {write_ecs(pii_ccs_beg, dM)}"{self.name}_scr_ccs_{ii+1}");'
                lines.append(ccs_line)

                scr_line = f'screen("{self.ccs_beg}", {write_ecs(pii_ccs_beg, dM)}0, "{self.name}_scr_ccs_{ii+1}");'
                lines.append(scr_line)

            if(np.abs(theta)>=np.abs(self._theta)/2.0):

                pii_ccs_end = in_ecs(pii_ccs_beg, p_end_ccs_beg, rotation_matrix(self._theta))

                Mii = np.matmul(dM, np.matmul( np.linalg.inv(self.M_end), self.M_beg))

                #print(pii_ccs_end.T, Mii[:,0].T, Mii[:,2].T)

                #print(self.name, Mii[:,0].T, Mii[:,2].T)

                #Mii = np.matmul(dM, self.M_end)
                #Mii = np.linalg.inv(np.matmul(dM, self.M_beg))

                #Mii = np.linalg.inv(dM)

                #print(pii.T, cvector(Mii[:,0]).T, cvector(Mii[:,2]).T)

                ccs_line = f'ccs("{self.ccs_end}", {write_ecs(pii_ccs_end, Mii)}"{self.name}_scr_ccs_{ii+1}");'
                #lines.append(ccs_line)

                scr_line = f'screen("{self.ccs_end}", {write_ecs(pii_ccs_end, Mii/2.0)}0, "{self.name}_scr_ccs_{ii+1}");'
                #lines.append(scr_line)
                """
                
        return lines

    def plot_fringe(self, y=0):

        if(self.b1!=0 or self.b2!=0):

            z = np.linspace(-10/self.b1, 10/self.b1, 100)

            f = (self.b1*z +self.b2*((z-self.dl)**2 - y**2))

            h = y*(self.b1 + 2*self.b2*(z-self.dl))

            C = np.cos(h)
            S = np.sin(h)

            E1 = np.exp(f)
            E2 = np.exp(2*f)

            D = 1 + 2*E1*C + E2

            By = self._B*(1 + E1*C)/D
            Bz = -self._B*E1*S/D

            fig, ax = plt.subplots(1, 2, constrained_layout=True)

            ax[0].plot(z, Bz)
            ax[0].set_xlabel('z (m)')
            ax[0].set_ylabel('$B_z$ (T)')

            ax[1].plot(z, By)
            ax[1].set_xlabel('z (m)')
            ax[1].set_ylabel('$B_y$ (T)')

        else:
            print('No fringe specified, skipping plot.')

    def track_ref(self, p0=1e-15, xacc=6.5, GBacc=5.5, dtmin=1e-14, dtmax=1e-10, Ntout=100):

        dz_ccs_beg = np.linalg.norm( self.p_beg - self._ccs_beg_origin )

        dz_fringe = 0

        if(np.abs(self._b1)>0):
            dz_fringe = 10.0/self._b1
        else:
            dz_fringe = 0

        settings={'xacc':xacc, 'GBacc':GBacc, 'dtmin':dtmin, 'dtmax':dtmax, 'Ntout':Ntout, 'ZSTART': -2*np.sign(dz_ccs_beg-dz_fringe)*dz_ccs_beg-dz_fringe}

        particle = single_particle(z=dz_ccs_beg-dz_fringe, pz=p0, t=0, weight=1, status=1, species=self.species)

        tfile = tempfile.NamedTemporaryFile()
        gpt_file = tfile.name

        self.write_element_to_gpt_file(basic_template(gpt_file))

        G = GPT(gpt_file, initial_particles=particle, ccs_beg=self.ccs_beg)
        G.set_variables(settings)
        G.track1_to_z(z_end=dz_fringe, ds=self.length + 2*dz_fringe, ccs_beg=self.ccs_beg, ccs_end=self.ccs_end, z0=dz_ccs_beg-dz_fringe, pz0=p0, species=self.species)

        #os.remove(gpt_file)

        return G

    def plot_field_profile(self, ax=None, normalize=False):

        if(ax == None):
            ax = plt.gca()

        s = getattr(self,'s')
        B = self.b_field(s)

        if(normalize):
            B = B/np.max(np.abs(B))

        s = s+0.5*(self.s_beg+self._s_end)

        ax.plot(s, B, self._color)
        ax.set_xlabel('s (m)')

        return ax

    def is_inside_field(self, s):

        if(self._b1!=0):
            inside = (np.abs(s) - (self.arc_length/2+10*self._gap))<=0
        else:
            inside = (np.abs(s) - (self.arc_length/2))<=0

        return inside

    def ds(self, sin):
        ds = np.abs(sin) - (self._dl + self.arc_length/2)
        return ds

    def b_field(self, s=None):

        if(s is None):
            s = getattr(self,'s')

        B = np.zeros(s.shape)

        # which z points are inside the field
        inside = self.is_inside_field(s)   

        p = self._b1*self.ds(s[inside])
        f = np.exp(p)
        B[inside] = self._B/(1+f)

        return B

    @property
    def s(self):
        return np.linspace(-self.arc_length, self.arc_length, 200)



class QuadF(Quad):

    def __init__(self, name, G, length, width=0.2, height=0, angles=[0,0,0], gap=None, b1=0, dl=0, npts=1000, color='b'):

        super().__init__(name, length, width=width, height=height, angles=angles, color=color)

        self._G = G

        if(gap == None):
            self._b1 = b1
        elif(gap>0 and gap < float('Inf')):
            self._b1 = 2/gap
        else:
            self._b1 = 0

        self._gap = gap

        self._dl=dl

        self._npts=npts

    def gpt_lines(self):

        lines = []

        name = self.name
  
        lines = lines + [f'\n#***********************************************']
        lines = lines + [f'#               Enge Quad: {self.name}         ']
        lines = lines + [f'#***********************************************']
        
        lines = lines + [f'{name}_gradient = {self._G};']
        lines = lines + [f'{name}_length = {self._length};']
        lines = lines + [f'{name}_fringe_dl = {self._dl};'] 
        lines = lines + [f'{name}_fringe_b1 = {self._b1};']    

        ds = np.linalg.norm( 0.5*(self.p_end + self.p_beg) - self._ccs_beg_origin) 

        lines.append(f'\nquadf("{self.ccs_beg}", 0, 0, {ds}, 1, 0, 0, 0, 1, 0, {name}_length, {name}_gradient, {name}_fringe_dl, {name}_fringe_b1);')

        return lines

    def plot_field_profile(self, ax=None, normalize=False):

        if(ax == None):
            ax = plt.gca()

        z = getattr(self,'z')
        G = getattr(self,'G')

        if(normalize):
            G = G/np.max(np.abs(G))

        s = z+0.5*(self.s_beg+self._s_end)

        ax.plot(s, G, self._color)
        ax.set_xlabel('s (m)')

        return ax

    def plot_fringe(self):

        if(self._b1!=0):

            z = np.linspace(-10/self._b1, 10/self._b1, 100)

            f = self._b1*(z-self._dl)

            plt.plot(z, self._G/(1+np.exp(f)))
            plt.xlabel('$\\Delta z$ (m)')
            plt.ylabel('$G$ (T/m)')

        else:
            print('No fringe specified, skipping plot.')

    def is_inside_field(self, z):

        if(self._b1!=0):
            inside = (np.abs(z) - (self.L/2+10*self._gap))<=0
        else:
            inside = (np.abs(z) - (self.L/2))<=0

        return inside

    def grad(self, z=None):

        if(z is None):
            z = getattr(self,'z')

        G = np.zeros(z.shape)

        # which z points are inside the field
        inside = self.is_inside_field(z)   

        p = self._b1*self.dz(z[inside])
        f = np.exp(p)
        G[inside] = self._G/(1+f)

        return G

    def dgrad_dz(self, z=None):

        if(z is None):
            z = getattr(self,'z')

        dGdz = np.zeros(z.shape)
        inside = self.is_inside_field(z)
        f = np.exp(self._b1*self.dz(z[inside]))
        dGdz[inside] = -np.sign(z[inside])*f*self._b1*self._G/(1+f)**2

        return dGdz

    def d2grad_dz2(self, z=None):

        if(z is None):
            z = getattr(self, 'z')

        d2Gdz2 = np.zeros(z.shape)

        inside = self.is_inside_field(z)

        a = self._b1
        dz = self.dz(z[inside])
        f = np.exp(self._b1*dz)
        D = (1+f)

        d2Gdz2[inside] = self._G*f*(f-1)*a**2 /(1+f)**3

        return d2Gdz2

    def dz(self, zin):
        dz = np.abs(zin) - (self._dl + self.L/2)
        return dz

    def plot(self, npts=101, ax=None, title=False):

        if(ax is None):
            ax = plt.gca()

        z = getattr(self,'z')
        G = getattr(self,'G')

        ax.plot(z, G)
        ax.set_xlabel('$\\Delta z$ (m)')
        ax.set_ylabel('G(z) (T/m)')

        if(title):
            ax.set_title(f'{self.name}: G0 = {self._G:.4f} T/m, Leff = {self.Leff:.4f} m, gap = {self._gap} m.' )

        return ax

    def plot_dGdz(self, ax=None):

        if(ax is None):
            ax = plt.gca()
        
        z = getattr(self,'z')
        dG = getattr(self,'dGdz')

        ax.plot(z, dG)
        ax.set_xlabel('z (m)')
        ax.set_ylabel("G'(z) ($T/m^2$)")

        return ax

    def plot_d2Gdz2(self, ax=None):

        if(ax is None):
            ax = plt.gca()

        z = getattr(self,'z')
        dG2 = getattr(self,'d2Gdz2')

        ax.plot(z, dG2)
        ax.set_xlabel('z (m)')
        ax.set_ylabel("G''(z) ($T/m^3$)")

        return ax

    @property
    def L (self):
        return self._length

    @property
    def dGdzs(self):
        return self._dGdzs

    @property
    def Leff(self):

        z = getattr(self,'z')
        G = self.grad(z)

        return np.trapz(G, z)/self._G

    @property
    def z(self):
        return np.linspace(-self.length, self.length, self._npts)

    @property
    def G(self):
        return self.grad()

    @property
    def dGdz(self):
        return self.dgrad_dz()

    @property
    def d2Gdz2(self):
        return self.d2grad_dz2()







        







