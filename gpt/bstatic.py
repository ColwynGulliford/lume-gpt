import numpy as np
from gpt.tools import cvector
from gpt.tools import rotation_matrix
from gpt.tools import get_arc
from gpt.tools import write_ecs
from gpt.tools import in_ecs
from gpt.element import p_in_ccs

from matplotlib import pyplot as plt


from gpt.element import SectorBend
from gpt.element import Element
from gpt.element import Quad
from gpt.element import Beg

#from . import GPT



from scipy.constants import physical_constants
mu0 = physical_constants['mag. constant'][0]
#c = physical_constants['c'][0]

from scipy.constants import c

from scipy import integrate 
from scipy import optimize

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
        n_screen=0,
        species='electron',
        plot_pole_faces=True,
        color='r',
        place=False):

        assert np.abs(angle)>0 and np.abs(angle)<180, 'Bend angle must be 0 < abs(angle) < 180'
        assert R>0, 'Bend radius must be > 0, if you set it negative, check the angle.'
        assert np.abs(phi_in) <= 90, 'Entrance edge angle must be < 90'
        assert np.abs(phi_out) <= 90, 'Entrance edge angle must be < 90'
        assert width < 2*R, 'Dipole width must be < R'
        assert n_screen>=0, 'Number of extra screens must be >= 0.'

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

        if gap is None:
            self._b1 = b1
        elif(gap>0 and gap < float('Inf')):
            self._b1 = 2/gap
        else:
            self._b1 = 0

        self._b2 = b2
        self._dl = dl

        self._n_screen = n_screen
        self._theta_screen=None
        self._s_screen=None

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

    @property
    def momentum(self):
        return self._p

    @momentum.setter
    def momentum(self, p):
        self._p = p
        self._B = p/self._R/c
    
    def place(self, previous_element=Beg(), ds=0, ref_origin='end', element_origin='beg'):

        """
        Places a sector bend in lattice, setting up required CCS
        """

        super().place(previous_element=previous_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)
        self.set_screens()
  
    def set_screens(self):

        e1 = self.e1_beg

        thetas = np.linspace(0, self._theta, self._n_screen)

        p_screen_a = np.zeros( (3,len(thetas)) )
        p_screen_b = np.zeros( (3,len(thetas)) )

        arc1_beg = self.p_beg + np.sign(self._theta)*(self._width/2)*cvector(self.e1_beg)
        self._p_screen_a = get_arc(self._R-self._width/2, arc1_beg, self.e1_beg, self._theta, npts=self._n_screen)

        arc2_beg = self.p_beg - np.sign(self._theta)*(self._width/2)*cvector(self.e1_beg)
        self._p_screen_b = get_arc(self._R+self._width/2, arc2_beg, self.e1_beg, self._theta, npts=self._n_screen)

        self._p_screen_center = get_arc(self._R, self.p_beg, self.e1_beg, self._theta, npts=self._n_screen)

        self._theta_screen = thetas
        self._s_screen = self.s_beg + np.linspace(0, self._length, self._n_screen)

    def plot_floor(self, axis='equal', ax=None, alpha=1, xlim=None, ylim=None, style='tao'):

        ax = super().plot_floor(axis=axis, ax=ax)
        for ii in range(len(self.p_screen_center[1,:])):
            ax.plot([self.p_screen_a[2,ii], self.p_screen_b[2,ii]], [self.p_screen_a[0,ii], self.p_screen_b[0,ii]], 'g')

        if(self._b1>0):

            M_fringe_beg  = rotation_matrix(+np.sign(self.angle)*self._phi_in)
            M_fringe_end  = rotation_matrix(-np.sign(self.angle)*self._phi_out)

            e1_fringe_beg = np.matmul(M_fringe_beg, self.e1_beg)
            e1_fringe_end = np.matmul(M_fringe_end, self.e1_end)

            p_fringe_beg_a = self.p_fringe_beg + (self._width/2.0)*e1_fringe_beg
            p_fringe_beg_b = self.p_fringe_beg - (self._width/2.0)*e1_fringe_beg

            p_fringe_end_a = self.p_fringe_end + (self._width/2.0)*e1_fringe_end
            p_fringe_end_b = self.p_fringe_end - (self._width/2.0)*e1_fringe_end

            ax.plot([p_fringe_beg_a[2,0], p_fringe_beg_b[2,0]], [p_fringe_beg_a[0,0], p_fringe_beg_b[0,0]], color='k', alpha=0.25)
            ax.plot([p_fringe_end_a[2,0], p_fringe_end_b[2,0]], [p_fringe_end_a[0,0], p_fringe_end_b[0,0]], color='k', alpha=0.25)

            #print(self.p_fringe_beg.T)


    def plot_field_profile(self, ax=None, normalize=False):

        if ax is None:
            ax = plt.gca()

    def gpt_lines(self):

        lines = []

        bname = self.name
  
        lines = lines + ['\n#***********************************************']
        lines = lines + [f'#               Sectormagnet: {self.name}         ']
        lines = lines + ['#***********************************************']

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

        btype = 'sectormagnet'

        bend_line = f'\n{btype}("{self.ccs_beg}", "{self.ccs_end}"'
        bend_line = bend_line + f', {bname}_radius, {bname}_Bfield, {bname}_phi_in/deg, {bname}_phi_out/deg'
        bend_line = bend_line + f', {bname}_fringe_dl, {bname}_fringe_b1, {bname}_fringe_b2);'

        lines = lines + [bend_line]
        
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
                lines.append(ccs_line)

                scr_line = f'screen("{self.ccs_end}", {write_ecs(pii_ccs_end, Mii/2.0)}0, "{self.name}_scr_ccs_{ii+1}");'
                lines.append(scr_line)

                
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
    """
    def track_ref(self, t0=0, p0=1e-15, xacc=6.5, GBacc=5.5, dtmin=1e-14, dtmax=1e-8, Ntout=100, workdir=None):

        dz_ccs_beg = np.linalg.norm( self.p_beg - self._ccs_beg_origin )

        dz_fringe = 0

        if(np.abs(self._b1)>0):
            dz_fringe = 10.0/self._b1
        else:
            dz_fringe = 0

        settings={'xacc':xacc, 'GBacc':GBacc, 'dtmin':dtmin, 'dtmax':dtmax, 'Ntout':Ntout, 'ZSTART': -2*np.sign(dz_ccs_beg-dz_fringe)*dz_ccs_beg-dz_fringe}

        particle = single_particle(z=dz_ccs_beg-dz_fringe, pz=p0, t=0, weight=1, status=1, species=self.species)

        if(workdir is None):
            tempdir = tempfile.TemporaryDirectory(dir=workdir)  
            gpt_file = os.path.join(tempdir.name, f'track_to_{self.name}.gpt.in')
            workdir = tempdir.name

        else:

            gpt_file = os.path.join(workdir, f'{self.name}.gpt.in' )

        self.write_element_to_gpt_file(basic_template(gpt_file))

        G = GPT(gpt_file, initial_particles=particle, ccs_beg=self.ccs_beg, workdir=workdir, use_tempdir=False)
        G.set_variables(settings)
        G.track1_to_z(z_end=dz_fringe, 
            ds=self.length + 2*dz_fringe, 
            ccs_beg=self.ccs_beg, 
            ccs_end=self.ccs_end, 
            z0=dz_ccs_beg-dz_fringe, 
            t0=t0,
            pz0=p0, 
            species=self.species,
            s_screen=self.s_end+dz_fringe)

        #os.remove(gpt_file)

        return G

    def exit_error(self, B, t0=0, p0=1e-15, xacc=6.5, GBacc=5.5, dtmin=1e-14, dtmax=1e-8, Ntout=100, workdir=None):

        self._B = B

        G = self.track_ref(p0=p0, t0=t0, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, workdir=workdir)

        assert G.n_screen > 0

        x_offset = G.screen[-1]['mean_x']
        x_angle = G.screen[-1]['mean_px']/G.screen[-1]['mean_pz']

        return np.abs(x_offset)

    def autoscale(self, t0=0, p0=None, xacc=6.5, GBacc=5.5, dtmin=1e-14, dtmax=1e-10, Ntout=100, workdir=None, track_through=True, verbose=True):

        if(p0 is None):
            p0 = self._p

        if(verbose):

            print(f'\n> Scaling: {self.name}')
            print(f'   t_beg = {t0} sec.')
            print(f'   s_beg = {self.s_beg} m.')
            print(f'   B-field = {self._B} T.')
            print(f'   momentum = {p0} eV/c.')
        
        if(p0 is None):
            p0 = self.momentum

        B0 = self._B
        f=0.1
        B = brent(lambda x: self.exit_error(x, t0=0, p0=p0, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, Ntout=Ntout, workdir=workdir), brack=( (1-f)*B0, (1+f)*B0))

        self._B = B

        if(track_through):

            G = self.track_ref(t0=0, p0=p0, xacc=xacc, GBacc=GBacc, dtmin=dtmin, dtmax=dtmax, workdir=workdir)

            position_err = np.sqrt( G.screen[-1]['mean_x']**2 + G.screen[-1]['mean_y']**2 )
            angle_err = np.sqrt( G.screen[-1]['mean_px']**2 + G.screen[-1]['mean_py']**2 )/G.screen[-1]['mean_pz']

            if(verbose):
                print(f'\n   B-field = {self._B} T.')
                print(f'   position error = {position_err*1e6} um.')
                print(f'   angle error = {angle_err*1e6} urad.')

            return G
    """

    def plot_field_profile(self, ax=None, normalize=False):

        if ax is None:
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


    @property
    def p_fringe_beg(self):

        if(self._b1>0):  
            return self.p_beg - 10/self._b1*self.e3_beg
        else:
            return self.p_beg


    @property
    def p_fringe_end(self):

        if(self._b1>0):  
            return self.p_end + 10/self._b1*self.e3_end
        else:
            return self.p_end

    @property
    def z_fringe_beg_ccs(self):
        return p_in_ccs(self.p_fringe_beg, self._ccs_beg_origin, self._M_beg)[2,0]

    @property
    def z_fringe_end_ccs(self):
        return p_in_ccs(self.p_fringe_end, self.p_end, self._M_end)[2,0]

    @property
    def s_fringe_beg(self):
        return self.s_beg - self.z_fringe_beg_ccs

    @property
    def s_fringe_end(self):
        return self.s_beg + self.z_fringe_end_ccs
    



class QuadF(Quad):

    def __init__(self, name, G, length, width=0.2, height=0, angles=[0,0,0], gap=None, b1=0, dl=0, npts=1000, color='b'):

        super().__init__(name, length, width=width, height=height, angles=angles, color=color)

        self._G = G

        if gap is None:
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
  
        lines = lines + ['\n#***********************************************']
        lines = lines + [f'#               Enge Quad: {self.name}         ']
        lines = lines + ['#***********************************************']
        
        lines = lines + [f'{name}_gradient = {self._G};']
        lines = lines + [f'{name}_length = {self._length};']
        lines = lines + [f'{name}_fringe_dl = {self._dl};'] 
        lines = lines + [f'{name}_fringe_b1 = {self._b1};']    

        ds = np.linalg.norm( 0.5*(self.p_end + self.p_beg) - self._ccs_beg_origin) 

        lines.append(f'\nquadf("{self.ccs_beg}", 0, 0, {ds}, 1, 0, 0, 0, 1, 0, {name}_length, {name}_gradient, {name}_fringe_dl, {name}_fringe_b1);')

        return lines

    def plot_field_profile(self, ax=None, normalize=False):

        if ax is None:
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


class Quadrupole(Quad):

    def __init__(self, name, G, length, width=0.2, height=0, angles=[0, 0, 0], gap=None, b1=0, npts=1000, color='b'):

        super().__init__(name, length, width=width, height=height, angles=angles, color=color)

        self._G = G

        if gap is None:
            self._b1 = b1
        elif(gap>0 and gap < float('Inf')):
            self._b1 = 2/gap
        else:
            self._b1 = 0

        self._gap = gap

        self._dl=0

        self._npts=npts

        self._type = 'quadrupole'

    def gpt_lines(self):

        lines = []

        name = self.name
  
        lines = lines + ['\n#***********************************************']
        lines = lines + [f'#               Enge Quad: {self.name}         ']
        lines = lines + ['#***********************************************']
        
        lines = lines + [f'{name}_gradient = {self._G};']
        lines = lines + [f'{name}_length = {self._length};']
        #lines = lines + [f'{name}_fringe_dl = {self._dl};'] 
        lines = lines + [f'{name}_fringe_b1 = {self._b1};']    

        ds = np.linalg.norm( 0.5*(self.p_end + self.p_beg) - self._ccs_beg_origin) 

        lines.append(f'\nquadrupole("{self.ccs_beg}", 0, 0, {ds}, 1, 0, 0, 0, 1, 0, {name}_length, {name}_gradient, {name}_fringe_b1);')

        return lines

    def plot_field_profile(self, ax=None, normalize=False):

        if ax is None:
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
    def G0(self):
        return self._G

    @G0.setter
    def G0(self, G0):
        self._G=G0

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
        return np.linspace(-2*self.length, 2*self.length, self._npts)

    @property
    def G(self):
        return self.grad()

    @property
    def dGdz(self):
        return self.dgrad_dz()

    @property
    def d2Gdz2(self):
        return self.d2grad_dz2()

    @property
    def int_G(self):
        return self.length * self.G0

        
    
    
    
class Bzsolenoid(Element):
    
    def __init__(self, name, L, R, nI, Lbound=None, width=None):
        
        if(Lbound is None):
            Lbound = L
            
        if(R==0):
            width = 0.1
        elif width is None:
            width = 2*R
        
        super().__init__(name, length=Lbound, width=width, height=0, angles=[0,0,0], color='k')
        
        self._L = L
        self._R = R
        self._nI = nI
    
    @property
    def L(self):
        return self._L
    
    @L.setter
    def L(self, L):
        self._L = L
    
    @property
    def R(self):
        return self._R
    
    @R.setter
    def R(self, R):
        self._R = R
    
    @property
    def nI(self):
        return self._nI
    
    def plot_Bz(self, ax=None):
        
        zpts = np.linspace(-3*self.L, 3*self.L, 1000)
        
        if(ax is None):
            ax = plt.gca()
            
        ax.plot(zpts, self.Bz(zpts)) 
        ax.set_xlabel('z (m)')
        ax.set_ylabel('$B_z(r=0)$ (T)')
        
        ax.plot(0, mu0*(self.L/2)/np.sqrt( (self.L/2)**2 + self.R**2 )*self.nI, '.')
            
    def Bz(self, z, r=0):
        
        if(isinstance(z, np.ndarray)):
        
            Bz = np.zeros(z.shape)

            is_inf = np.isinf(z)
            not_inf = ~is_inf

            zp = z[not_inf] + self.L/2
            zm = z[not_inf] - self.L/2

            Tzp = zp / np.sqrt(zp**2 + self.R**2)
            Tzm = zm / np.sqrt(zm**2 + self.R**2)

            Bz[not_inf] = 0.5*mu0*self.nI*(Tzp - Tzm)
            
        else:
            
            if(np.isinf(z)): return 0.0
            
            zp = z + self.L/2
            zm = z - self.L/2
            
            Tzp = zp / np.sqrt(zp**2 + self.R**2)
            Tzm = zm / np.sqrt(zm**2 + self.R**2)

            Bz = 0.5*mu0*self.nI*(Tzp - Tzm)
        
        return Bz
    
    def Bz2(self, z, r=0):
        return self.Bz(z, r=r)**2
        
        
    def Br(self, z, r):
        
        zp = z + self.L/2
        zm = z - self.L/2
        
        Trp = 1 / np.power(zp**2 + self.R**2, 3/2)
        Trm = 1 / np.power(zm**2 + self.R**2, 3/2)
        
        Br = -(r/4)*mu0*self.nI*self.R**2*(Trp-Trm)
        
        return Br
        
    
    def int_Bzdz(self, zmin=None, zmax=None):
        
        if(zmin is None):
            zmin = -np.inf
        
        if(zmax is None):
            zmax = +np.inf
        
        return integrate.quad(self.Bz, zmin, zmax)[0] 
    
    @property
    def intBzdz(self):
        return self.int_Bzdz()
    
    def int_Bz2dz(self, zmin=None, zmax=None):
        
        if(zmin is None):
            zmin = -np.inf
        
        if(zmax is None):
            zmax = +np.inf
            
        return integrate.quad(self.Bz2, zmin, zmax)[0] 
    
    @property
    def intBz2dz(self):
        return self.int_Bz2dz()
    
    @property
    def L_hard_edge(self):
        return self.intBzdz**2 / self.intBz2dz
        
    @property
    def max_abs_Bz(self):
        return np.abs(self.Bz(0))
    
    @property
    def B0(self):
        return self.Bz(0)
    
    @property
    def bs_field(self):
        return mu0*self.nI
    
    @bs_field.setter
    def bs_field(self, bsf):
        self._nI = bsf/mu0
   
    @max_abs_Bz.setter
    def max_abs_Bz(self, B):
        self._nI = (np.abs(B)/self.max_abs_Bz)*self.nI
        
    @B0.setter
    def B0(self, B):
        self._nI = (B/self.B0)*self.nI
        
        
    def fit_hard_edge_model(self, BHE, LHE, R=None):
        
        if(LHE<0):
            raise ValueError('Hardedge Solenoid length must be positive')

        BLHE, B2LHE = BHE*LHE, BHE**2 * LHE
        
        def chi2(alpha):
            
            B, L = alpha[0], np.abs(alpha[1])
            
            self.L = L
            self.B0 = B
            
            if(R is not None):
                self.R = R
            
            res = np.array([(self.intBzdz - BLHE)/BLHE, (self.intBz2dz-B2LHE)/B2LHE])
            
            return np.sum(res**2)
        
        
        bounds = ( (None, None), (0, None) )
        
        res = optimize.minimize(chi2, [BHE, LHE], bounds = bounds)
        
        self.B0 = res['x'][0]
        self.L = res['x'][1]
        
        return res
    
    
    def gpt_lines(self, ccs=None):

        lines = []

        name = self.name
  
        lines.append('\n#***********************************************')
        lines.append(f'#               Bzsolenoid: {self.name}         ')
        lines.append('#***********************************************')
        
        lines.append(f'{name}_R = {self._R};')
        lines.append(f'{name}_L = {self.L};')
        lines.append(f'{name}_bs_field = {self.bs_field};')
        lines.append(f'{name}_mu0 = {mu0};')
        lines.append(f'{name}_nI = {name}_bs_field/{name}_mu0;')      
        lines.append(f'{name}_x = 0;') 
        lines.append(f'{name}_y = 0;')
        
        ds = np.linalg.norm( 0.5*(self.p_end + self.p_beg) - self._ccs_beg_origin) 
        
        lines.append(f'{name}_z = {ds};') 

        lines.append(f'\nbzsolenoid("{self.ccs_beg}", {name}_x, {name}_y, {name}_z, 1, 0, 0, 0, 1, 0, {name}_R, {name}_L, {name}_nI);')
        
        return lines
                         
                         
    def fit_to_1d_map(self, z, Bz, set_results=True):
        
        intBzdz = np.trapz(Bz, z)
        intBz2dz = np.trapz(Bz**2, z)
        
        Leff = intBz2dz/intBzdz
        B = Bz[np.argmax(np.abs(Bz))]
        guess = [Leff,  0.1]
        
        def chi2(z, *alpha):
            
            L, R = np.abs(alpha)
            
            fsol = Bzsolenoid('fitsol', L, R, 1)
            fsol.B0 = B
            
            return fsol.Bz(z) 
        
        bounds = ( 0, np.inf )
            
        popt, pcov = optimize.curve_fit(chi2, z, Bz, guess, bounds=bounds)
        
        if(set_results):
            self.L = popt[0]
            self.R = popt[1]
            self.B0 = B
            
            
    def z_cutoff(self, f):
        
        Bf = f*self.B0
        
        res = optimize.brentq(lambda z: self.Bz(z) - Bf, 0, 100*self.L, maxiter=1000)
        
        return res
    
    def plot_floor(self, ax = None, axis=None, alpha=1, xlim=None, ylim=None, style=None):
        
        if(ax is None):
            ax = plt.gca()
        
        p1 = self.p_beg + (self._width/2)*cvector(self._M_beg[:,0])
        p2 = self.p_beg - (self._width/2)*cvector(self._M_beg[:,0])
        p3 = self.p_end + (self._width/2)*cvector(self._M_end[:,0])
        p4 = self.p_end - (self._width/2)*cvector(self._M_end[:,0])

        ps = np.concatenate( (p1, p3, p4, p2, p1), axis=1)
        
        ax.plot(ps[2], ps[0], self.color, alpha=0.1)
        ax.plot(ps[2], ps[0], self.color, alpha=0.1)
        
        dp = self.L*self.e3_beg
        
        pc = 0.5*(self.p_end + self.p_beg)
        
        p1 = pc + (self.R)*cvector(self._M_beg[:,0]) - 0.5*dp
        p2 = pc - (self.R)*cvector(self._M_beg[:,0]) - 0.5*dp
        p3 = pc + (self.R)*cvector(self._M_end[:,0]) + 0.5*dp
        p4 = pc - (self.R)*cvector(self._M_end[:,0]) + 0.5*dp
        
        ps = np.concatenate( (p1, p3, p4, p2, p1), axis=1)
        
        ax.plot(ps[2], ps[0], self.color, alpha=0.5)
        ax.plot(ps[2], ps[0], self.color, alpha=0.5)

        
        #ax.plot([p1o[2], p3o[2]], [p1o[0], p3o[0]], self.color, alpha=alpha)
        #ax.plot([p2o[2], p4o[2]], [p2o[0], p4o[0]], self.color, alpha=alpha)
    #p0 = (zL-element.z0[0])*element.e3_beg + element.p_beg + p00

    #p1 = p0 + (element._width/2)*element.e1_beg 
    #p2 = p1 + effective_plot_length*element.e3_beg
    #p3 = p2 - (element._width)*element.e1_beg
    #p4 = p3 - effective_plot_length*element.e3_beg

    #ps2 = np.concatenate( (p1, p2, p3, p4, p1), axis=1)

        
        
        
        
                         
    
                         

                         

    
   
        
        
        
        
    
        
        









        







