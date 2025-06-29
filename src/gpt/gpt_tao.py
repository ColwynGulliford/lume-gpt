from collections import Counter

from pmd_beamphysics import FieldMesh
from pmd_beamphysics.fields.analysis import accelerating_voltage_and_phase

#from . import GPT
from .element import Screen
from .bstatic import Bzsolenoid, Quadrupole, Sectormagnet
from .maps import Map2D_B, Map2D_E, Map25D_TM, Map3D_B, Map3D_E
from .lattice import Lattice

from scipy.constants import physical_constants

MC2 = physical_constants['electron mass energy equivalent in MeV'][0]*1e6


import numpy as np

def tao_unique_names(tao):
    """
    Invent a unique name

    Parameters
    ----------
    tao: Pytao.Tao instance

    Returns
    -------
    dict of int:str
        Mapping of ix_ele to a unique name
    """
    # Get all ixs
    ixs = set(tao.lat_list('*', 'ele.ix_ele'))
    ixs.update(set(tao.lat_list('*', 'ele.ix_ele', flags='-array_out -no_slaves')))
    ixs = list(sorted(ixs))
    
    names = [tao.ele_head(ix)['name'] for ix in ixs]
    
    count = Counter(names)
    unique_name = {}
    found = {name:0 for name in names}
    for ix, name in zip(ixs, names):
        if count[name] > 1:
            new_count = found[name]+1
            found[name] = new_count
            unique_name[ix] =  (f'{name}_{new_count}')
        else:
            unique_name[ix] = name
    return unique_name

def ele_info(tao, ele_id):
    """
    Returns a dict of element attributes from ele_head and ele_gen_attribs
    """
    edat = tao.ele_head(ele_id)
    edat.update(tao.ele_gen_attribs(ele_id))
    s = edat['s']
    
    if('L' in edat):
        L = edat['L']
    else:
        L=0
        
    edat['s_begin'] = s-L
    edat['s_center'] = (s + edat['s_begin'])/2    
    
    return edat


def pack_fieldmap(ele_id, tao):
    
    edat = ele_info(tao, ele_id)
    
    ekey = edat['key']
    ele_key = ekey.upper() 
    
    gpt_name = edat['name'].replace('.', '_')
    
    info = {}
    
    grid_params = tao.ele_grid_field(ele_id, 1, 'base')
    
    # Load the fieldmesh and save
    field_mesh = FieldMesh(str(grid_params['file']))  
    info['FieldMesh'] = field_mesh
    
    freq = edat.get('RF_FREQUENCY', 0)
    assert np.allclose(freq, field_mesh.frequency), f'{freq} != {field_mesh.frequency}'  
    
    master_parameter = grid_params['master_parameter']
    if master_parameter == '<None>':
        master_parameter = None
    
    if ele_key == 'E_GUN':
        zmirror = True
    else:
        zmirror = False
        
    L_fm = field_mesh.dz * (field_mesh.shape[2]-1)
    z0 = field_mesh.coord_vec('z')
    
    # Find zedge
    eleAnchorPt = field_mesh.attrs['eleAnchorPt']
    if eleAnchorPt == 'beginning':
        zedge = edat['s_begin']
        
    elif eleAnchorPt == 'center':
        # Use full fieldmap!!!
        zedge = edat['s_center'] + z0[0] # Wrong: -L_fm/2
    else:
        raise NotImplementedError(f'{eleAnchorPt} not implemented')
    
    info['z_edge'] = zedge
    
    # Phase and scale
    if ele_key == 'SOLENOID':
        assert  master_parameter is not None
        scale = edat[master_parameter]   
        
        bfactor = np.abs(field_mesh.components['magneticField/z'][0,0,:]).max() 
        if not np.isclose(bfactor, 1):
            scale *= bfactor
        phi0_tot = 0
        phi0_oncrest = 0
       
        info['gpt_element'] = Map2D_B(gpt_name, str(grid_params['file']), scale=scale, roll=edat['TILT'])
        
    elif ele_key in ('E_GUN', 'LCAVITY'):
        
        if master_parameter is None:
            scale = edat['FIELD_AUTOSCALE']
        else:
            scale = edat[master_parameter]
        
        Ez0 = field_mesh.components['electricField/z'][0,0,:]
        efactor = np.abs(Ez0).max()               
        #if not np.isclose(efactor, 1):
        #    scale *= efactor
        
        # Get ref_time_start
        ref_time_start = tao.ele_param(ele_id, 'ele.ref_time_start')['ele_ref_time_start']
        phi0_ref = freq*ref_time_start
        
        #phi0_fieldmap = field_mesh.attrs['RFphase'] / (2*np.pi) # Bmad doesn't use at this point
        phi0_fieldmap = grid_params['phi0_fieldmap']
        
        # Phase based on absolute time tracking
        phi0_user = sum([edat['PHI0'], edat['PHI0_ERR'] ])
        phi0_oncrest = sum([edat['PHI0_AUTOSCALE'], phi0_fieldmap, -phi0_ref]) 
        phi0_tot =  (phi0_oncrest + phi0_user) % 1
        theta0_deg = phi0_tot * 360  
        
        # Useful info for scaling
        acc_v0, acc_phase0 = accelerating_voltage_and_phase(z0, Ez0/np.abs(Ez0).max(), field_mesh.frequency)
        #print(f"v=c accelerating voltage per max field {acc_v0} (V/(V/m))")
        
        # Add phasing info
        info['v=c accelerating voltage per max field'] = acc_v0
        info['phi0_oncrest'] = phi0_oncrest % 1
        
        if(ele_key=='E_GUN'):
            info['gpt_element'] = Map2D_E(gpt_name, str(grid_params['file']), scale=scale, roll=edat['TILT'])
            
        elif(ele_key=='LCAVITY'):
            info['gpt_element'] = Map25D_TM(gpt_name, str(grid_params['file']), 
                                            scale = scale, 
                                            relative_phase = phi0_user*360,
                                            oncrest_phase = phi0_oncrest*360,
                                            frequency=freq,
                                            roll=edat['TILT'])

    elif ele_key in ['EM_FIELD']:

        if master_parameter is None:
            scale = edat['FIELD_AUTOSCALE']
        else:
            scale = edat[master_parameter]
            
        
        if str(grid_params['field_type'])=='Magnetic':
            #print('FOUND BSTATIC')
            #for k, v in edat.items():
            #    print(k)
            info['gpt_element'] = Map3D_B(gpt_name, str(grid_params['file']), scale=scale, roll=edat['TILT'])
            

    else:
        raise NotImplementedError
    
    return info


def pack_bend(ele_id, tao):

    edat = ele_info(tao, ele_id)
    gpt_name = edat['name'].replace('.', '_')
    
    gen_attrs = tao.ele_gen_attribs(ele_id)
    
    R = 1/np.abs(gen_attrs['G'])
    L = gen_attrs['L']
    theta = -(180/np.pi)*np.sign(gen_attrs['G'])*L/R
    e1 = -(180/np.pi)*gen_attrs['E1']
    e2 = +(180/np.pi)*gen_attrs['E2']
    p = np.sqrt(gen_attrs['E_TOT']**2 - MC2)
    
    s_beg = edat['s_begin']
    s_end = edat['s']
    
    return gpt_name, R, theta, p, e1, e2, s_beg, s_end


def pack_bmad_softedge_solenoid(ele_id, tao):
    
    edat = ele_info(tao, ele_id)
    gpt_name = edat['name'].replace('.', '_')
    
    gen_attrs = tao.ele_gen_attribs(ele_id)
    s_beg = edat['s_start']
    
    return gpt_name, gen_attrs['L_SOFT_EDGE'], gen_attrs['L'], gen_attrs['R_SOLENOID'], gen_attrs['BS_FIELD'], s_beg
    

def is_grid_field(ele_id, tao):
    
    try:
        tao.ele_grid_field(ele_id, 1, 'base')
        return True
    except:
        return False
    
    
def tao_create_gpt_lattice_def(tao,
                               solrf_eles=['E_Gun', 'Solenoid', 'Lcavity', 'EM_Field'], 
                               marker_eles = ['Marker'], #'MARKER::*',
                               quadrupole_eles = ['Quadrupole'], # Quads not implented
                               bend_eles = ['SBend'] # Bends not implemented
                              ):
    
    # Get unique name dict
    unique_name = tao_unique_names(tao)
    
    
    
    

    # Extract elements to use
    all_eles_wild_card_str = '::*,'.join(solrf_eles) + '::*,' + '::*,'.join(marker_eles) + '::*,'.join(bend_eles)+'::*'
    
    #print(all_eles_wild_card_str)
                            
    ele_ixs = tao.lat_list('*', 'ele.ix_ele', flags='-array_out -no_slaves')
    
    lat = Lattice('tao2gpt-lat')

    last_bend_name = 'beg'
    last_bend_s = 0
    
    
    
    for ii, ele_ix in enumerate(ele_ixs):
        
        ele_inf = ele_info(tao, ele_ix)

        #print(ii, ele_ix, ele_inf['key'], ele_inf['key'] in bend_eles, f"'{ele_inf['key']}'")
        
        if(ele_inf['key'] in solrf_eles and is_grid_field(ele_ix, tao)):    

            # Extract additional parameters required from Tao to define the map for GPT
            ele_inf = {**ele_inf, **pack_fieldmap(ele_ix, tao)}

            lat.add( ele_inf['gpt_element'], ds = ele_inf['z_edge'] - last_bend_s, element_origin='beg', ref_element=last_bend_name)
            
        elif(ele_inf['key'] == 'Solenoid'):   
            
            gpt_name, LSE, L, R, B, s_beg = pack_bmad_softedge_solenoid(ele_ix, tao)
            
            if(LSE==0):
                print('Soft edge length was set to zero')
                LSE = L
            
            S = Bzsolenoid(gpt_name, LSE, R, 1, L)
            S.bs_field = B
            
            if(R==0):
                print('Fitting an effective soft-edge model to Bmad hard edge solenoid.') 
                S.R = 1e-2*L
                S.fit_hard_edge_model(B, L)
            
            lat.add(S, ds=s_beg - last_bend_s, ref_element=last_bend_name, element_origin='beg')

            
        elif(ele_inf['key'] in bend_eles):

            gpt_name, R, theta, p, e1, e2, s_beg, s_end = pack_bend(ele_ix, tao)
            lat.add(Sectormagnet(gpt_name, R, theta, p, phi_in=e1, phi_out=e2), ds=s_beg - last_bend_s, ref_element=last_bend_name)

            last_bend_s = s_end
            last_bend_name = gpt_name
            
        elif(ele_inf['key'] == 'Marker' or ele_inf['key'] == 'Instrument' or ele_inf['key'] == 'Monitor'):
        
            gpt_name = ele_inf['name'].replace('.', '_')
            lat.add(Screen(gpt_name), ds=ele_inf['s'] - last_bend_s, ref_element=last_bend_name)
            
        elif(ele_inf['key'] == 'Quadrupole'):
            
            gpt_name = ele_inf['name'].replace('.', '_')
            gen_attrs = tao.ele_gen_attribs(ele_ix)

            L, G = gen_attrs['L'], gen_attrs['B1_GRADIENT']

            if(gpt_name in lat.names):
                print(f'Warning: duplicate element found for {gpt_name}, appending tao ele_id to gpt name.')
                gpt_name = gpt_name+f'{ele_ix}'

            # Fit hard edge?
            b1 = 0 # TODO

            #print(ele_
            
            lat.add(Quadrupole(gpt_name, G, L, b1=b1), ds=ele_inf['s']-L - last_bend_s, 
                    ref_element=last_bend_name,
                   )
            
            
            
            
            
            
            
            
         
    return lat

def gpt_from_tao(gpt_object, tao, workdir=None, use_tempdir=True, legacy_phasing=True):
    
    """
    Create a complete GPT object from a running Pytao Tao instance.

    Parameters
    ----------
    tao: Tao object

    Returns
    -------
    gpt_object: GPT
        Converted GPT object
    """
    
    lattice = tao_create_gpt_lattice_def(tao)
    
    template_dir, output_file = lattice.create_template_dir(template_dir=workdir, 
                                                            legacy_phasing=legacy_phasing)
    
    gpt_object.__init__(output_file, parse_layout=False)
    gpt_object.lattice = lattice
    
    zstop = max([ele.s_end for ele in gpt_object.lattice]) + 0.1
    gpt_object.set_variable('ZSTOP', zstop+0.1)
    
    tmax = 1.1*zstop/3e8
    gpt_object.set_variable('tmax', tmax)
    
    return gpt_object




    
    


    
    
    
    
    