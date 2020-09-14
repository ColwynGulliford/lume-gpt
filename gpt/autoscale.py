from gpt import GPT
from gpt.element import Lattice
from gpt.element import Screen
from gpt.element import is_bend
from  gpt.template import ztrack1_template

import tempfile
import os
import numpy as np
import time 

from distgen.physical_constants import qe, c, MC2
MC2=MC2.magnitude
c = c.magnitude

def p2e(p):
    return np.sqrt(p**2 + MC2**2)


def autophase1(lattice, t=0, p=1e-15, z=None, workdir=None, ztrack1_through=True, verbose=True, n_screen=200):

    ts=[]
    ps=[]
    zs=[]

    runs = []

    rf_elements = [element for element in lattice._elements if(element.type in ['Map1D_TM', 'Map25D_TM'])]

    if(len(rf_elements)<1):

        if(verbose):
            print('autophase1: no cavities to phase')
            print(f'\n> Tracking: BEG:END')

        ts.append(t)
        ps.append(p)
        zs.append(lattice[0].z_beg_ccs)

        fparticle = ztrack1_to_autoscale_element(lattice, 0, p, lattice[0].z_beg_ccs, workdir=None)
        assert fparticle is not None, f'Particle tracking from BEG to END failed.'
        assert np.abs( fparticle.screen[-1]['mean_z']-lattice[-1].z_end_ccs ) < 1e-14, f'Error tracking to END: particle was not located at cavity entrance.'

        runs.append(fparticle)

        current_t = fparticle.screen[-1]['mean_t']
        current_p = fparticle.screen[-1]['mean_p']
        current_z = fparticle.screen[-1]['mean_z']

        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)
        runs.append(fparticle)

        if(verbose):
            print(f'   energy gain: {p2e(current_p)-p2e(p)} eV.')

        if(ztrack1_through):

            if(workdir is None):
                tempdir = tempfile.TemporaryDirectory(dir=workdir)  
                gpt_file = os.path.join(tempdir.name, f'track_lattice.gpt.in')
                workdir = tempdir.name

            else:

                gpt_file = os.path.join(workdir, f'gpt.temp.in' )

            lattice.write_gpt_lines(ztrack1_template(gpt_file), output_file=gpt_file)

            G = GPT(gpt_file, workdir=workdir, use_tempdir=False)
            G = G.track1_in_ccs(lattice[0].z_beg_ccs, lattice[-1].z_end_ccs, pz0=p, t0=t, n_screen=n_screen)
    
            return  (ts, ps, zs, runs, G)

    current_t = t
    current_p = p
    current_z = lattice[0].z_beg_ccs

    if(current_z < rf_elements[0].z_beg_ccs):

        if(verbose):
            print(f'\n> Tracking: BEG:{rf_elements[0].name}')

        fparticle = ztrack1_to_autoscale_element(lattice, current_t, current_p, current_z, rf_elements[0], workdir=None)
        assert fparticle is not None, f'Particle tracking from BEG to {rf_elements[0].name} failed.'
        assert np.abs( fparticle.screen[-1]['mean_z']-rf_elements[0].z_beg_ccs ) < 1e-14, f'Error tracking to {rf_elements[0].name}: particle was not located at cavity entrance.'

        runs.append(fparticle)

        current_t = fparticle.screen[-1]['mean_t']
        current_p = fparticle.screen[-1]['mean_p']
        current_z = fparticle.screen[-1]['mean_z']

        if(verbose):
            print(f'   energy gain: {p2e(current_p)-p2e(p)} eV.')

    # Check that rf_elements do not overlap:
    for ii, cav in enumerate(rf_elements[:-1]):
        if(ii+1<len(rf_elements)):
            next_cav = rf_elements[ii+1]
            assert cav.z_end_ccs <= next_cav.z_beg_ccs, f'Autophasing Error: cavities {cav.name} and {next_cav.name} overlap and cannot be phased.'

    # Autophase first cavity
    for ii, rf_element in enumerate(rf_elements):
    
        assert np.abs(current_z - rf_element.z_beg_ccs)<1e-14, f'Error Phasing {rf_element.name}: particle was not located at cavity entrance.'
    
        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)

        # phase
        t1 = time.time()
        run = rf_element.autophase(t=current_t, p=current_p, workdir=workdir)
        t2 = time.time()
    
        runs.append(run)

        current_t = run.screen[-1]['mean_t']
        current_p = run.screen[-1]['mean_p']
        current_z = run.screen[-1]['mean_z']
    
        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)

        if(ii+1 < len(rf_elements)):  # Track to next cavity
            next_rf_element=rf_elements[ii+1]
            msg = f'\n> Tracking: {rf_element.name}:{next_rf_element.name}'
        elif(rf_element.name!=lattice[-1].name):
            next_rf_element=None
            msg = f'\n> Tracking: {rf_element.name}:END'
        else:
            break

        if(verbose):
            print(msg)
        
        fparticle = ztrack1_to_autoscale_element(lattice, current_t, current_p, current_z, next_rf_element, workdir=workdir)
        runs.append(fparticle)
        
        if(next_rf_element):
            
            assert fparticle is not None, f'Particle tracking from {rf_element.name} to {rf_elements[ii+1].name} failed.'
            assert np.abs( fparticle.screen[-1]['mean_z']-rf_elements[ii+1].z_beg_ccs ) < 1e-14, f'Error Phasing {rf_element.name}: particle was not located at cavity entrance after tracking to cavity.'
        else:
            assert fparticle is not None, f'Particle tracking from {rf_element.name} to END failed.'

        current_t = fparticle.screen[-1]['mean_t']
        current_p = fparticle.screen[-1]['mean_p']
        current_z = fparticle.screen[-1]['mean_z']

        if(verbose):
            print(f'   energy gain: { (p2e(current_p)-p2e(ps[-1]))/p2e(ps[-1])} eV.')

    if(ztrack1_through):

        if(workdir is None):
            tempdir = tempfile.TemporaryDirectory(dir=workdir)  
            gpt_file = os.path.join(tempdir.name, f'track_lattice.gpt.in')
            workdir = tempdir.name

        else:

            gpt_file = os.path.join(workdir, f'gpt.temp.in' )

        lattice.write_gpt_lines(ztrack1_template(gpt_file), output_file=gpt_file)

        G = GPT(gpt_file, workdir=workdir, use_tempdir=False)
        G = G.track1_in_ccs(zs[0], lattice[-1].z_end_ccs, pz0=p, t0=t, n_screen=n_screen)
    
    return  (ts, ps, zs, runs, G)

def set_ztrack1(lattice, workdir=None):

    if(workdir is None):
        tempdir = tempfile.TemporaryDirectory(dir=workdir)  
        gpt_file = os.path.join(tempdir.name, f'track_to_{autoscale_element}.gpt.in')
        workdir = tempdir.name

    else:

        gpt_file = os.path.join(workdir, f'{element.name}.gpt.in' )

    lat_temp.write_gpt_lines(ztrack1_template(gpt_file), output_file=gpt_file)


def ztrack1_to_autoscale_element(lattice, t, p, z, autoscale_element=None, workdir=None, ccs='wcs'):

    lat_temp = Lattice('temp')

    if(autoscale_element):
        stop_element_index = lattice.element_index(autoscale_element.name)
        lat_temp._elements = lattice._elements[:stop_element_index]
        z_stop = get_auto_element_z_ccs_beg(autoscale_element)
        stop_name = autoscale_element.name
        
    else:
        lat_temp._elements = lattice._elements
        z_stop = lattice[-1].z_end_ccs
        stop_name = 'END'    

    if(workdir is None):
        tempdir = tempfile.TemporaryDirectory(dir=workdir)  
        gpt_file = os.path.join(tempdir.name, f'track_to_{stop_name}.gpt.in')
        workdir = tempdir.name

    else:

        gpt_file = os.path.join(workdir, f'track_to_{stop_name}.gpt.in' )

    lat_temp.write_gpt_lines(ztrack1_template(gpt_file), output_file=gpt_file)

    print(z, z_stop, ccs, gpt_file)

    G = GPT(gpt_file, workdir=workdir, use_tempdir=False, ccs_beg=ccs)
    return G.track1_in_ccs(z, z_stop, pz0=p, t0=t, ccs=ccs)


def autoscale1_ccs(lattice, t, p=1e15, workdir=None, ztrack1_through=True, verbose=True, n_screen=200):

    if(lattice[1].type not in ['Sectormagnet']):
        pass

def autoscale1(lattice, t=0, p=1e-15, workdir=None, ztrack1_through=True, verbose=True, n_screen=200):

    ts=[]
    ps=[]
    zs=[]
    ss=[]

    runs = []

    auto_elements = [element for element in lattice._elements if(element.type in ['Map1D_TM', 'Map25D_TM', 'Sectormagnet'])]

    if(len(auto_elements)<1):

        if(verbose):
            print('autoscale1: no cavities to phase')
            print(f'\n> Tracking: BEG:END')

        ts.append(t)
        ps.append(p)
        zs.append(lattice[0].z_beg_ccs)
        ss.append(lattice[0].s_beg)

        fparticle = ztrack1_to_autoscale_element(lattice, 0, p, lattice[0].z_beg_ccs, workdir=None)
        assert fparticle is not None, f'Particle tracking from BEG to END failed.'
        assert np.abs( fparticle.screen[-1]['mean_z']-lattice[-1].z_end_ccs ) < 1e-14, f'Error tracking to END: particle was not located at cavity entrance.'

        runs.append(fparticle)

        current_t = fparticle.screen[-1]['mean_t']
        current_p = fparticle.screen[-1]['mean_p']
        current_z = fparticle.screen[-1]['mean_z']
        current_s = fparticle.screen[-1]['mean_z']

        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)
        ss.append(current_s)
        runs.append(fparticle)

        if(verbose):
            print(f'   energy gain: {p2e(current_p)-p2e(p)} eV.')

        if(ztrack1_through):

            if(workdir is None):
                tempdir = tempfile.TemporaryDirectory(dir=workdir)  
                gpt_file = os.path.join(tempdir.name, f'track_lattice.gpt.in')
                workdir = tempdir.name

            else:

                gpt_file = os.path.join(workdir, f'gpt.temp.in' )

            lattice.write_gpt_lines(ztrack1_template(gpt_file), output_file=gpt_file)

            G = GPT(gpt_file, workdir=workdir, use_tempdir=False)
            G = G.track1_in_ccs(lattice[0].z_beg_ccs, lattice[-1].z_end_ccs, pz0=p, t0=t, n_screen=n_screen)
    
            return  (ts, ps, zs, ss, runs, G)

    # Only here if lattice contains rf_elements and bends

    # Check that auto_elements do not overlap:
    for ii, cav in enumerate(auto_elements[:-1]):
        if(ii+1<len(auto_elements)):
            next_cav = auto_elements[ii+1]
            assert cav.s_end <= next_cav.s_beg, f'Autophasing Error: cavities {cav.name} and {next_cav.name} overlap and cannot be phased.'

    current_t = t
    current_p = p
    current_z = lattice[0].s_beg
    current_s = lattice[0].s_beg

    ts.append(current_t)
    ps.append(current_p)
    zs.append(current_z)
    ss.append(current_s)

    current_ccs = lattice[0].ccs_beg

    if(current_z < auto_elements[0].z_beg_ccs):

        if(verbose):
            print(f'\n> Tracking: BEG:{auto_elements[0].name}')

        fparticle = ztrack1_to_autoscale_element(lattice, current_t, current_p, current_z, auto_elements[0], workdir=None)
        assert fparticle is not None, f'Particle tracking from BEG to {auto_elements[0].name} failed.'
        assert np.abs( fparticle.screen[-1]['mean_z']-auto_elements[0].z_beg_ccs ) < 1e-14, f'Error tracking to {auto_elements[0].name}: particle was not located at cavity entrance.'

        runs.append(fparticle)

        current_t = fparticle.screen[-1]['mean_t']
        current_p = fparticle.screen[-1]['mean_p']
        current_z = fparticle.screen[-1]['mean_z']
        current_s = fparticle.screen[-1]['mean_z']

        if(verbose):
            print(f'   energy gain: {p2e(current_p)-p2e(p)} eV.')

    # Autoscale first element
    for ii, auto_element in enumerate(auto_elements):

        #print(current_z, auto_element.z_beg_ccs)
    
        assert np.abs(current_z - get_auto_element_z_ccs_beg(auto_element))<1e-14, f'Error Phasing {auto_element.name}: particle was not located at cavity entrance.'
    
        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)

        # phase
        t1 = time.time()
        run = autoscale1_element(current_t, current_p, auto_element, verbose=True)
        current_ccs = auto_element.ccs_end
        t2 = time.time()
    
        runs.append(run)

        current_t = run.screen[-1]['mean_t']
        current_p = run.screen[-1]['mean_p']
        current_z = run.screen[-1]['mean_z']

        current_t = run.screen[-1]['mean_t']
        current_p = run.screen[-1]['mean_p']
        current_z = get_auto_element_z_ccs_end(auto_element)
        current_s = get_auto_element_s_end(auto_element)
    
        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)
        ss.append(current_s)

        if(ii+1 < len(auto_elements)):  # Track to next cavity
            next_auto_element=auto_elements[ii+1]
            msg = f'\n> Tracking: {auto_element.name}:{next_auto_element.name}'
        elif(auto_element.name!=lattice[-1].name):
            next_auto_element=None
            msg = f'\n> Tracking: {auto_element.name}:END'
        else:
            break

        if(verbose):
            print(msg)

        fparticle = ztrack1_to_autoscale_element(lattice, current_t, current_p, current_z, next_auto_element, workdir=workdir, ccs=current_ccs)
        runs.append(fparticle)

        if(next_auto_element):
            
            assert fparticle is not None, f'Particle tracking from {auto_element.name} to {auto_elements[ii+1].name} failed.'

            print(fparticle.screen[-1]['mean_z'], get_auto_element_z_ccs_beg(auto_elements[ii+1]))

            assert np.abs( fparticle.screen[-1]['mean_z']-get_auto_element_z_ccs_beg(auto_elements[ii+1]) ) < 1e-14, f'Error scaling {auto_element.name}: particle was not located at next element entrance after tracking.'
        else:
            assert fparticle is not None, f'Particle tracking from {auto_element.name} to END failed.'

        current_t = fparticle.screen[-1]['mean_t']
        current_p = fparticle.screen[-1]['mean_p']
        current_z = fparticle.screen[-1]['mean_z']

        if(verbose):
            print(f'   energy gain: { (p2e(current_p)-p2e(ps[-1]))/p2e(ps[-1])} eV.')

    if(ztrack1_through):

        if(workdir is None):
            tempdir = tempfile.TemporaryDirectory(dir=workdir)  
            gpt_file = os.path.join(tempdir.name, f'track_lattice.gpt.in')
            workdir = tempdir.name

        else:

            gpt_file = os.path.join(workdir, f'gpt.temp.in' )

        lattice.write_gpt_lines(ztrack1_template(gpt_file), output_file=gpt_file)

        G = GPT(gpt_file, workdir=workdir, use_tempdir=False)
        G = G.track1_in_ccs(zs[0], lattice[-1].z_end_ccs, pz0=p, t0=t, n_screen=n_screen)
    
    return  (ts, ps, zs, ss, runs, G)


def get_auto_element_z_ccs_beg(auto_element):

    if(auto_element.type in ['Sectormagnet']):
        return auto_element.z_fringe_beg_ccs
        
    else:
        return auto_element.z_beg_ccs

def get_auto_element_z_ccs_end(auto_element):

    if(auto_element.type in ['Sectormagnet']):
        return auto_element.z_fringe_end_ccs
        
    else:
        return auto_element.z_end_ccs

def get_auto_element_s_beg(auto_element):

    if(auto_element.type in ['Sectormagnet']):
        return auto_element.s_fringe_beg
        
    else:
        return auto_element.s_beg

def get_auto_element_s_end(auto_element):

    if(auto_element.type in ['Sectormagnet']):
        return auto_element.s_fringe_end
        
    else:
        return auto_element.s_end


def autoscale1_element(t, p, auto_element, verbose=True, workdir=None):

    if(auto_element.type in ['Map1D_TM', 'Map25_TM']):
        run = auto_element.autophase(t, p, workdir=workdir)

    else:
        run = auto_element.autoscale(t, p, workdir=workdir)

    return run









