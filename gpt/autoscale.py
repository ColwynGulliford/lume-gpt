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

def autophase1(lattice, t=0, p=1e-15, workdir=None, ztrack1_through=True, verbose=True, n_screen=200):

    ts=[]
    ps=[]
    zs=[]

    runs = []

    rf_elements = [element for element in lattice._elements if(element.type in ['Map1D_TM', 'Map25D_TM'])]

    if(len(rf_elements)<1):
        if(verbose):
            print('autophase1: no cavities to phase')
        return

    # Check that rf_elements do not overlap:
    for ii, cav in enumerate(rf_elements[:-1]):
        if(ii+1<len(rf_elements)):
            next_cav = rf_elements[ii+1]
            assert cav.z_end_ccs <= next_cav.z_beg_ccs, f'Autophasing Error: cavities {cav.name} and {next_cav.name} overlap and cannot be phased.'

    current_t = t
    current_p = p
    current_z = lattice[0].s_beg

    ts.append(current_t)
    ps.append(current_p)
    zs.append(current_z)

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

    # Autophase first cavity
    for ii, rf_element in enumerate(rf_elements):
    
        assert np.abs(current_z - rf_element.z_beg_ccs)<1e-14, f'Error Phasing {rf_element.name}: particle was not located at cavity entrance.'

        if(verbose):
            print(f'\n> Phasing: {rf_element.name}')
            print(f'   t_beg = {current_t} sec.')
            print(f'   s_beg = {rf_element.s_beg} m.')
            print(f'   scale = {rf_element._scale}.')
            print(f'   relative phase = {rf_element._relative_phase} deg.')
    
        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)
    
        # phase
        t1 = time.time()
        run = rf_element.autophase(t=current_t, p=current_p)
        t2 = time.time()
    
        runs.append(run)

        current_t = run.screen[-1]['mean_t']
        current_p = run.screen[-1]['mean_p']
        current_z = run.screen[-1]['mean_z']
        
        if(verbose):
            print(f'\n   t_end = {current_t} m.')
            print(f'   s_end = {rf_element.s_end} m.')
            print(f'   oncrest phase = {rf_element.oncrest_phase}')
            print(f'   energy gain =  {p2e(current_p)-p2e(ps[-1]):0.3f} eV.')
            print(f'\n   Ellapsed time =  {t2-t1:0.3f} sec.')
    
        ts.append(current_t)
        ps.append(current_p)
        zs.append(current_z)

        if(ii+1 < len(rf_elements)):  # Track to next cavity
            next_rf_element=rf_elements[ii+1]
            msg = f'\n> Tracking: {rf_element.name}:{next_rf_element.name}'
        else:
            next_rf_element=None
            msg = f'\n> Tracking: {rf_element.name}:END'

        if(verbose):
            print(msg)
        
        fparticle = ztrack1_to_autoscale_element(lattice, current_t, current_p, current_z, next_rf_element, workdir=None)
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


def ztrack1_to_autoscale_element(lattice, t, p, z, autoscale_element=None, workdir=None):

    lat_temp = Lattice('temp')

    if(autoscale_element):
        stop_element_index = lattice.element_index(autoscale_element.name)
        lat_temp._elements = lattice._elements[:stop_element_index]
        z_stop = autoscale_element.z_beg_ccs
    else:
        lat_temp._elements = lattice._elements
        z_stop = lattice[-1].z_end_ccs

    if(workdir is None):
        tempdir = tempfile.TemporaryDirectory(dir=workdir)  
        gpt_file = os.path.join(tempdir.name, f'track_to_{autoscale_element}.gpt.in')
        workdir = tempdir.name

    else:

        gpt_file = os.path.join(workdir, f'{element.name}.gpt.in' )

    lat_temp.write_gpt_lines(ztrack1_template(gpt_file), output_file=gpt_file)

    G = GPT(gpt_file, workdir=workdir, use_tempdir=False)
    return G.track1_in_ccs(z, z_stop, pz0=p, t0=t)











