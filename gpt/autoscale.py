from gpt.element import Lattice
from gpt.element import Screen
from gpt.element import is_bend

from gpt import GPT

import numpy as np


def autoscale_lattice(lattice, initial_particles, space_charge=False):

    auto_elements = [ele for ele in lattice._elements if(is_bend(ele))]

    if(len(auto_elements)<1):
        print('autoscale_lattice: no bends present, job done.')

    runs = []

    last_auto_element = lattice._elements[0]  # Start with beg
    for ii, current_auto_element in enumerate(auto_elements):

        last_index = lattice.index(last_auto_element.name)
        current_index = lattice.index(current_auto_element.name)

        #---------------------------------------------------------------------------------
        # Track up to bend
        #---------------------------------------------------------------------------------
        lat_up_to = Lattice(f'Partial {lattice._name}, elements 1:{ii+1}')
        lat_up_to._elements = list(lattice._elements[last_index:current_index])
        
        ds = np.linalg.norm( last_auto_element.p_end - current_auto_element.p_beg)
 
        lat_up_to.add(Screen(f'{current_auto_element.name}_entrance_scr'), ds=ds, ref_element = last_auto_element.name)
        lat_up_to.write_gpt_lines(output_file='gpt.temp.in')

        G = GPT(input_file='gpt.temp.in', initial_particles=initial_particles)
        G.run()

        runs.append(G)

        #---------------------------------------------------------------------------------
        # Track up to bend
        #---------------------------------------------------------------------------------

        # Track through bend

        previous_bend = current_auto_element
        intial_particles = G.tout[-1]

    return runs

"""
    for ii, ele in enumerate(lattice._elements):

        lat = Lattice(f'Partial {lattice._name}, elements 1:{ii+1}')
        lat._elements = lattice[:ii+1]

        if(is_bend(ele)):

            lat = Lattice(f'Partial {lattice._name}, elements 1:{ii+1}')
            lat._elements = lattice[:ii+1]
            lat.add(Screen(f'{ele.name}_autoscale_scr'), ds=0.25)

            print(lat[-1])

        else: # Track

            if(not is_screen(ele)):

            if(ii<len(lattice._elements)):
            print(f'Tracking: {ele.name} -> {lattice_elements}')
            #final_particles.append(track)
            #print(lat[0].name)


def autoscale_bend(lattice, initial_particles, space_charge=False):
    pass
"""











