#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 10:32:35 2021

@author: colwyngulliford
"""

from gpt.element import Beg
from gpt.element import is_bend

from gpt.template import BASIC_TEMPLATE
from gpt.tools import full_path
from gpt.tools import is_floatable
from gpt.maps import Map1D_E, Map1D_B, Map1D_TM, Map2D_E, Map2D_B, Map25D_TM
import numpy as np

from matplotlib import pyplot as plt
import os
import copy

import tempfile

class Lattice():

    def __init__(self, name, s=0, origin=[0,0,0], angles=[0,0,0]):

        self._name=name
        self._elements=[]
        self._ds=[]

        self._bends=[]

        self._elements.append(Beg(s, origin, angles))
        self._bends.append(Beg(s, origin, angles))
        
        self.template_dir = None

    #def get_element_ds(self, ds, ref_origin, p_beg_ref, p_end_ref, element_origin, element_length):

    def element_index(self, name):
        """
        Returns the array index of an element whose name is 'name':
        Inputs:
            name: str, desired element's name
        Outputs:
            integer, index of element in lattice._elements
        """
        for ii, element in enumerate(self._elements):
            if(element.name == name):
                return ii
        return []

    def sort(self):

        s = [0.5*(ele.s_beg+ele.s_end) for ele in self._elements]
        sorted_indices = sorted(range(len(s)), key=lambda k: s[k])
        self._elements = [self._elements[sindex] for sindex in sorted_indices]

    def add(self, element, ds, ref_element=None, ref_origin='end', element_origin='beg'):

        for ele in self._elements:
            assert ele.name != element.name, f'Lattice.add Error: cannot add elemnt with name = "{element.name}" to lattice, name already exists.'

        if(ref_element is None):
            ref_element = self._elements[-1]

        else:
            ref_element = self[ref_element]

        element.place(ref_element, ds=ds, ref_origin=ref_origin, element_origin=element_origin)
        self._elements.append(element)
        self.sort()

    def __getitem__(self, identity):

        if(isinstance(identity,int)):
            return self._elements[identity]
        elif(isinstance(identity, str)):
            for ele in self._elements:
                if(identity==ele.name):
                    return ele
            raise ValueError(f'No element in lattice with identity {identity}!')

        return self._elements[identity]

    def index(self, element_name):
        """
        Returns the array index of an element whose name is 'element_name':
        Inputs:
            element_name: str, desired element's name
        Outputs:
            integer or None, index of element in lattice._elements if found
        """
        for ii, element in enumerate(self._elements):
            if(element.name==element_name):
                return ii
        return None


    @property
    def s_ref(self):
        s = self._elements[0].s_ref 
        for ele in self._elements[1:]:
            s = np.concatenate( (s, ele.s_ref) )    
        return s

    @property
    def orbit_ref(self):
        p = self._elements[0].p_ref
        for ele in self._elements[1:]:
            p = np.concatenate( (p, ele.p_ref), axis=1 )
        return p

    @property
    def s_ccs(self):
        s = []
        for ele in self._elements:
            if(ele.type in ['Sectormagnet']):
                s = np.concatenate( (s, ele.s_screen))

        return s

    @property
    def s_beg(self):
        return self._elements[0].s_beg

    @property
    def s_end(self):
        return self._elements[-1].s_end

    def plot_floor(self, 
                   axis='equal', 
                   ax=None, 
                   box_on=True, 
                   labels_on=True, 
                   alpha=1,
                   xlim=None,
                   ylim=None,
                   style='tao'):

        """
        Plots the lattice in z-x floor coordinates 
        Inputs: 
            axis: str, if 'equal' makes the z/x scales the same
            ax: axes handle for plot, if None, generates new one
        Outputs:
            ax: axes handle used for plots
        """

        if(ax == None):
            ax = plt.gca()

        orbit = self.orbit_ref

        ax.plot(orbit[2], orbit[0], 'k')
        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')

        for ele in self._elements:
            ele.plot_floor(ax=ax, axis=axis, alpha=alpha, xlim=xlim, ylim=ylim, style=style)
            

        if(axis=='equal'):
            ax.set_aspect('equal')

        if(not box_on):
            ax.box(on=False)

        if(not labels_on):
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        return ax

    def plot_field_profile(self, ax=None, normalize=False):

        """
        Plots on axis field profiles of the lattice 
        Inputs: 
            normalize: boolean - normalize all field to a max of 1
            ax: axes handle for plot, if None, generates new one
        Outputs:
            ax: axes handle used for plots
        """

        if(ax == None):
            ax = plt.gca()

        for ele in self._elements:
            ele.plot_field_profile(ax=ax, normalize=normalize)

        if(normalize):
            ax.set_ylabel('Field Profile (arb.)')

        return ax

    def __str__(self):

        ostr = f'Lattice: {self._name}\ns-start: {self.s_beg} m.\ns-stop: {self.s_end}'
        for ele in self._elements:
            ostr = ostr+'\n\n'+ele.__str__()

        return ostr

    def get_M_at_s(self, s):
        pass

    def combine(self, lattice, ds=0):

        new_lattice = copy.deepcopy(self)
        new_lattice._name = new_lattice._name + '_and_' + lattice._name

        if(len(lattice._elements)<=1):
            return

        elements = lattice._elements[1:]
        ref_element = self._elements[-1]

        s_offset = ref_element.s_end

        for ii, element in enumerate(elements):

            if(ii>0):
                ds = s_offset + element.s_beg - ref_element.s_end
            else:
                ds = element.s_beg + ds

            new_element = copy.deepcopy(element)

            new_lattice.add(new_element, ds)
            ref_element = new_element

        return new_lattice

    def __add__(self, lattice):
        return self.combine(lattice)

    def write_gpt_lines(self, 
                        template_file=None, 
                        output_file=None, 
                        slices=None, 
                        legacy_phasing=False,
                        use_element_name_for_gdf_files=False
                       ):

        file_lines = []
        element_lines = []

        if(slices is None):
            elements = self._elements
        else:
            elements = self._elements[slices]

        # Write all the bend elements first
        for element in elements:
            if(is_bend(element)):
                element_lines.append(f'\n# {element.name}\n')
                for line in element.gpt_lines():
                    element_lines.append(line+'\n')

        for element in elements:
            if(not is_bend(element)):
                element_lines.append(f'\n# {element.name}\n')
                
                if(hasattr(element, 'source_data_file') and use_element_name_for_gdf_files):
                    new_ele_lines = element.gpt_lines(gdf_file=f'{element.name}.gdf')
                else:
                    new_ele_lines = element.gpt_lines()
                
                for line in new_ele_lines:
                    element_lines.append(line+'\n')

        if(template_file is None):
            #print('load basic template')
            file_lines = [line+'\n' for line in BASIC_TEMPLATE]

        else:

            assert os.path.exists(template_file), 'Template GPT file "template_file" does not exist.'

            with open(template_file,'r') as fid:
                for line in fid:
                    file_lines.append(line.strip())

        lines = file_lines + element_lines

        if(legacy_phasing):

            lines.append('\n#Legacy Phasing Lines\n')

            count=0
            for element in self._elements:

                #print(element)

                if(element._type in ['Map25D_TM', 'Map1D_TM']):

                    lines.append(f'phasing_amplitude_{count} = {element._name}_scale;\n')
                    lines.append(f'phasing_on_crest_{count} = {element._name}_oncrest_phase;\n')
                    lines.append(f'phasing_relative_{count} = {element._name}_relative_phase;\n')
                    lines.append(f'phasing_gamma_{count} = {element._name}_gamma;\n\n')
                    
                    count=count+1

        if(output_file is not None):
            
            with open(output_file,'w') as fid:
                for line in lines:
                    #print(line)
                    if(not line.endswith('\n')):
                        #print(line, 'fixing line')
                        line = line + '\n'
                    
                    fid.write(line)

        return lines


    @property
    def names(self):
        """
        Returns a list of element names
        """
        return [element.name for element in self._elements]


    def to_dict(self):
        return {ele._name:ele.to_dict() for ele in self._elements}

    def parse(self, gpt_file, style=None):
        
        abs_gpt_file = full_path(gpt_file)
        
        with open(gpt_file, 'r') as fid:
        
            lines = [line.strip().split('#')[0].strip() for line in fid.readlines() if(not line.strip().startswith('#') and line.strip()!='')]
            
            variables = { line.split("=")[0].strip():float(line.split("=")[1][:-1].strip()) for line in lines if( len(line.split("="))==2 and is_floatable(line.split("=")[1][:-1])) }
            
            map_lines = [line for line in lines if line.startswith('Map')]
            
            for mline in map_lines: 
                
                #try:
                self.parse_field_map(mline, variables, os.path.dirname(abs_gpt_file), style=style)
                #except:
                #    print(f'Could not parse: {mline}')
                    
            #fmap = [self.parse_field_map(mline, variables, os.path.dirname(abs_gpt_file), style=style) for mline in map_lines]
        


    def parse_field_map(self, mline, variables, gpt_file_dir, style=None):
        
        mtype = mline.split('(')[0]

        #print(mtype, gpt_file_dir)
        
        if(mtype not in ['Map1D_E', 'Map1D_B', 'Map1D_TM', 'Map2D_E', 'Map2D_B', 'Map25D_TM']):
            print(f'Unknown field map type: {mtype}')
            
        tokens = [t.strip() for t in mline.split(',')]
        tokens[0] = tokens[0].split('(')[1]
        tokens[-1] = tokens[-1].split(')')[0]

        #print(tokens)
        
        fmap_token = [token for token in tokens if('.gdf' in token)][0]
        fmap_token_index = tokens.index(fmap_token)
        fmap_file = os.path.expandvars(fmap_token[1:-1])
        
        fmap_name = os.path.basename(fmap_file).replace('.gdf','')

        
        if(not os.path.isabs(fmap_file)):
            
            fmap_file = os.path.abspath(os.path.join(gpt_file_dir, fmap_file))
            
        #print(fmap_file, mtype, fmap_token_index) 

        if(fmap_token_index==10 and tokens[0]=='"wcs"'):    
        
            zstr = tokens[3]
            if(is_floatable(zstr)):
                zpos=float(zstr)
            elif(zstr in variables):
                zpos=variables[zstr]
                
            else:
                print('Could not parse z-position for mline')

            #print(zpos)

            assert( int(tokens[4])==1 and #xhat = (1, 0, 0)
               int(tokens[5])==0 and 
               int(tokens[6])==0 and 
               int(tokens[7])==0 and #yhat = (0, 1, 0)
               int(tokens[8])==1 and 
               int(tokens[9])==0) 
            
            ele_name = f'ele_{len(self._elements) + 1}'

            #print(ele_name)
            
            if(mtype=='Map1D_B'):
                ele = Map1D_B(ele_name, fmap_file, style=style)
            elif(mtype=='Map2D_B'):
                ele = Map2D_B(ele_name, fmap_file, style=style)
            elif(mtype=='Map1D_TM'):
                ele = Map1D_TM(ele_name, fmap_file, 0, style=style)
            elif(mtype=='Map2D_E'):
                ele = Map2D_E(ele_name, fmap_file, style=style)
            elif(mtype=='Map25D_TM'):
                ele = Map25D_TM(ele_name, fmap_file, 0, style=style)
            else:
                print('Unknown element.')
                
            if(ele.z0[0]==0):
                ele_origin = 'beg'
            else:
                ele_origin = 'center'
                
            self.add(ele, ds=zpos, ref_element='beg', element_origin=ele_origin)
            
            
    def create_template_dir(self, 
                            template_dir=None, 
                            template_file=None, 
                            output_file=None, 
                            slices=None, 
                            legacy_phasing=False):
        
        if(template_dir==None):
            self.template_dir = tempfile.TemporaryDirectory()
            template_dir_str = str(self.template_dir.name)          
        else:
            self.template_dir = template_dir
            template_dir_str = template_dir
                
        if(output_file is None):
            output_file = f'{template_dir_str}/gpt.in'
            
        for ele in self._elements:
            
            if(hasattr(ele, 'source_data_file')):
                
                fields_dir_str = os.path.join(template_dir_str, 'fields')
                
                if(not os.path.exists(fields_dir_str)):
                    os.mkdir(fields_dir_str)
                
                gdf_file = os.path.join(fields_dir_str, ele.name+'.gdf')

                ele.write_gdf(gdf_file)
                ele.source_data_file=gdf_file.replace(f'{template_dir_str}/', '')
        
        lines = self.write_gpt_lines(template_file=template_file, 
                                     output_file=output_file,
                                     slices=slices,
                                     legacy_phasing=legacy_phasing,
                                     use_element_name_for_gdf_files=False)

        return template_dir, output_file
            

               
        
        
        
            