from pmd_beamphysics.units import nice_array, nice_scale_prefix

import matplotlib.pyplot as plt
import numpy as np 

#from .lattice import Lattice

def plot_stats_with_layout(gpt_object, ykeys=['sigma_x', 'sigma_y'], ykeys2=['mean_kinetic_energy'], 
                           xkey='mean_z', xlim=None, ylim=None, ylim2=None,
                           nice=True, 
                           include_layout=True,
                           include_labels=True, 
                           include_legend=True,
                           return_figure=False,
                           **kwargs):
    """
    Plots stat output multiple keys.
    
    If a list of ykeys2 is given, these will be put on the right hand axis. This can also be given as a single key. 
    
    Logical switches:
        nice: a nice SI prefix and scaling will be used to make the numbers reasonably sized. Default: True
        
        include_legend: The plot will include the legend.  Default: True
        
        include_layout: the layout plot will be displayed at the bottom.  Default: False
        
        include_labels: the layout will include element labels.  Default: True
        
        return_figure: return the figure object for further manipulation. Default: False

    """
    I = gpt_object # convenience
    
    if include_layout:
        fig, all_axis = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]        
        ax_plot = [all_axis[0]]
    else:
        fig, all_axis = plt.subplots( **kwargs)
        ax_plot = [all_axis]

     
 
    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_twinx = ax_plot[0].twinx()
        ax_plot.append(ax_twinx)



    # No need for a legend if there is only one plot
    if len(ykeys)==1 and not ykeys2:
        include_legend=False
    
    #assert xkey == 'mean_z', 'TODO: other x keys'
        
    X = I.stat(xkey)
    
    # Only get the data we need
    if xlim:
        good = np.logical_and(X >= xlim[0], X <= xlim[1])
        X = X[good]
    else:
        xlim = X.min(), X.max()
        good = slice(None,None,None) # everything 
        
    # X axis scaling    
    units_x = str(I.units(xkey))
    if nice:
        X, factor_x, prefix_x = nice_array(X)
        units_x  = prefix_x+units_x
    else:
        factor_x = 1   
    
    # set all but the layout
    for ax in ax_plot:
        ax.set_xlim(xlim[0]/factor_x, xlim[1]/factor_x)          
        ax.set_xlabel(f'{xkey} ({units_x})')    
    

    # Draw for Y1 and Y2 
    
    linestyles = ['solid','dashed']
    
    ii = -1 # counter for colors
    for ix, keys in enumerate([ykeys, ykeys2]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]
        
        # Check that units are compatible
        ulist = [I.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f'Incompatible units: {ulist[0]} and {u2}'
        # String representation
        unit = str(ulist[0])
        
        # Data
        data = [I.stat(key)[good] for key in keys]        
        
        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix+unit
        else:
            factor = 1

        # Make a line and point
        for key, dat in zip(keys, data):
            #
            ii += 1
            color = 'C'+str(ii)
            ax.plot(X, dat/factor, label=f'{key} ({unit})', color=color, linestyle=linestyle)
            
        ax.set_ylabel(', '.join(keys)+f' ({unit})')            
        #if len(keys) > 1:
        
        # Set limits, considering the scaling. 
        if ix==0 and ylim:
            new_ylim = np.array(ylim)/factor
            ax.set_ylim(new_ylim)
        # Set limits, considering the scaling. 
        if ix==1 and ylim2:
            pass
        # TODO
            if ylim2:
                new_ylim2 = np.array(ylim2)/factor
                ax_twinx.set_ylim(new_ylim2)            
            else:
                pass      
        
    # Collect legend
    if include_legend:
        lines = []
        labels = []
        for ax in ax_plot:
            a, b = ax.get_legend_handles_labels()
            lines += a
            labels += b
        ax_plot[0].legend(lines, labels, loc='best')        
    
    # Layout   
    if include_layout:
        print('inclucde layout!')
        #prit(new_)
        #ax_plot.set_xlim(xlim)
        gpt_object.lattice.plot_floor(axis=None, ax=ax_layout, style='tao')
        ax_layout.set_xlim(xlim)

    if return_figure:
        return fig          