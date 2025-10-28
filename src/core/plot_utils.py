import matplotlib.pyplot as plt
from metpy.plots import SkewT, Hodograph
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np

def prepare_skewt_hodograph(include_hodograph=True, include_indices_box=True, figsize=(12, 8)):
    """
    Prepare a figure with Skew-T diagram and optional hodograph/indices box.
    
    Parameters
    ----------
    include_hodograph : bool, optional
        If True, add a hodograph subplot (default: True)
    include_indices_box : bool, optional
        If True, add a text box for atmospheric indices (default: True)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 8))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    skew : metpy.plots.SkewT
        The Skew-T object for plotting
    h : metpy.plots.Hodograph or None
        The Hodograph object (None if include_hodograph=False)
    ax_indices : matplotlib.axes.Axes or None
        The axes for indices text (None if include_indices_box=False)
    """
    fig = plt.figure(figsize=figsize)
    
    # Create Skew-T
    skew = SkewT(fig, rotation=45, rect=(0.05, 0.05, 0.50, 0.90))
    
    # Set limits and add reference lines
    skew.ax.set_xlim(-20, 40)
    skew.ax.set_ylim(1050, 100)
    skew.plot_dry_adiabats(linewidths=0.8, alpha=0.5)
    skew.plot_moist_adiabats(linewidths=0.8, alpha=0.5)
    skew.plot_mixing_lines(linewidths=0.8, alpha=0.5)
    skew.ax.set_xlabel('Temperature (°C)', weight='bold')
    skew.ax.set_ylabel('Pressure (hPa)', weight='bold')
    
    # Optional: Create Hodograph
    h = None
    if include_hodograph:
        ax_h = fig.add_axes([0.54, 0.67, 0.28, 0.28])
        h = Hodograph(ax_h, component_range=15)
        h.add_grid(increment=5)
        
        ticks = np.arange(-15, 20, 5)
        h.ax.set_yticklabels([str(t) for t in ticks])
        h.ax.set_xticklabels([str(t) for t in ticks])
        h.ax.set_xticks(ticks)
        h.ax.set_yticks(ticks)
        h.ax.set_xlabel('U (m/s)', weight='bold', fontsize=9)
        h.ax.set_ylabel('V (m/s)', weight='bold', fontsize=9)
        h.ax.set_box_aspect(1)
        
        # Add range circles
        for i in range(10, 120, 10):
            h.ax.annotate(str(i), (i, 0), xytext=(0, 2), textcoords='offset pixels',
                         clip_on=True, fontsize=8, weight='bold', alpha=0.3, zorder=0)
    
    # Optional: Create indices text box
    ax_indices = None
    if include_indices_box:
        ax_indices = fig.add_axes([0.345, 0.11, 0.5, 0.5])
        ax_indices.axis('off')
    
    return fig, skew, h, ax_indices


def plot_sounding(fig, skew, h, ax_indices, sounding_data,
                 plot_temp=True,
                 plot_dewpoint=True,
                 plot_parcel=False,
                 plot_cape_cin=False,
                 plot_barbs=False,
                 plot_hodograph=False,
                 plot_indices_box=False,
                 color_temp='red',
                 color_dewpoint='green',
                 color_parcel='black',
                 alpha=1.0,
                 linewidth=1.5,
                 label=None,
                 label_fontsize=10,
                 show=False):
    """
    Plot a sounding on existing Skew-T axes with flexible options.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object
    skew : metpy.plots.SkewT
        The Skew-T object to plot on
    h : metpy.plots.Hodograph or None
        The Hodograph object (can be None)
    ax_indices : matplotlib.axes.Axes or None
        The axes for indices text (can be None)
    sounding_data : dict
        Dictionary containing 'p', 't', 'qv', 'height', 'u', 'v'
    plot_temp : bool, optional
        Plot temperature profile (default: True)
    plot_dewpoint : bool, optional
        Plot dewpoint profile (default: True)
    plot_parcel : bool, optional
        Plot parcel profile and mark LCL (default: False)
    plot_cape_cin : bool, optional
        Shade CAPE and CIN areas (default: False)
    plot_barbs : bool, optional
        Plot wind barbs (default: False)
    plot_hodograph : bool, optional
        Add to hodograph (default: False)
    plot_indices_box : bool, optional
        Display atmospheric indices in text box (default: False)
    color_temp : str, optional
        Color for temperature line (default: 'red')
    color_dewpoint : str, optional
        Color for dewpoint line (default: 'green')
    color_parcel : str, optional
        Color for parcel profile (default: 'black')
    alpha : float, optional
        Transparency (0-1) for all elements (default: 1.0)
    linewidth : float, optional
        Line width for T and Td (default: 1.5)
    label : str or None, optional
        Text label to add to the temperature line (e.g., '1', '2', '3')
    label_fontsize : int, optional
        Font size for the label (default: 10)
    show : bool, optional
        Call plt.show() at the end (default: False)
    
    Returns
    -------
    artists : dict
        Dictionary of matplotlib artists created (for later manipulation if needed)
    """
    # Extract and prepare data with units
    p = sounding_data['p'] * units.hPa
    t = sounding_data['t'] * units.kelvin
    t_degC = t.to('degC')
    z = sounding_data['height'] * units.meter
    u = sounding_data['u'] * units.meter / units.second
    v = sounding_data['v'] * units.meter / units.second
    qv = sounding_data['qv'] / 1000 * units('kg/kg')
    
    # Calculate dewpoint
    td = mpcalc.dewpoint_from_specific_humidity(p, qv)
    
    # Dictionary to store created artists
    artists = {
        'temp_line': None,
        'dewpoint_line': None,
        'parcel_line': None,
        'lcl_marker': None,
        'barbs': None,
        'cape_shade': None,
        'cin_shade': None,
        'hodograph_points': None
    }
    
    # Plot temperature
    if plot_temp:
        temp_line = skew.plot(p, t_degC, color=color_temp, alpha=alpha, linewidth=linewidth)[0]
        artists['temp_line'] = temp_line
        
        # Add label if provided
        if label is not None:
            # Find a good position for the label (at ~700 hPa)
            idx_label = np.argmin(np.abs(p.magnitude - 700))
            skew.ax.text(t_degC[idx_label].magnitude + 1, p[idx_label].magnitude,
                        label, fontsize=label_fontsize, weight='bold',
                        color=color_temp, alpha=alpha)
    
    # Plot dewpoint
    if plot_dewpoint:
        dewpoint_line = skew.plot(p, td, color=color_dewpoint, alpha=alpha, linewidth=linewidth)[0]
        artists['dewpoint_line'] = dewpoint_line
    
    # Plot parcel profile and LCL
    if plot_parcel:
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], t_degC[0], td[0])
        lcl_marker = skew.plot(lcl_pressure, lcl_temperature, 'ko', 
                              markerfacecolor='black', alpha=alpha)[0]
        
        prof = mpcalc.parcel_profile(p, t_degC[0], td[0]).to('degC')
        parcel_line = skew.plot(p, prof, color=color_parcel, linewidth=2, 
                               alpha=alpha, linestyle='--')[0]
        
        artists['parcel_line'] = parcel_line
        artists['lcl_marker'] = lcl_marker
    
    # Shade CAPE and CIN
    if plot_cape_cin and plot_parcel:
        prof = mpcalc.parcel_profile(p, t_degC[0], td[0]).to('degC')
        cape_shade = skew.shade_cape(p, t_degC, prof, alpha=0.3)
        cin_shade = skew.shade_cin(p, t_degC, prof, alpha=0.3)
        artists['cape_shade'] = cape_shade
        artists['cin_shade'] = cin_shade
    
    # Plot wind barbs
    if plot_barbs:
        mask = p >= 100 * units.hPa
        barbs = skew.plot_barbs(p[mask][::5], u[mask][::5], v[mask][::5], alpha=alpha)
        artists['barbs'] = barbs
    
    # Add to hodograph
    if plot_hodograph and h is not None:
        z_km = z.to('km')
        idx_10km = np.argmin(np.abs(z_km.m - 10))
        points = h.ax.scatter(u[0:idx_10km+1].m, v[0:idx_10km+1].m,
                            c=z_km[0:idx_10km+1].m, cmap='Spectral_r',
                            s=15, alpha=alpha)
        artists['hodograph_points'] = points
    
    # Display atmospheric indices
    if plot_indices_box and ax_indices is not None:
        # Calculate indices
        mucape, mucin = mpcalc.most_unstable_cape_cin(p, t_degC, td, depth=50 * units.hPa)
        sbcape, sbcin = mpcalc.surface_based_cape_cin(p, t_degC, td)
        lfc = mpcalc.lfc(p, t_degC, td)
        
        # Calculate bulk shear
        ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
        bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
        ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
        bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
        ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
        bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)
        
        # Display text
        ax_indices.text(0.48, 0.95,
                       f"   MUCAPE: {mucape.to('J/kg').m:.0f} J/kg\n\n\n\n\n\n\n\n\n\n",
                       color='crimson', weight='bold', verticalalignment='top',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='palegoldenrod',
                                             edgecolor='white', alpha=1))
        ax_indices.text(0.48, 0.895, f"   SBCAPE: {sbcape.to('J/kg').m:.0f} J/kg\n",
                       color='crimson', weight='bold', verticalalignment='top', fontsize=12)
        ax_indices.text(0.48, 0.835, f"   MUCIN: {mucin.to('J/kg').m:.0f} J/kg\n",
                       color='blue', weight='bold', verticalalignment='top', fontsize=12)
        ax_indices.text(0.48, 0.775, f"   SBCIN: {sbcin.to('J/kg').m:.0f} J/kg\n",
                       color='blue', weight='bold', verticalalignment='top', fontsize=12)
        ax_indices.text(0.48, 0.715,
                       f"   LFC: {lfc[0].m:.0f} hPa / {lfc[1].m:.1f} °C\n",
                       color='black', weight='bold', verticalalignment='top', fontsize=12)
        ax_indices.text(0.48, 0.655,
                       r"   $\bf{S_1}$" + f": {bshear1.to('m/s').m:.0f} m/s",
                       color='green', weight='bold', verticalalignment='top', fontsize=12)
        ax_indices.text(0.48, 0.595,
                       r"   $\bf{S_3}$" + f": {bshear3.to('m/s').m:.0f} m/s",
                       color='green', weight='bold', verticalalignment='top', fontsize=12)
        ax_indices.text(0.48, 0.535,
                       r"   $\bf{S_6}$" + f": {bshear6.to('m/s').m:.0f} m/s",
                       color='green', weight='bold', verticalalignment='top', fontsize=12)
    
    if show:
        plt.show()
    
    return artists


def plot_sounding_full(sounding_data, figsize=(12, 8), show=True):
    """
    Convenience function to plot a complete sounding with all diagnostics.
    
    This creates a new figure and plots temperature, dewpoint, parcel profile,
    CAPE/CIN shading, wind barbs, hodograph, and indices box all at once.
    
    Parameters
    ----------
    sounding_data : dict
        Dictionary containing 'p', 't', 'qv', 'height', 'u', 'v'
    figsize : tuple, optional
        Figure size (default: (12, 8))
    show : bool, optional
        Call plt.show() at the end (default: True)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    skew : metpy.plots.SkewT
    h : metpy.plots.Hodograph
    ax_indices : matplotlib.axes.Axes
    """
    fig, skew, h, ax_indices = prepare_skewt_hodograph(
        include_hodograph=True,
        include_indices_box=True,
        figsize=figsize
    )
    
    plot_sounding(
        fig, skew, h, ax_indices, sounding_data,
        plot_temp=True,
        plot_dewpoint=True,
        plot_parcel=True,
        plot_cape_cin=True,
        plot_barbs=True,
        plot_hodograph=True,
        plot_indices_box=True,
        show=show
    )
    
    return fig, skew, h, ax_indices


def plot_sounding_overlay(soundings_list, labels=None, colors=None, figsize=(12, 8), show=True):
    """
    Convenience function to overlay multiple soundings (temperature only).
    
    Useful for comparing parameter variations or ensemble members.
    
    Parameters
    ----------
    soundings_list : list of dict
        List of sounding dictionaries
    labels : list of str or None, optional
        Labels for each sounding (e.g., ['1', '2', '3'])
    colors : list of str or None, optional
        Colors for each sounding (default: use a color cycle)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    show : bool, optional
        Call plt.show() at the end (default: True)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    skew : metpy.plots.SkewT
    """
    fig, skew, _, _ = prepare_skewt_hodograph(
        include_hodograph=False,
        include_indices_box=False,
        figsize=figsize
    )
    
    # Default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(soundings_list)))
    
    # Default labels if not provided
    if labels is None:
        labels = [str(i+1) for i in range(len(soundings_list))]
    
    for i, sounding in enumerate(soundings_list):
        plot_sounding(
            fig, skew, None, None, sounding,
            plot_temp=True,
            plot_dewpoint=False,
            plot_parcel=False,
            plot_cape_cin=False,
            plot_barbs=False,
            plot_hodograph=False,
            plot_indices_box=False,
            color_temp=colors[i],
            alpha=0.7,
            linewidth=2,
            label=labels[i],
            show=False
        )
    
    if show:
        plt.show()
    
    return fig, skew