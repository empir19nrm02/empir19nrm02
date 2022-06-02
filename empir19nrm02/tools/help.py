from matplotlib import pyplot
import luxpy as lx
import numpy as np
from luxpy import _CMF, plot_spectrum_colors
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable

quantil = 0.05
dCutOff=0.003

strd = {
    'f1p': '$f_{1}^{´}$',
    'f1pE': '$f_{1,\mathrm{E}}^{´}$',
    'f1pLED': '$f_{1,\mathrm{L}}^{´}$',
    'f1pMin': '$f_{1,\mathrm{Min}}^{´}$',
    'f1pp': '$f_{1}^{´´}$',
    'f1ppR': '$f_{1,\mathrm{R}}^{´´}$',
    'f1pBW': '$f_{1,\mathrm{BW}}^{´}$',
    'q_plus_a': '$F^{a}_{i,q+}$',
    'srelLambda': '$s_{\mathrm{rel}}(\lambda)$',
    'SDLambda': '$S(\lambda)$',
    'xlambda': '$\lambda$ / nm',
    'pernm_e' : ' / \mathrm{nm^{-1}}',
    'smel_e': 's_{\mathrm{mel}}',
    'AU': ''}

fig_number = 1
fig_type= '.svg'
table_type= '.csv'

label_font_size=14


#pyplot.rcParams["figure.figsize"] = (7,7)
#pyplot.rcParams["figure.figsize"] = pyplot.rcParamsDefault["figure.figsize"]

def get_fig_file_name(dir=None, filename=None, table=False):
    global fig_number
    global fig_type
    if filename is None:
        file_name = dir + r'\Fig' + str(fig_number) + fig_type
        fig_number+=1
    else:
        if table:
            if 'xls' in filename:
                file_name = dir + r'\Table' + filename
            else:
                file_name = dir + r'\Table' + filename + table_type
        else:
            file_name = dir + r'\Fig' + filename + fig_type
    return file_name

def save_fig(dir = None, filename=None):
    pyplot.savefig( get_fig_file_name(dir=dir, filename=filename), bbox_inches='tight', pad_inches=0)
    pyplot.show()

def plot_cmf2( ax=None, name = '1931_2', cmf_symbols = ['x', 'y', 'z'], cmf_colors = ['r-', 'g-','b-'], single = False, spectrum_color = True):
    if ax is None:
        fig, ax = pyplot.subplots()
    if single:
        ax.plot(_CMF[name]['bar'][0], _CMF[name]['bar'][2], cmf_colors[0],  label='$'+ cmf_symbols[0] + '(\lambda)$')
    else:
        ax.plot(_CMF[name]['bar'][0], _CMF[name]['bar'][1], cmf_colors[0],  label='$\overline{'+cmf_symbols[0]+'}'+'(\lambda)$')
        ax.plot(_CMF[name]['bar'][0], _CMF[name]['bar'][2], cmf_colors[1],  label='$\overline{'+cmf_symbols[1]+'}'+'(\lambda)$')
        ax.plot(_CMF[name]['bar'][0], _CMF[name]['bar'][3], cmf_colors[2],  label='$\overline{'+cmf_symbols[2]+'}'+'(\lambda)$')

    if spectrum_color:
        if single:
            plot_spectrum_colors(spdmax=np.max(_CMF[name]['bar'][2]), axh=ax, wavelength_height=-0.05)
        else:
            plot_spectrum_colors(spdmax=np.max(_CMF[name]['bar'][3]), axh = ax, wavelength_height = -0.05)

    ax.set_xlabel(strd['xlambda'], fontsize=label_font_size)
    ax.set_ylabel('Sensitivity', fontsize=label_font_size)
    ax.legend()
    return ax

# modified from luxpy vlbar

def get_target(cieobs =lx._CIEOBS, target_index = 2, scr ='dict', wl_new = None, kind ='np', out = 1):
    """
    Get target functions.

    Args:
        :cieobs:
            | str, optional
            | Sets the type of Vlambda function to obtain.
        :target_index:
            | 1, 2 or 3, optional
            |   index of the CMF to return (1...X, 2...Y, 3...Z)
        :scr:
            | 'dict' or array, optional
            | - 'dict': get from ybar from _CMF
            | - 'array': ndarray in :cieobs:
            | Determines whether to load cmfs from file (./data/cmfs/)
            | or from dict defined in .cmf.py
            | Vlambda is obtained by collecting Ybar.
        :wl:
            | None, optional
            | New wavelength range for interpolation.
            | Defaults to wavelengths specified by luxpy._WL3.
        :kind:
            | str ['np','df'], optional
            | Determines type(:returns:), np: ndarray, df: pandas.dataframe
        :out:
            | 1 or 2, optional
            |     1: returns Vlambda
            |     2: returns (Vlambda, Km)

    Returns:
        :returns:
            | dataframe or ndarray with target function of type :cieobs:


    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    if scr == 'dict':
        dict_or_file = _CMF[cieobs]['bar'][[0,target_index],:]
        K = _CMF[cieobs]['K']
    elif scr == 'vltype':
        dict_or_file = cieobs #can be file or data itself
        K = 1
    Vl = lx.spd(data = dict_or_file, wl = wl_new, interpolation = 'linear', kind = kind, columns = ['wl','Vl'])

    if out == 2:
        return Vl, K
    else:
        return Vl

def label_management( locfig0, locax2nd, strColor2nd='blue'):
    lines_labels = [ax.get_legend_handles_labels() for ax in locfig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    locax2nd.spines['right'].set_color(strColor2nd)
    locax2nd.yaxis.label.set_color(strColor2nd)
    locax2nd.tick_params(axis='y', colors=strColor2nd)
    return [lines, labels]

def display_responsivity( name, detectors, cieobs='1931_2', s_target_index=2, out_dir = None, plots=['plot1', 'plot2']):
    if cieobs=='VS':
        cieobs='1951_20_scotopic'

    print( name)
    dl = lx.getwld(detectors[0])
    # LED_L41 is only available in a modified version of luxpy at the moment
    SC_L41_org = lx._CIE_ILLUMINANTS['LED_L41'].copy()
    # LED_L41 in the right wavelength resolution (of the detector set)
    SC_L41 = lx.cie_interp(SC_L41_org, detectors[0], negative_values_allowed=True, kind='linear')
    # target function in the right resolution
    target = get_target(cieobs=cieobs, target_index=s_target_index, wl_new= detectors[0])

    # Integral (target*L41)
    targetNorm = lx.utils.np2d(np.dot(target[1],dl*SC_L41[1]))
    # Integral (detector * L41)
    integralNorm = lx.utils.np2d(np.dot(detectors[1:],dl*SC_L41[1])).T
    # normalized detector responsivities
    detectorNorm=targetNorm/integralNorm*detectors[1:]
    # add the wavelength scale to the field
    detectorNorm = np.vstack((detectors[0], detectorNorm))

    if 'plot1' in plots:
        # plot all normalized detectors
        for i in range(1, detectorNorm.shape[0]):
            pyplot.plot(detectorNorm[0], detectorNorm[i])
        pyplot.plot(target[0], target[1], 'g-', label= r'target')
        pyplot.ylabel(strd['srelLambda'],fontsize=label_font_size)
        pyplot.xlabel(strd['xlambda'],fontsize=label_font_size)
        pyplot.legend()
        if out_dir is not None:
            save_fig( out_dir, name+'_all')

    if 'plot2' in plots:
        # plot the mean value with some statistical data
        fig, ax1 = pyplot.subplots()
        ax1.plot(detectorNorm[0], np.mean(detectorNorm[1:,:], axis=0), 'b', label= r'$\bar {s}_{\mathrm{rel}}^{\mathrm{L41}}(\lambda)$')
        ax1.fill_between(detectorNorm[0],np.max(detectorNorm[1:,:], axis=0), np.min(detectorNorm[1:,:],axis=0), alpha=0.2)
        ax1.fill_between(detectorNorm[0],np.quantile(detectorNorm[1:,:], 1-quantil/2, axis=0), np.quantile(detectorNorm[1:,:],quantil/2, axis=0), alpha=0.5)
        ax1.plot(target[0], target[1], 'g-', label= r'target')
        ax1.set_ylabel(strd['srelLambda'],fontsize=label_font_size)
        ax1.set_xlabel(strd['xlambda'],fontsize=label_font_size)

        ax2 = ax1.twinx()
        ax2.plot(detectorNorm[0], np.std(detectorNorm[1:,:], axis=0), 'r', label=r'$\sigma({s}_{\mathrm{rel}}^{\mathrm{L41}}(\lambda)$)')
        ax2.set_ylabel('$\sigma(s_{\mathrm{rel}}^{\mathrm{L41}}(\lambda))$',fontsize=label_font_size)


        [lines, labels]=label_management( fig, ax2, 'red')
        fig.legend(lines, labels, bbox_to_anchor=(0.65, 0.55, 0.4, 0.3), loc='upper left')
        if out_dir is not None:
            save_fig( out_dir, name+'_meansigma')

    # short evaluation with f1p values (calibration A, for the selected target function and observer)
    f1p=lx.spectral_mismatch_and_uncertainty.f1prime(detectors, S_C='A', cieobs=cieobs, s_target_index=s_target_index)
    print(f1p)

def plotCorrelation( image, wl_scale, name):
    fig, ax1 = pyplot.subplots(figsize=(7,7))
    im1 = ax1.imshow(image,
                 extent=[wl_scale[0], wl_scale[-1], wl_scale[-1], wl_scale[0]],
                 cmap="jet", interpolation="nearest")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    ax1.set_title(name)
    ax1.set_xlabel('$\lambda$ / nm', fontsize=label_font_size)
    ax1.set_ylabel('$\lambda$ / nm', fontsize=label_font_size)